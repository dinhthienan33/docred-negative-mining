#!/usr/bin/env python3
"""
Training script for the DocRED SOTA pipeline.

Usage::

    python scripts/train.py --config configs/default.yaml [--overrides key=value ...]

The script supports:
  - Linear warmup + cosine decay scheduling
  - Gradient accumulation for large effective batch sizes
  - Mixed-precision (fp16) training via :mod:`torch.cuda.amp`
  - Separate learning rates for the PLM backbone vs GNN/head parameters
  - BMM warm-up phase (CE-only for the first N epochs)
  - Dev evaluation and best-model checkpointing every epoch
  - Early stopping based on dev Ign F1
  - Optional Weights & Biases logging
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from src.data.docred_dataset import DocREDDataset, docred_collate_fn
from src.models.pipeline import DocREDPipeline
from src.utils.load_dataset import ensure_docred_data_paths
from src.losses.joint_loss import JointLoss
from src.losses.evidence_negatives import (
    DocEvid,
    DocLabel,
    docred_collate_item_to_miner_format,
)
from src.utils.helpers import (
    set_seed,
    get_device,
    count_parameters,
    load_config,
    merge_config,
    create_logger,
    format_metrics,
    save_checkpoint,
)

logger = logging.getLogger("docred.train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Train the DocRED SOTA pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Optional config overrides in dot-notation, e.g. "
            "training.epochs=10 model.use_lora=true"
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; only evaluate the checkpoint specified by --resume.",
    )
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Apply CLI dot-notation overrides to a config dict.

    Args:
        config: Config dict (modified in-place).
        overrides: List of ``"key.sub_key=value"`` strings.

    Returns:
        Updated config dict.
    """
    for override in overrides:
        if "=" not in override:
            logger.warning("Ignoring invalid override (no '='): %s", override)
            continue
        key_path, _, raw_value = override.partition("=")
        keys = key_path.strip().split(".")
        # Attempt to cast to int / float / bool
        try:
            value: Any = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                if raw_value.lower() in ("true", "false"):
                    value = raw_value.lower() == "true"
                else:
                    value = raw_value

        node = config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
        logger.info("Override: %s = %s", key_path, value)
    return config


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------

def build_optimizer(
    model: DocREDPipeline,
    config: Dict[str, Any],
) -> AdamW:
    """Build an AdamW optimiser with separate per-group learning rates.

    PLM backbone parameters use a low learning rate to preserve pre-trained
    representations, while GNN and head parameters use a higher rate.

    Args:
        model: The full pipeline model.
        config: Training config sub-dict.

    Returns:
        Configured :class:`~torch.optim.AdamW` optimizer.
    """
    lr_plm = config["learning_rate_plm"]
    lr_other = config["learning_rate_other"]
    wd = config["weight_decay"]

    # Separate PLM parameters from GNN / head parameters
    plm_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder.") or name.startswith("plm."):
            plm_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": plm_params, "lr": lr_plm, "weight_decay": wd},
        {"params": other_params, "lr": lr_other, "weight_decay": wd},
    ]

    optimizer = AdamW(param_groups)
    logger.info(
        "Optimizer: PLM params=%d (lr=%g), other params=%d (lr=%g)",
        len(plm_params),
        lr_plm,
        len(other_params),
        lr_other,
    )
    return optimizer


def build_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_ratio: float,
):
    """Build a linear-warmup + cosine-decay LR scheduler.

    Args:
        optimizer: The optimiser whose LR will be scheduled.
        total_steps: Total number of optimizer steps across all epochs.
        warmup_ratio: Fraction of total steps devoted to linear warmup.

    Returns:
        HuggingFace cosine scheduler with warmup.
    """
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        "Scheduler: total_steps=%d, warmup_steps=%d", total_steps, warmup_steps
    )
    return scheduler


# ---------------------------------------------------------------------------
# Model batch (DocREDPipeline.forward expects a single dict, not **kwargs)
# ---------------------------------------------------------------------------

def _flatten_docred_labels_to_pairs(labels: torch.Tensor) -> torch.Tensor:
    """``[E, E, R]`` -> ``[E*(E-1), R]`` in the same pair order as ``DocREDPipeline._compute_pairs``."""
    e, _, r = labels.shape
    rows: List[torch.Tensor] = []
    for h in range(e):
        for t in range(e):
            if h == t:
                continue
            rows.append(labels[h, t])
    if not rows:
        return labels.new_zeros(0, r)
    return torch.stack(rows, dim=0)


def _evidence_to_pair_map(
    evidence: Dict[Tuple[int, int, int], List[int]],
) -> Dict[Tuple[int, int], List[int]]:
    """Merge DocRED ``(h,t,r) -> sents`` into ``(h,t) -> sents`` for the pipeline context path."""
    merged: Dict[Tuple[int, int], set] = defaultdict(set)
    for (h, t, _r_id), sents in evidence.items():
        merged[(h, t)].update(sents)
    return {k: sorted(v) for k, v in merged.items()}


def build_model_batch(
    batch: Dict[str, Any],
    device: torch.device,
    *,
    include_flat_labels: bool = True,
) -> Dict[str, Any]:
    """Build the ``batch`` dict for :meth:`DocREDPipeline.forward`.

    Collate uses ``sentence_boundaries``; the pipeline expects key ``sentences``.
    Per-document label tensors are flattened to ``[total_pairs, R]`` to match logits.
    """
    bsz = batch["input_ids"].size(0)
    evidence_maps: List[Dict[Tuple[int, int], List[int]]] = []
    for i in range(bsz):
        evidence_maps.append(_evidence_to_pair_map(batch["evidence"][i]))

    out: Dict[str, Any] = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "entity_spans": batch["entity_spans"],
        "mention_to_entity": batch["mention_to_entity"],
        "sentences": batch["sentence_boundaries"],
        "evidence_map": evidence_maps,
    }
    if include_flat_labels:
        flats = [_flatten_docred_labels_to_pairs(batch["labels"][i]) for i in range(bsz)]
        out["labels"] = torch.cat(flats, dim=0).to(device)
        # Per-doc [E,E,R] so the pipeline can align labels with ``max_pairs_per_doc`` truncation
        out["labels_per_doc"] = [batch["labels"][i].to(device) for i in range(bsz)]
    return out


def pack_joint_loss_inputs(
    model: DocREDPipeline,
    outputs_view1: Dict[str, Any],
    outputs_view2: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the dict expected by :class:`JointLoss` from two forward passes."""
    pair_embs = outputs_view1["pair_embs"]
    labels = outputs_view1["labels"]
    relation_ids = labels.argmax(dim=-1).long()

    co1 = model.get_contrastive_outputs(pair_embs, relation_ids)
    co2 = model.get_contrastive_outputs(outputs_view2["pair_embs"], relation_ids)

    epi = outputs_view1["entity_pair_ids"]
    entity_pair_ids = [(h, t) for (_d, h, t) in epi]

    return {
        "logits": outputs_view1["logits"],
        "labels": labels,
        "pair_embs": pair_embs,
        "contrastive_embs": co1["contrastive_embs"],
        "positive_contrastive_embs": co2["contrastive_embs"],
        "evidence_sets": outputs_view1["evidence_sets"],
        "entity_pair_ids": entity_pair_ids,
    }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_dev(
    model: DocREDPipeline,
    dev_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    train_triples: Optional[set] = None,
) -> Dict[str, float]:
    """Run inference on the dev set and return F1 / Ign F1 metrics.

    Args:
        model: Trained pipeline model.
        dev_loader: DataLoader for the dev split.
        device: Compute device.
        threshold: Sigmoid threshold for positive prediction.
        train_triples: Set of ``(title, h_idx, t_idx, r_str)`` tuples from
            the training set, used to compute Ign F1.

    Returns:
        Dict with keys ``"f1"``, ``"ign_f1"``, ``"precision"``, ``"recall"``.
    """
    # Import evaluate utilities lazily to avoid circular deps at top of script
    from scripts.evaluate import compute_f1, compute_ign_f1

    model.eval()
    all_preds: List[Dict[str, Any]] = []
    all_gold: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dev_loader:
            mb = build_model_batch(batch, device, include_flat_labels=False)
            outputs = model(mb)

            logits_all = outputs["logits"]  # [total_pairs, num_relations]
            pair_off = 0
            for title, hts, gold_labels in zip(
                batch["titles"],
                batch["hts"],
                batch["labels"],
            ):
                n_pairs = len(hts)
                doc_logits = logits_all[pair_off : pair_off + n_pairs]
                pair_off += n_pairs

                probs = torch.sigmoid(doc_logits)
                preds = (probs > threshold).cpu().numpy()

                # Collect predictions
                for pair_idx, (h, t) in enumerate(hts):
                    for r_id in range(1, preds.shape[1]):  # skip NA (id=0)
                        if preds[pair_idx, r_id]:
                            all_preds.append(
                                {
                                    "title": title,
                                    "h_idx": h,
                                    "t_idx": t,
                                    "r_id": r_id,
                                }
                            )

                # Collect gold labels
                num_ents = gold_labels.shape[0]
                for h in range(num_ents):
                    for t in range(num_ents):
                        if h == t:
                            continue
                        for r_id in range(1, gold_labels.shape[2]):
                            if gold_labels[h, t, r_id] > 0:
                                all_gold.append(
                                    {
                                        "title": title,
                                        "h_idx": h,
                                        "t_idx": t,
                                        "r_id": r_id,
                                    }
                                )

    f1, prec, rec = compute_f1(all_preds, all_gold)
    ign_f1, _, _ = compute_ign_f1(all_preds, all_gold, train_triples or set())

    model.train()
    return {"f1": f1, "ign_f1": ign_f1, "precision": prec, "recall": rec}


# ---------------------------------------------------------------------------
# Evidence miner statistics update
# ---------------------------------------------------------------------------

def update_evidence_stats(
    loss_fn: JointLoss,
    train_loader: DataLoader,
) -> None:
    """Update co-occurrence statistics in the evidence-aware miner.

    Iterates over training batches to collect relation co-occurrence and
    evidence overlap counts.  Call once before training begins (and
    optionally at the start of each epoch).

    Args:
        loss_fn: Joint loss instance containing the evidence miner.
        train_loader: DataLoader for the training split.
    """
    if not hasattr(loss_fn, "evidence_miner"):
        return
    miner = loss_fn.evidence_miner
    for batch in train_loader:
        batch_labels: List[List[DocLabel]] = []
        batch_evidence: List[List[DocEvid]] = []
        for labels, evidence in zip(batch["labels"], batch["evidence"]):
            doc_labs, doc_evid = docred_collate_item_to_miner_format(labels, evidence)
            batch_labels.append(doc_labs)
            batch_evidence.append(doc_evid)
        miner.update_statistics(batch_labels, batch_evidence)
    logger.info("Evidence miner co-occurrence statistics updated.")


# ---------------------------------------------------------------------------
# Training helpers (speed / AMP)
# ---------------------------------------------------------------------------

def _pipeline_cfg(full_config: Dict[str, Any]) -> Dict[str, Any]:
    """``DocREDPipeline`` expects the flat ``model`` section, not the whole YAML."""
    m = full_config.get("model")
    if isinstance(m, dict):
        return m
    return full_config


def _resolve_amp_dtype(train_cfg: Dict[str, Any], device: torch.device) -> Optional[torch.dtype]:
    """Pick autocast dtype: bf16 on supported GPUs when ``bf16: true``, else fp16."""
    if device.type != "cuda":
        return None
    if train_cfg.get("bf16", False) and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if train_cfg.get("fp16", False) or train_cfg.get("bf16", False):
        return torch.float16
    return None


def _make_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    train_cfg: Dict[str, Any],
    device: torch.device,
) -> DataLoader:
    nw = int(train_cfg.get("dataloader_num_workers", 4))
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": docred_collate_fn,
        "num_workers": nw,
        "pin_memory": device.type == "cuda",
    }
    if nw > 0 and train_cfg.get("dataloader_persistent_workers", False):
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: Dict[str, Any], resume_path: Optional[str] = None) -> None:
    """Main training function.

    Args:
        config: Full configuration dictionary.
        resume_path: Optional path to a checkpoint to resume training from.
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    train_cfg = config["training"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    log_cfg = config["logging"]

    output_dir = Path(log_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log = create_logger(output_dir)
    data_cfg = ensure_docred_data_paths(config["data"], log)
    config["data"] = data_cfg
    set_seed(train_cfg["seed"])
    device = get_device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Optional lightweight timing profiler for bottleneck diagnosis.
    profile_timing = bool(train_cfg.get("profile_timing", False))
    profile_batches = max(1, int(train_cfg.get("profile_batches", 100)))

    # Optional W&B (guarded by training.wandb)
    use_wandb = bool(train_cfg.get("wandb", False))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=log_cfg.get("wandb_project", "docred-gcl"),
                entity=log_cfg.get("wandb_entity"),
                config=config,
            )
            log.info("Weights & Biases logging enabled.")
        except Exception:
            use_wandb = False
            log.info("wandb not available or init failed; logging to file only.")
    else:
        log.info("Weights & Biases logging disabled by config (training.wandb=false).")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    log.info("Building datasets...")
    train_dataset = DocREDDataset(
        data_path=data_cfg["train_path"],
        tokenizer_name=model_cfg["plm_name"],
        max_length=data_cfg["max_length"],
        relation_map_path=data_cfg.get("rel_info_path"),
        use_entity_markers=data_cfg.get("use_entity_markers", True),
    )
    dev_dataset = DocREDDataset(
        data_path=data_cfg["dev_path"],
        tokenizer_name=model_cfg["plm_name"],
        max_length=data_cfg["max_length"],
        relation_map_path=data_cfg.get("rel_info_path"),
        use_entity_markers=data_cfg.get("use_entity_markers", True),
    )

    train_loader = _make_dataloader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        train_cfg=train_cfg,
        device=device,
    )
    dev_loader = _make_dataloader(
        dev_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        train_cfg=train_cfg,
        device=device,
    )
    log.info("Train docs: %d | Dev docs: %d", len(train_dataset), len(dev_dataset))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    log.info("Initialising DocREDPipeline...")
    model = DocREDPipeline(_pipeline_cfg(config))
    model.to(device)
    count_parameters(model)
    mp = model_cfg.get("max_pairs_per_doc", -1)
    if mp and mp > 0:
        log.info("max_pairs_per_doc=%d (reduces GNN/triple cost on entity-heavy docs)", mp)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    log.info("Initialising JointLoss...")
    loss_fn = JointLoss(
        num_relations=model_cfg["num_relations"],
        lambda_gcl=loss_cfg["lambda_gcl"],
        lambda_evidence=loss_cfg["lambda_evidence"],
        temperature=loss_cfg["contrastive_temperature"],
        bmm_warmup_epochs=loss_cfg["bmm_warmup_epochs"],
        bmm_update_every=loss_cfg["bmm_update_every"],
        # bmm_em_iters=loss_cfg["bmm_em_iters"],
        # num_hard_negatives=loss_cfg["num_hard_negatives"],
        # num_medium_negatives=loss_cfg["num_medium_negatives"],
        # num_easy_negatives=loss_cfg["num_easy_negatives"],
    )

    # Update evidence co-occurrence statistics from training data
    update_evidence_stats(loss_fn, train_loader)

    # Move loss (ATLOP threshold, BMM, evidence miner buffers, …) to same device as logits
    loss_fn.to(device)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    accum_steps = train_cfg["grad_accumulation_steps"]
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = steps_per_epoch * train_cfg["epochs"]

    optimizer = build_optimizer(model, train_cfg)
    loss_params = [p for p in loss_fn.parameters() if p.requires_grad]
    if loss_params:
        optimizer.add_param_group(
            {
                "params": loss_params,
                "lr": train_cfg["learning_rate_other"],
                "weight_decay": train_cfg["weight_decay"],
            }
        )
    scheduler = build_scheduler(optimizer, total_steps, train_cfg["warmup_ratio"])

    # Mixed precision: bf16/fp16 autocast. GradScaler is only for fp16 — bf16 has
    # sufficient range without scaling; using GradScaler with bf16 can raise
    # "Attempting to unscale FP16 gradients."
    amp_dtype = _resolve_amp_dtype(train_cfg, device)
    scaler: Optional[torch.amp.GradScaler] = None  # type: ignore[attr-defined]
    if amp_dtype is not None:
        log.info(
            "Mixed-precision training enabled (dtype=%s).",
            str(amp_dtype).replace("torch.", ""),
        )
        if amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")  # type: ignore[attr-defined]
            log.info("GradScaler enabled (fp16).")
        else:
            log.info("GradScaler disabled (bf16 uses autocast only).")

    # ------------------------------------------------------------------
    # Optional: resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    best_ign_f1 = 0.0
    no_improve_count = 0
    global_step = 0

    if resume_path is not None:
        log.info("Resuming from checkpoint: %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_ign_f1 = ckpt.get("best_ign_f1", 0.0)
        global_step = ckpt.get("global_step", 0)
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        log.info("Resumed at epoch %d (best Ign F1 so far: %.4f)", start_epoch, best_ign_f1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log.info("Starting training for %d epochs...", train_cfg["epochs"])
    model.train()

    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_loss_total = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_gcl = 0.0
        epoch_loss_evid = 0.0
        num_batches = 0
        profiled_batches = 0
        time_acc = {
            "data_wait": 0.0,
            "batch_prep": 0.0,
            "forward": 0.0,
            "loss": 0.0,
            "backward": 0.0,
            "opt_step": 0.0,
            "compute_total": 0.0,
        }

        # Determine whether BMM should be active this epoch
        bmm_active = epoch >= loss_cfg["bmm_warmup_epochs"]
        if not bmm_active:
            log.info("Epoch %d/%d: BMM warm-up (CE-only mode)", epoch + 1, train_cfg["epochs"])
        else:
            log.info("Epoch %d/%d: Joint CE + GCL mode", epoch + 1, train_cfg["epochs"])

        optimizer.zero_grad()

        batch_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        iter_end_time = time.perf_counter()
        for batch_idx, batch in batch_pbar:
            iter_start_time = time.perf_counter()
            do_profile = profile_timing and profiled_batches < profile_batches
            data_wait_s = iter_start_time - iter_end_time

            def _tstamp() -> float:
                if do_profile and device.type == "cuda":
                    torch.cuda.synchronize()
                return time.perf_counter()

            t0 = _tstamp()
            model_batch = build_model_batch(batch, device, include_flat_labels=True)
            t1 = _tstamp()

            # ----------------------------------------------------------------
            # Forward (optional second pass for contrastive dropout augmentation)
            # ----------------------------------------------------------------
            amp_ctx = (
                torch.amp.autocast(  # type: ignore[attr-defined]
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=True,
                )
                if device.type == "cuda" and amp_dtype is not None
                else contextlib.nullcontext()
            )
            with amp_ctx:
                outputs_1 = model(model_batch)

            if train_cfg.get("single_forward_contrastive", True):
                outputs_2 = outputs_1
            else:
                with amp_ctx:
                    outputs_2 = model(model_batch)
            t2 = _tstamp()

            # ----------------------------------------------------------------
            # Compute joint loss
            # ----------------------------------------------------------------
            with amp_ctx:
                loss_inputs = pack_joint_loss_inputs(model, outputs_1, outputs_2)
                loss_dict = loss_fn(loss_inputs, epoch, global_step)
            t3 = _tstamp()

            total_loss: torch.Tensor = loss_dict["total"] / accum_steps

            # ----------------------------------------------------------------
            # Backward
            # ----------------------------------------------------------------
            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            t4 = _tstamp()

            # ----------------------------------------------------------------
            # Accumulate stats
            # ----------------------------------------------------------------
            epoch_loss_total += loss_dict["total"].item()
            epoch_loss_ce += loss_dict.get("ce", torch.tensor(0.0)).item()
            epoch_loss_gcl += loss_dict.get("gcl", torch.tensor(0.0)).item()
            epoch_loss_evid += loss_dict.get("evidence_cl", torch.tensor(0.0)).item()
            num_batches += 1

            batch_pbar.set_postfix(
                loss=f"{loss_dict['total'].item():.4f}",
                avg=f"{epoch_loss_total / num_batches:.4f}",
                refresh=False,
            )

            # ----------------------------------------------------------------
            # Optimizer step (after accumulation)
            # ----------------------------------------------------------------
            if (batch_idx + 1) % accum_steps == 0:
                clip_params = list(model.parameters()) + list(loss_fn.parameters())
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        clip_params, train_cfg["max_grad_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        clip_params, train_cfg["max_grad_norm"]
                    )
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # --------------------------------------------------------
                # Periodic logging
                # --------------------------------------------------------
                if global_step % log_cfg["log_every"] == 0:
                    step_metrics = {
                        "loss_total": epoch_loss_total / num_batches,
                        "loss_ce": epoch_loss_ce / num_batches,
                        "loss_gcl": epoch_loss_gcl / num_batches,
                        "loss_evidence": epoch_loss_evid / num_batches,
                        "lr_plm": scheduler.get_last_lr()[0],
                        "lr_other": scheduler.get_last_lr()[-1],
                    }
                    log.info(
                        "Epoch %d | Step %d\n%s",
                        epoch + 1,
                        global_step,
                        format_metrics(step_metrics, prefix="  "),
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({"train/" + k: v for k, v in step_metrics.items()}, step=global_step)

                # BMM parameters are updated inside HardNegativeWeighter.compute_weights
                # when ``epoch >= bmm_warmup_epochs`` (see joint_loss / bmm.py).

            t5 = _tstamp()
            if do_profile:
                time_acc["data_wait"] += data_wait_s
                time_acc["batch_prep"] += (t1 - t0)
                time_acc["forward"] += (t2 - t1)
                time_acc["loss"] += (t3 - t2)
                time_acc["backward"] += (t4 - t3)
                if (batch_idx + 1) % accum_steps == 0:
                    time_acc["opt_step"] += (t5 - t4)
                time_acc["compute_total"] += (t5 - t0)
                profiled_batches += 1

            iter_end_time = t5

        if profile_timing and profiled_batches > 0:
            total_s = time_acc["data_wait"] + time_acc["compute_total"]
            denom = total_s if total_s > 0.0 else 1.0

            def _pct(x: float) -> float:
                return 100.0 * x / denom

            def _ms_per_batch(x: float) -> float:
                return 1000.0 * x / profiled_batches

            log.info(
                "Epoch %d timing profile (first %d batches):\n"
                "  data_wait : %8.2f ms/batch (%5.1f%%)\n"
                "  batch_prep: %8.2f ms/batch (%5.1f%%)\n"
                "  forward   : %8.2f ms/batch (%5.1f%%)\n"
                "  loss      : %8.2f ms/batch (%5.1f%%)\n"
                "  backward  : %8.2f ms/batch (%5.1f%%)\n"
                "  opt_step* : %8.2f ms/batch (%5.1f%%)\n"
                "  total     : %8.2f ms/batch (100.0%%)\n"
                "  *opt_step is averaged over all profiled batches (includes accumulation gaps).",
                epoch + 1,
                profiled_batches,
                _ms_per_batch(time_acc["data_wait"]),
                _pct(time_acc["data_wait"]),
                _ms_per_batch(time_acc["batch_prep"]),
                _pct(time_acc["batch_prep"]),
                _ms_per_batch(time_acc["forward"]),
                _pct(time_acc["forward"]),
                _ms_per_batch(time_acc["loss"]),
                _pct(time_acc["loss"]),
                _ms_per_batch(time_acc["backward"]),
                _pct(time_acc["backward"]),
                _ms_per_batch(time_acc["opt_step"]),
                _pct(time_acc["opt_step"]),
                _ms_per_batch(total_s),
            )

        # ------------------------------------------------------------------
        # End-of-epoch evaluation
        # ------------------------------------------------------------------
        if log_cfg.get("eval_every_epoch", True):
            log.info("Evaluating on dev set after epoch %d...", epoch + 1)
            dev_metrics = evaluate_dev(model, dev_loader, device)
            log.info(
                "Dev metrics (epoch %d):\n%s",
                epoch + 1,
                format_metrics(dev_metrics, prefix="  "),
            )

            if use_wandb:
                import wandb
                wandb.log(
                    {"dev/" + k: v for k, v in dev_metrics.items()},
                    step=global_step,
                )

            # Save best model
            ign_f1 = dev_metrics["ign_f1"]
            if ign_f1 > best_ign_f1:
                best_ign_f1 = ign_f1
                no_improve_count = 0
                if log_cfg.get("save_best", True):
                    ckpt_state = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_ign_f1": best_ign_f1,
                        "global_step": global_step,
                        "config": config,
                    }
                    if scaler is not None:
                        ckpt_state["scaler_state_dict"] = scaler.state_dict()
                    save_checkpoint(ckpt_state, output_dir, "best_model.pt")
                    log.info(
                        "New best Ign F1: %.4f — checkpoint saved.", best_ign_f1
                    )
            else:
                no_improve_count += 1
                log.info(
                    "No improvement for %d epoch(s) (best=%.4f).",
                    no_improve_count,
                    best_ign_f1,
                )

            # Early stopping
            patience = train_cfg.get("patience", 5)
            if no_improve_count >= patience:
                log.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    patience,
                )
                break

    log.info("Training complete. Best dev Ign F1: %.4f", best_ign_f1)
    if use_wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the training script."""
    args = parse_args()

    # Load base config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.overrides:
        config = apply_overrides(config, args.overrides)

    if args.eval_only:
        if args.resume is None:
            raise ValueError("--eval_only requires --resume <checkpoint_path>")
        # Delegate to evaluation script
        from scripts.evaluate import run_evaluation
        run_evaluation(config, args.resume, split="dev")
    else:
        train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
