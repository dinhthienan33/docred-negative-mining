#!/usr/bin/env python3
"""
Evaluation script for the DocRED SOTA pipeline.

Usage::

    # Evaluate on dev set with threshold search
    python scripts/evaluate.py \\
        --config configs/default.yaml \\
        --checkpoint outputs/best_model.pt \\
        --split dev \\
        --threshold_search

    # Evaluate on test set with a fixed threshold and produce submission file
    python scripts/evaluate.py \\
        --config configs/default.yaml \\
        --checkpoint outputs/best_model.pt \\
        --split test \\
        --threshold 0.35 \\
        --output outputs/test_predictions.json

Outputs:
  - Console: F1, Ign F1, per-relation F1 breakdown
  - File (--output): DocRED official submission format JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import contextlib
import torch
from torch.utils.data import DataLoader

from src.data.docred_dataset import DocREDDataset, DocREDRelationInfo, docred_collate_fn
from src.models.pipeline import DocREDPipeline
from src.utils.load_dataset import ensure_docred_data_paths
from src.utils.helpers import (
    get_device,
    load_config,
    load_checkpoint,
    create_logger,
    format_metrics,
)

logger = logging.getLogger("docred.evaluate")


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def compute_f1(
    predictions: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
) -> Tuple[float, float, float]:
    """Compute micro-averaged precision, recall, and F1.

    Args:
        predictions: List of prediction dicts, each with keys
            ``"title"``, ``"h_idx"``, ``"t_idx"``, ``"r_id"``.
        gold: List of gold-label dicts with the same keys.

    Returns:
        Tuple of ``(f1, precision, recall)`` as floats in ``[0, 1]``.
    """
    pred_set: Set[Tuple] = {
        (p["title"], p["h_idx"], p["t_idx"], p["r_id"]) for p in predictions
    }
    gold_set: Set[Tuple] = {
        (g["title"], g["h_idx"], g["t_idx"], g["r_id"]) for g in gold
    }

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return f1, precision, recall


def compute_ign_f1(
    predictions: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
    train_triples: Set[Tuple],
) -> Tuple[float, float, float]:
    """Compute Ign F1 — F1 after ignoring triples shared with the training set.

    This is the standard DocRED evaluation metric that prevents inflated scores
    from memorisation of training-set facts.

    Args:
        predictions: List of prediction dicts (same format as :func:`compute_f1`).
        gold: List of gold-label dicts (same format as :func:`compute_f1`).
        train_triples: Set of ``(title, h_idx, t_idx, r_id)`` tuples from the
            *training* set.  Triples present here are excluded from both
            predictions and gold before computing F1.

    Returns:
        Tuple of ``(ign_f1, precision, recall)``.
    """
    # Filter out triples that appear in training
    pred_ign = [
        p for p in predictions
        if (p["title"], p["h_idx"], p["t_idx"], p["r_id"]) not in train_triples
    ]
    gold_ign = [
        g for g in gold
        if (g["title"], g["h_idx"], g["t_idx"], g["r_id"]) not in train_triples
    ]
    return compute_f1(pred_ign, gold_ign)


def compute_per_relation_f1(
    predictions: List[Dict[str, Any]],
    gold: List[Dict[str, Any]],
    relation_info: DocREDRelationInfo,
) -> Dict[str, Dict[str, float]]:
    """Compute per-relation precision, recall, and F1.

    Args:
        predictions: List of prediction dicts.
        gold: List of gold-label dicts.
        relation_info: :class:`DocREDRelationInfo` instance for id→name mapping.

    Returns:
        Dict mapping relation name → dict with keys ``"f1"``, ``"precision"``,
        ``"recall"``, ``"support"`` (number of gold instances).
    """
    # Group by relation id
    pred_by_rel: Dict[int, Set[Tuple]] = defaultdict(set)
    gold_by_rel: Dict[int, Set[Tuple]] = defaultdict(set)

    for p in predictions:
        pred_by_rel[p["r_id"]].add((p["title"], p["h_idx"], p["t_idx"]))
    for g in gold:
        gold_by_rel[g["r_id"]].add((g["title"], g["h_idx"], g["t_idx"]))

    results: Dict[str, Dict[str, float]] = {}
    all_rel_ids = set(pred_by_rel.keys()) | set(gold_by_rel.keys())

    for r_id in sorted(all_rel_ids):
        preds_r = pred_by_rel.get(r_id, set())
        golds_r = gold_by_rel.get(r_id, set())
        tp = len(preds_r & golds_r)
        fp = len(preds_r - golds_r)
        fn = len(golds_r - preds_r)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rel_name = relation_info.get_name(r_id)
        results[rel_name] = {
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "support": len(golds_r),
        }

    return results


# ---------------------------------------------------------------------------
# Threshold search
# ---------------------------------------------------------------------------

def search_threshold(
    model: DocREDPipeline,
    dev_loader: DataLoader,
    device: torch.device,
    gold_records: List[Dict[str, Any]],
    thresholds: List[float],
    use_fp16: bool = False,
) -> Tuple[float, float]:
    """Grid-search for the optimal sigmoid threshold on the dev set.

    Runs inference once and caches per-pair probabilities, then evaluates
    every candidate threshold without re-running the model.

    Args:
        model: Trained pipeline model in eval mode.
        dev_loader: DataLoader for the dev split.
        device: Compute device.
        gold_records: Pre-collected gold label records (list of dicts).
        thresholds: Sorted list of threshold candidates in ``[0, 1]``.
        use_fp16: Whether to run inference with autocast (fp16).

    Returns:
        Tuple ``(best_threshold, best_ign_f1)``  where ``best_ign_f1`` is the
        F1 at the returned threshold (not Ign F1 here — full F1 used for speed;
        caller can re-evaluate with train triples if needed).
    """
    model.eval()
    # Collect (title, h, t, r_id, prob) for all pairs
    all_pair_probs: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)  # type: ignore[attr-defined]
                if use_fp16 and device.type == "cuda"
                else contextlib.nullcontext()
            )
            with amp_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_spans=batch["entity_spans"],
                    mention_to_entity=batch["mention_to_entity"],
                    sentence_boundaries=batch["sentence_boundaries"],
                    hts=batch["hts"],
                )

            for title, hts, doc_logits in zip(
                batch["titles"], batch["hts"], outputs["logits"]
            ):
                probs = torch.sigmoid(doc_logits).cpu()  # [num_pairs, num_relations]
                for pair_idx, (h, t) in enumerate(hts):
                    for r_id in range(1, probs.shape[1]):
                        all_pair_probs.append(
                            {
                                "title": title,
                                "h_idx": h,
                                "t_idx": t,
                                "r_id": r_id,
                                "prob": probs[pair_idx, r_id].item(),
                            }
                        )

    # Grid search
    gold_set: Set[Tuple] = {
        (g["title"], g["h_idx"], g["t_idx"], g["r_id"]) for g in gold_records
    }
    best_thresh = thresholds[0]
    best_f1 = 0.0

    for thresh in thresholds:
        preds = [p for p in all_pair_probs if p["prob"] > thresh]
        pred_set = {(p["title"], p["h_idx"], p["t_idx"], p["r_id"]) for p in preds}
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    model.train()
    logger.info("Threshold search: best=%.4f (F1=%.4f)", best_thresh, best_f1)
    return best_thresh, best_f1


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: DocREDPipeline,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float,
    relation_info: DocREDRelationInfo,
    use_fp16: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run inference and collect predictions and gold records.

    Args:
        model: Trained model.
        data_loader: DataLoader for the target split.
        device: Compute device.
        threshold: Sigmoid threshold for positive prediction.
        relation_info: Relation id → name mapper.
        use_fp16: Whether to use mixed-precision inference.

    Returns:
        Tuple of ``(predictions, gold_records)`` where each element is a list
        of dicts with keys ``"title"``, ``"h_idx"``, ``"t_idx"``, ``"r_id"``.
        ``gold_records`` is empty for the test split (no labels).
    """
    model.eval()
    predictions: List[Dict[str, Any]] = []
    gold_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            amp_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)  # type: ignore[attr-defined]
                if use_fp16 and device.type == "cuda"
                else contextlib.nullcontext()
            )
            with amp_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    entity_spans=batch["entity_spans"],
                    mention_to_entity=batch["mention_to_entity"],
                    sentence_boundaries=batch["sentence_boundaries"],
                    hts=batch["hts"],
                )

            for doc_idx, (title, hts, doc_logits, gold_labels) in enumerate(
                zip(
                    batch["titles"],
                    batch["hts"],
                    outputs["logits"],
                    batch["labels"],
                )
            ):
                probs = torch.sigmoid(doc_logits).cpu()  # [num_pairs, num_rel]

                # Evidence from model (if available)
                doc_evidence = outputs.get("evidence_scores")
                if doc_evidence is not None:
                    ev_scores = doc_evidence[doc_idx].cpu()
                else:
                    ev_scores = None

                for pair_idx, (h, t) in enumerate(hts):
                    for r_id in range(1, probs.shape[1]):
                        if probs[pair_idx, r_id] > threshold:
                            # Retrieve evidence sentence indices if available
                            evid_sents: List[int] = []
                            if (h, t, r_id) in batch["evidence"][doc_idx]:
                                evid_sents = batch["evidence"][doc_idx][(h, t, r_id)]
                            predictions.append(
                                {
                                    "title": title,
                                    "h_idx": h,
                                    "t_idx": t,
                                    "r": relation_info.get_name(r_id),
                                    "r_id": r_id,
                                    "evidence": evid_sents,
                                }
                            )

                # Collect gold labels
                if gold_labels is not None:
                    num_ents = gold_labels.shape[0]
                    for h in range(num_ents):
                        for t in range(num_ents):
                            if h == t:
                                continue
                            for r_id in range(1, gold_labels.shape[2]):
                                if gold_labels[h, t, r_id] > 0:
                                    gold_records.append(
                                        {
                                            "title": title,
                                            "h_idx": h,
                                            "t_idx": t,
                                            "r_id": r_id,
                                        }
                                    )

    return predictions, gold_records


# ---------------------------------------------------------------------------
# Submission format
# ---------------------------------------------------------------------------

def format_submission(
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert internal prediction dicts to DocRED official submission format.

    Args:
        predictions: List of prediction dicts with keys
            ``"title"``, ``"h_idx"``, ``"t_idx"``, ``"r"``, ``"evidence"``.

    Returns:
        List of dicts in the official format::

            [{"title": ..., "h_idx": ..., "t_idx": ..., "r": "P17", "evidence": [0, 3]}]
    """
    return [
        {
            "title": p["title"],
            "h_idx": p["h_idx"],
            "t_idx": p["t_idx"],
            "r": p["r"],
            "evidence": p.get("evidence", []),
        }
        for p in predictions
    ]


# ---------------------------------------------------------------------------
# Build train-triple set for Ign F1
# ---------------------------------------------------------------------------

def load_train_triples(
    train_path: str,
    relation_info: DocREDRelationInfo,
) -> Set[Tuple]:
    """Load all relation triples from the training set as a set.

    Used to filter shared triples when computing Ign F1.

    Args:
        train_path: Path to ``train_annotated.json``.
        relation_info: Relation name → id mapper.

    Returns:
        Set of ``(title, h_idx, t_idx, r_id)`` tuples.
    """
    try:
        with open(train_path, "r", encoding="utf-8") as fh:
            train_docs = json.load(fh)
    except FileNotFoundError:
        logger.warning("Training file not found at %s; Ign F1 will equal F1.", train_path)
        return set()

    triples: Set[Tuple] = set()
    for doc in train_docs:
        title = doc.get("title", "")
        for label in doc.get("labels", []):
            r_id = relation_info.get_id(label["r"])
            triples.add((title, label["h"], label["t"], r_id))
    logger.info("Loaded %d training triples for Ign F1.", len(triples))
    return triples


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(
    config: Dict[str, Any],
    checkpoint_path: str,
    split: str = "dev",
    output_path: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Load a trained model and evaluate on the specified split.

    Args:
        config: Full configuration dict.
        checkpoint_path: Path to the model checkpoint.
        split: ``"dev"`` or ``"test"``.
        output_path: If provided, write predictions in DocRED submission format.
        threshold: Fixed classification threshold.  When ``None`` and
            ``config["evaluation"]["threshold_search"]`` is ``True``, the
            optimal threshold is found via grid search on the dev set.

    Returns:
        Dict with evaluation metrics.
    """
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    log_cfg = config["logging"]

    output_dir = Path(log_cfg["output_dir"])
    log = create_logger(output_dir)
    data_cfg = ensure_docred_data_paths(config["data"], log)
    config["data"] = data_cfg
    device = get_device()

    # ------------------------------------------------------------------
    # Relation info
    # ------------------------------------------------------------------
    relation_info = DocREDRelationInfo(data_cfg.get("rel_info_path"))

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    if split == "dev":
        data_path = data_cfg["dev_path"]
    elif split == "test":
        data_path = data_cfg["test_path"]
    else:
        raise ValueError(f"Unknown split: {split!r}. Choose 'dev' or 'test'.")

    dataset = DocREDDataset(
        data_path=data_path,
        tokenizer_name=model_cfg["plm_name"],
        max_length=data_cfg["max_length"],
        relation_map_path=data_cfg.get("rel_info_path"),
        use_entity_markers=data_cfg.get("use_entity_markers", True),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=docred_collate_fn,
        num_workers=2,
    )
    log.info("Evaluating on %s split (%d documents)", split, len(dataset))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    log.info("Loading model from %s", checkpoint_path)
    model = DocREDPipeline(model_cfg)
    ckpt = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    use_fp16 = config["training"].get("fp16", False) and device.type == "cuda"

    # ------------------------------------------------------------------
    # Threshold search / selection
    # ------------------------------------------------------------------
    if threshold is None:
        if eval_cfg.get("threshold_search", True) and split == "dev":
            trange = eval_cfg.get("threshold_range", [0.0, 1.0])
            tsteps = eval_cfg.get("threshold_steps", 100)
            import numpy as np
            thresh_candidates = list(np.linspace(trange[0], trange[1], tsteps + 1))

            # Collect gold on dev first
            _, gold_for_search = run_inference(
                model, data_loader, device, threshold=0.0, relation_info=relation_info, use_fp16=use_fp16
            )
            threshold, _ = search_threshold(
                model, data_loader, device, gold_for_search, thresh_candidates, use_fp16=use_fp16
            )
        else:
            threshold = 0.5

    log.info("Using classification threshold: %.4f", threshold)

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    predictions, gold_records = run_inference(
        model, data_loader, device, threshold=threshold,
        relation_info=relation_info, use_fp16=use_fp16,
    )
    log.info("Total predictions: %d | Total gold: %d", len(predictions), len(gold_records))

    # ------------------------------------------------------------------
    # Compute metrics (only for dev; test has no gold labels)
    # ------------------------------------------------------------------
    metrics: Dict[str, float] = {}
    if split == "dev" and gold_records:
        f1, prec, rec = compute_f1(predictions, gold_records)
        train_triples = load_train_triples(data_cfg["train_path"], relation_info)
        ign_f1, ign_prec, ign_rec = compute_ign_f1(predictions, gold_records, train_triples)
        per_rel_f1 = compute_per_relation_f1(predictions, gold_records, relation_info)

        metrics = {
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "ign_f1": ign_f1,
            "ign_precision": ign_prec,
            "ign_recall": ign_rec,
            "threshold": threshold,
        }

        log.info("Evaluation results (split=%s):\n%s", split, format_metrics(metrics, prefix="  "))

        # Per-relation breakdown
        log.info("Per-relation F1:")
        for rel_name, rel_metrics in sorted(
            per_rel_f1.items(), key=lambda x: -x[1]["support"]
        ):
            log.info(
                "  %-12s  F1=%.4f  P=%.4f  R=%.4f  support=%d",
                rel_name,
                rel_metrics["f1"],
                rel_metrics["precision"],
                rel_metrics["recall"],
                rel_metrics["support"],
            )

    # ------------------------------------------------------------------
    # Write submission file
    # ------------------------------------------------------------------
    if output_path is not None:
        submission = format_submission(predictions)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(submission, fh, ensure_ascii=False, indent=2)
        log.info("Predictions written to %s (%d entries)", out, len(submission))

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DocRED SOTA model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Classification threshold.  When not provided and "
            "--threshold_search is used, the threshold is searched on the dev set."
        ),
    )
    parser.add_argument(
        "--threshold_search",
        action="store_true",
        help="Override config to enable threshold search.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for predictions in DocRED submission format.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the evaluation script."""
    args = parse_args()
    config = load_config(args.config)

    if args.threshold_search:
        config["evaluation"]["threshold_search"] = True

    run_evaluation(
        config=config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_path=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
