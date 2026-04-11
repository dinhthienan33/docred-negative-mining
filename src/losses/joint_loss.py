"""
Joint Training Loss for DocRED SOTA Pipeline.

Combines:
  L_total = L_CE + λ_gcl * L_gcl + λ_evidence * L_evidence_cl

where:
  - L_CE         : Adaptive Thresholding Loss (ATLOP) for multi-label relation classification
  - L_gcl        : BMM-reweighted InfoNCE contrastive loss over triple graph
  - L_evidence_cl: Same InfoNCE but with evidence-aware hard negatives

References:
  - ATLOP: Document-Level Relation Extraction with Adaptive Thresholding
           and Localized Context Pooling (Zhou et al., 2021)
  - ProGCL: Prototypical Graph Contrastive Learning (Wang et al., 2022)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .bmm import HardNegativeWeighter
from .evidence_negatives import EvidenceNegativeMiner

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _pair_evidence_to_miner_tuples(
    h: int,
    t: int,
    rels: List[int],
    evid_i: Union[set, Dict[int, Union[set, List[int]]]],
) -> List[Tuple[int, int, int, List[int]]]:
    """Map pipeline evidence to ``DocEvid`` rows for :meth:`EvidenceNegativeMiner.update_statistics`.

    The pipeline stores one ``set`` of sentence ids per (h, t) (merged across relations).
    The miner can also accept a per-relation dict ``r -> sentences``.
    """
    if isinstance(evid_i, dict):
        rows: List[Tuple[int, int, int, List[int]]] = []
        for r, sents in evid_i.items():
            s = set(sents) if not isinstance(sents, set) else sents
            rows.append((h, t, int(r), sorted(s)))
        return rows
    sents_list = sorted(evid_i) if evid_i else []
    return [(h, t, int(r), sents_list) for r in rels]


def _pair_evidence_to_relation_dict(
    rels: List[int],
    evid_i: Union[set, Dict[int, Union[set, List[int]]]],
) -> Dict[int, set]:
    """Map to ``r_id -> set(sentence_ids)`` for :meth:`EvidenceNegativeMiner.get_hard_negatives`."""
    if isinstance(evid_i, dict):
        return {
            int(r): (s if isinstance(s, set) else set(s))
            for r, s in evid_i.items()
        }
    s = evid_i if isinstance(evid_i, set) else set(evid_i)
    if not rels:
        return {}
    return {int(r): set(s) for r in rels}


# ===========================================================================
# 1.  ATLOP Adaptive Thresholding Loss
# ===========================================================================


class ATLOPLoss(nn.Module):
    """
    Adaptive Thresholding Loss for multi-label document-level relation extraction.

    Introduces a learnable *threshold class* (TH) alongside the ``num_relations``
    relation classes.  For a given entity pair:

    - Positive relations must score above the threshold logit.
    - Negative relations must score below the threshold logit.

    The loss pushes each positive relation's logit above TH, and each negative
    relation's logit below TH, using a cross-entropy formulation:

        For positives:  -log( exp(l_r) / (exp(l_r) + exp(l_TH)) )
        For negatives:  -log( exp(l_TH) / (exp(l_TH) + Σ_{r∈neg} exp(l_r)) )

    where l_r is the logit for relation r and l_TH is the threshold logit.

    Parameters
    ----------
    num_relations : int
        Number of relation types (excluding the NA/TH class).
    """

    def __init__(self, num_relations: int = 97) -> None:
        super().__init__()
        self.num_relations = num_relations
        # Learnable threshold logit (one scalar per relation is also possible;
        # we use a single shared threshold as per the original ATLOP paper)
        self.threshold = nn.Parameter(torch.zeros(1))  # scalar

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute the adaptive thresholding loss.

        Parameters
        ----------
        logits : Tensor
            Raw (un-normalised) scores, shape [batch_pairs, num_relations].
        labels : Tensor
            Multi-hot binary labels, shape [batch_pairs, num_relations].
            1 indicates a positive relation; 0 indicates a negative/NA relation.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        # logits  : [B, R]
        # labels  : [B, R]  — binary multi-hot
        B, R = logits.shape
        device = logits.device

        # Broadcast threshold to match logit dimensionality: [B, 1]
        th = self.threshold.expand(B, 1)  # [B, 1]

        # Append threshold as an extra "relation" column: [B, R+1]
        # Index R corresponds to the threshold class
        logits_with_th = torch.cat([logits, th], dim=-1)  # [B, R+1]

        # ---- Positive loss ----
        # For each positive relation r: push l_r above l_TH
        # = -log σ(l_r - l_TH)  summed over positives
        pos_mask = labels.bool()  # [B, R]

        # For positive relations, consider the 2-way contest {r, TH}
        # log_softmax over [l_r, l_TH] → take the index-0 term
        # We compute this for ALL relations then mask to positives
        pos_logits = torch.stack(
            [logits, th.expand(B, R)], dim=-1
        )  # [B, R, 2]
        pos_log_prob = F.log_softmax(pos_logits, dim=-1)[:, :, 0]  # [B, R]
        # Average over positive relations per example
        pos_count = pos_mask.float().sum(dim=-1).clamp(min=1.0)    # [B]
        L_pos = -(pos_log_prob * pos_mask.float()).sum(dim=-1) / pos_count  # [B]

        # ---- Negative loss ----
        # For negative relations: push l_TH above all negative l_r's
        # = -log( exp(l_TH) / (exp(l_TH) + Σ_{neg} exp(l_r)) )
        neg_mask = (~pos_mask).float()  # [B, R]
        # Set positive logits to a large negative number so they don't contribute
        neg_logits_masked = logits.masked_fill(pos_mask, -1e9)       # [B, R]
        # Concatenate TH: [B, R+1]; TH is at position 0 for log_softmax
        neg_input = torch.cat([th, neg_logits_masked], dim=-1)       # [B, R+1]
        neg_log_prob = F.log_softmax(neg_input, dim=-1)[:, 0]        # [B]
        L_neg = -neg_log_prob                                         # [B]

        # ---- Combine ----
        # Only compute negative loss when there are negatives in the example
        has_neg = neg_mask.sum(dim=-1) > 0                            # [B]
        L_total = L_pos + L_neg * has_neg.float()                     # [B]
        return L_total.mean()


# ===========================================================================
# 2.  BMM-reweighted InfoNCE
# ===========================================================================


class BMM_InfoNCE(nn.Module):
    """
    BMM-reweighted InfoNCE contrastive loss.

    Standard InfoNCE weights each negative equally.  Here, per-negative weights
    ``w_k`` (from the BMM) modulate the negative term so that likely *false*
    negatives (high similarity, same class) are down-weighted:

        L = -mean_n [ log(
                exp(sim(a_n, p_n) / τ)
                ──────────────────────────────────────────────────────────
                exp(sim(a_n, p_n) / τ) + Σ_k  w_nk * exp(sim(a_n, neg_nk) / τ)
            ) ]

    Parameters
    ----------
    temperature : float
        Softmax temperature τ.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor_embs: Tensor,
        positive_embs: Tensor,
        negative_embs: Tensor,
        negative_weights: Tensor,
        neg_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute BMM-reweighted InfoNCE loss.

        Parameters
        ----------
        anchor_embs : Tensor
            Anchor triple embeddings, shape [N, dim].
        positive_embs : Tensor
            Augmented positive view embeddings, shape [N, dim].
        negative_embs : Tensor
            Negative embeddings, shape [N, K, dim]  (K negatives per anchor).
        negative_weights : Tensor
            BMM-derived weights, shape [N, K].
            Higher weight ⟹ more likely a true negative ⟹ kept at full strength.
            Lower weight ⟹ suspected false negative ⟹ down-weighted.
        neg_valid_mask : Tensor, optional
            Boolean mask [N, K].  If provided, invalid positions are excluded from
            the denominator (for padded evidence-CL negatives).

        Returns
        -------
        Tensor
            Scalar loss.
        """
        # anchor_embs   : [N, dim]
        # positive_embs : [N, dim]
        # negative_embs : [N, K, dim]
        # negative_weights: [N, K]

        N, K, dim = negative_embs.shape
        device = anchor_embs.device
        tau = self.temperature

        # L2-normalise all embeddings
        a = F.normalize(anchor_embs.float(),   p=2, dim=-1)   # [N, dim]
        p = F.normalize(positive_embs.float(), p=2, dim=-1)   # [N, dim]
        n = F.normalize(negative_embs.float(), p=2, dim=-1)   # [N, K, dim]

        # ---- Anchor–positive similarity ----
        sim_ap = (a * p).sum(dim=-1) / tau  # [N]   (element-wise dot product)

        # ---- Anchor–negative similarity ----
        # a[:, None, :] → [N, 1, dim]  ·  n → [N, K, dim] → sum → [N, K]
        sim_an = (a.unsqueeze(1) * n).sum(dim=-1) / tau  # [N, K]

        # ---- Apply BMM weights ----
        w = negative_weights.to(device=device, dtype=a.dtype)  # [N, K]
        if neg_valid_mask is not None:
            w = w.masked_fill(~neg_valid_mask, 1.0)
        w = w.clamp(min=_EPS)

        # ---- Numerically stable log-sum-exp ----
        # numerator   = exp(sim_ap)
        # denominator = exp(sim_ap) + Σ_k w_k * exp(sim_an_k)

        # Use log-sum-exp trick:
        # log-denom = log( exp(sim_ap) + Σ_k w_k * exp(sim_an_k) )
        #           = logsumexp([ sim_ap,  sim_an_k + log(w_k) ])

        log_w = torch.log(w)  # [N, K]
        # Weighted negative terms in log-space: sim_an + log(w) → [N, K]
        neg_log_terms = sim_an + log_w  # [N, K]
        if neg_valid_mask is not None:
            neg_log_terms = neg_log_terms.masked_fill(
                ~neg_valid_mask, float("-inf")
            )

        # Concatenate positive and weighted negative log-terms: [N, K+1]
        all_log_terms = torch.cat(
            [sim_ap.unsqueeze(1), neg_log_terms], dim=1
        )  # [N, K+1]

        log_denom = torch.logsumexp(all_log_terms, dim=1)  # [N]
        log_numer = sim_ap                                  # [N]

        loss = -(log_numer - log_denom).mean()
        return loss


# ===========================================================================
# 3.  Joint Loss (coordinator)
# ===========================================================================


class JointLoss(nn.Module):
    """
    Joint training loss combining CE (ATLOP), GCL (BMM-InfoNCE), and evidence CL.

    L_total = L_CE + λ_gcl * L_gcl + λ_evidence * L_evidence_cl

    Manages the BMM weighter and EvidenceNegativeMiner internally.

    Parameters
    ----------
    num_relations : int
        Number of relation types.
    lambda_gcl : float
        Weight for the standard GCL contrastive term.
    lambda_evidence : float
        Weight for the evidence-aware contrastive term.
    temperature : float
        InfoNCE temperature.
    bmm_warmup_epochs : int
        Epochs before BMM is activated.
    bmm_update_every : int
        Steps between BMM re-fits.
    evidence_overlap_threshold : float
        Hard-negative evidence overlap threshold.
    topk_hard : int
        Number of top hard negatives from EvidenceMiner.
    contrastive_top_k : int, optional
        If set, each anchor uses only this many in-batch negatives: those with
        highest cosine similarity to the anchor (hardest negatives).  The
        InfoNCE/BMM forward is O(B·K) in negatives instead of O(B²).  ``None``
        uses all B-1 other batch members (same negative set as before).
    """

    def __init__(
        self,
        num_relations: int = 97,
        lambda_gcl: float = 0.5,
        lambda_evidence: float = 0.3,
        temperature: float = 0.07,
        bmm_warmup_epochs: int = 3,
        bmm_update_every: int = 100,
        evidence_overlap_threshold: float = 0.3,
        topk_hard: int = 10,
        contrastive_top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_relations   = num_relations
        self.lambda_gcl      = lambda_gcl
        self.lambda_evidence = lambda_evidence
        self.temperature     = temperature
        self.contrastive_top_k = contrastive_top_k

        # Sub-modules
        self.atlop_loss = ATLOPLoss(num_relations=num_relations)
        self.bmm_infonce = BMM_InfoNCE(temperature=temperature)

        # Stateful helpers (not nn.Module — managed separately)
        self.hard_neg_weighter = HardNegativeWeighter(
            bmm_warmup_epochs=bmm_warmup_epochs,
            update_every_n_steps=bmm_update_every,
            temperature=temperature,
        )
        self.evidence_miner = EvidenceNegativeMiner(
            num_relations=num_relations,
            evidence_overlap_threshold=evidence_overlap_threshold,
            topk_hard=topk_hard,
        )

    def forward(
        self,
        model_outputs: Dict[str, object],
        epoch: int,
        step: int,
    ) -> Dict[str, Tensor]:
        """
        Compute the full joint loss.

        Parameters
        ----------
        model_outputs : dict
            Expected keys:

            - "logits"                     : Tensor [B, num_relations]
                Relation logits from the classification head.
            - "labels"                     : Tensor [B, num_relations]
                Multi-hot ground-truth labels.
            - "pair_embs"                  : Tensor [B, dim]
                Entity-pair embeddings (used for evidence miner).
            - "contrastive_embs"           : Tensor [B, dim]
                Projected anchor embeddings for contrastive loss.
            - "positive_contrastive_embs"  : Tensor [B, dim]
                Projected positive-view embeddings (augmented).
            - "evidence_sets"              : List[Dict[int, Set[int]]]  (optional)
                Per-sample mapping r_id → set of evidence sentence IDs.
            - "entity_pair_ids"            : List[Tuple[int,int]]  (optional)
                (h_idx, t_idx) per sample.
            - "relation_labels"            : same as "labels" (alias, optional)

        epoch : int
            Current training epoch (0-indexed).
        step : int
            Current global training step.

        Returns
        -------
        dict with keys:
            - "total"       : scalar loss for back-propagation
            - "ce"          : ATLOP cross-entropy loss
            - "gcl"         : standard BMM-InfoNCE loss
            - "evidence_cl" : evidence-aware BMM-InfoNCE loss
        """
        # ---- Unpack model outputs ----
        logits: Tensor          = model_outputs["logits"]                          # [B, R]
        labels: Tensor          = model_outputs["labels"]                          # [B, R]
        pair_embs: Tensor       = model_outputs["pair_embs"]                       # [B, dim]
        anchor_embs: Tensor     = model_outputs["contrastive_embs"]                # [B, dim]
        positive_embs: Tensor   = model_outputs["positive_contrastive_embs"]       # [B, dim]
        evidence_sets: List     = model_outputs.get("evidence_sets",   [])
        entity_pair_ids: List   = model_outputs.get("entity_pair_ids", [])
        # Support "relation_labels" as an alias for "labels"
        relation_labels: Tensor = model_outputs.get("relation_labels", labels)     # [B, R]

        B = logits.shape[0]
        device = logits.device

        # ==================================================================
        # 1. Classification Loss (ATLOP)
        # ==================================================================
        L_ce: Tensor = self.atlop_loss(logits, labels.float())

        # ==================================================================
        # 2. Prepare for contrastive losses
        # ==================================================================
        # If batch is too small for contrastive learning, skip
        if B < 2:
            zero = torch.tensor(0.0, device=device, requires_grad=False)
            return {
                "total":       L_ce,
                "ce":          L_ce.detach(),
                "gcl":         zero,
                "evidence_cl": zero,
            }

        # ---- 2a. BMM weights ----
        # anchor_embs and all other pair embeddings are the negatives (in-batch)
        # Negatives are all other anchors in the batch (rows != anchor row)
        # For simplicity we treat the full batch as the negative pool
        bmm_weights: Tensor = self.hard_neg_weighter.compute_weights(
            anchor_embs=anchor_embs,   # [B, dim]
            negative_embs=anchor_embs, # [B, dim]  (in-batch)
            epoch=epoch,
            step=step,
        )  # [B, B]

        # GCL: [B, K] hardest in-batch negatives (or all B-1 if contrastive_top_k is null)
        L_gcl: Tensor = self._compute_gcl_loss(
            anchor_embs=anchor_embs,
            positive_embs=positive_embs,
            all_embs=anchor_embs,
            bmm_weights=bmm_weights,
        )

        # ==================================================================
        # 3. Evidence-Aware Contrastive Loss
        # ==================================================================
        L_evidence_cl: Tensor = self._compute_evidence_cl_loss(
            anchor_embs=anchor_embs,
            positive_embs=positive_embs,
            pair_embs=pair_embs,
            bmm_weights=bmm_weights,
            evidence_sets=evidence_sets,
            entity_pair_ids=entity_pair_ids,
            relation_labels=relation_labels,
            device=device,
        )

        # ==================================================================
        # 4. Total Loss
        # ==================================================================
        L_total = L_ce + self.lambda_gcl * L_gcl + self.lambda_evidence * L_evidence_cl

        return {
            "total":       L_total,
            "ce":          L_ce.detach(),
            "gcl":         L_gcl.detach(),
            "evidence_cl": L_evidence_cl.detach(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_gcl_loss(
        self,
        anchor_embs: Tensor,      # [B, dim]
        positive_embs: Tensor,    # [B, dim]
        all_embs: Tensor,         # [B, dim]  — pool of in-batch negatives
        bmm_weights: Tensor,      # [B, B]
    ) -> Tensor:
        """
        Standard in-batch negative InfoNCE with BMM reweighting.

        For each anchor i, negatives are other batch members j ≠ i.  When
        ``contrastive_top_k`` is set, only the K negatives with highest cosine
        similarity to i are kept (hard negatives); otherwise all B-1 are used.
        The multiset is the same when K = B-1, so the loss matches the former
        all-negative formulation up to tie-breaking in ``topk``.

        Parameters
        ----------
        anchor_embs : Tensor [B, dim]
        positive_embs : Tensor [B, dim]
        all_embs : Tensor [B, dim]   — the full batch used as the negative pool
        bmm_weights : Tensor [B, B]  — bmm_weights[i, j] is the weight for
                                       using j as a negative for anchor i.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        B, _dim = anchor_embs.shape
        device = anchor_embs.device

        # Pairwise cosine similarities (same space as BMM); mask self so top-k
        # never selects the anchor row.
        a = F.normalize(anchor_embs.detach().float(), p=2, dim=-1)
        sim = torch.mm(a, a.t())
        sim = sim.masked_fill(
            torch.eye(B, dtype=torch.bool, device=device), float("-inf")
        )

        k_all = B - 1
        k = (
            k_all
            if self.contrastive_top_k is None
            else min(int(self.contrastive_top_k), k_all)
        )
        _, idx = torch.topk(sim, k=k, dim=-1, largest=True, sorted=False)
        neg_embs = all_embs[idx.long()]  # [B, k, dim]
        neg_w = bmm_weights.gather(1, idx.long())  # [B, k]

        return self.bmm_infonce(anchor_embs, positive_embs, neg_embs, neg_w)

    def _compute_evidence_cl_loss(
        self,
        anchor_embs: Tensor,
        positive_embs: Tensor,
        pair_embs: Tensor,
        bmm_weights: Tensor,
        evidence_sets: List,
        entity_pair_ids: List,
        relation_labels: Tensor,
        device: torch.device,
    ) -> Tensor:
        """
        Evidence-aware contrastive loss.

        Calls EvidenceNegativeMiner to sample tiered negatives for each anchor,
        then computes BMM-InfoNCE with those negatives.

        Returns a scalar loss, or 0.0 if evidence information is unavailable.
        """
        B = anchor_embs.shape[0]

        if not evidence_sets or not entity_pair_ids:
            # Fall back to standard in-batch negative InfoNCE
            return self._compute_gcl_loss(
                anchor_embs, positive_embs, anchor_embs, bmm_weights
            )

        # Update miner statistics if we have label data
        # (Lightweight: just use this batch's data)
        if len(evidence_sets) == B and isinstance(relation_labels, Tensor):
            # Convert tensors to the miner's expected format
            batch_labels_fmt: List[List] = []
            batch_evid_fmt:   List[List] = []
            for i, (h, t) in enumerate(entity_pair_ids):
                rels_i = relation_labels[i].nonzero(as_tuple=False).squeeze(-1).tolist()
                batch_labels_fmt.append([(h, t, rels_i)])
                evid_i = evidence_sets[i] if i < len(evidence_sets) else set()
                batch_evid_fmt.append(
                    _pair_evidence_to_miner_tuples(h, t, rels_i, evid_i)
                )
            self.evidence_miner.update_statistics(batch_labels_fmt, batch_evid_fmt)

        # Pipeline passes ``List[set]`` (sentence ids per pair); miner expects r -> set(sents).
        norm_evidence_sets: List[Dict[int, set]] = []
        for i in range(B):
            rels_i = relation_labels[i].nonzero(as_tuple=False).squeeze(-1).tolist()
            evid_i = evidence_sets[i] if i < len(evidence_sets) else set()
            norm_evidence_sets.append(_pair_evidence_to_relation_dict(rels_i, evid_i))

        # Prepare batch_data for sample_negatives
        batch_data = {
            "pair_embs":       pair_embs,
            "entity_pair_ids": entity_pair_ids,
            "evidence_sets":   norm_evidence_sets,
            "relation_labels": relation_labels,
        }

        # Collect evidence-aware negatives for each anchor (batched padding + mask)
        num_hard_per_anchor = min(5, B - 1)
        total_neg_per_anchor = min(B - 1, 15)
        num_med = min(5, total_neg_per_anchor - num_hard_per_anchor)
        num_easy = max(0, total_neg_per_anchor - num_hard_per_anchor - num_med)

        mined = self.evidence_miner.sample_negatives_batch(
            batch_data,
            num_hard=num_hard_per_anchor,
            num_medium=num_med,
            num_easy=num_easy,
        )
        idx = mined["indices"].to(device)
        padded_neg_w = mined["weights"].to(device)
        vm = mined["valid_mask"].to(device)
        if idx.numel() == 0 or idx.size(1) == 0:
            return torch.tensor(0.0, device=device, requires_grad=False)

        # Rows with no valid negatives: fall back to other in-batch indices (truncated to width)
        max_width = idx.size(1)
        for i in range(B):
            if not vm[i].any():
                fallback_idx = torch.cat(
                    [
                        torch.arange(0, i, device=device),
                        torch.arange(i + 1, B, device=device),
                    ]
                )
                kf = fallback_idx.numel()
                if kf == 0:
                    continue
                if kf > max_width:
                    fallback_idx = fallback_idx[:max_width]
                    kf = max_width
                idx[i, :kf] = fallback_idx
                padded_neg_w[i, :kf] = 1.0
                vm[i, :kf] = True

        safe_idx = torch.where(vm, idx, torch.zeros_like(idx))
        padded_neg_embs = anchor_embs[safe_idx]
        padded_neg_embs = padded_neg_embs * vm.unsqueeze(-1).to(
            dtype=padded_neg_embs.dtype
        )

        return self.bmm_infonce(
            anchor_embs,
            positive_embs,
            padded_neg_embs,
            padded_neg_w,
            neg_valid_mask=vm,
        )
