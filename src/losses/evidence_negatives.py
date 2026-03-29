"""
Evidence-Aware Negative Mining for DocRED.

DocRED provides sentence-level evidence annotations for each relation triple.
This module exploits those annotations to construct *hard* negatives — i.e.,
relation candidates that share evidence with a positive triple and frequently
co-occur with its relation type across the training corpus.

The scoring formula for hardness of a candidate negative relation r' for an
anchor (h, t, r) is:

    hardness(r') = alpha * evidence_overlap(E_r, E_r') + (1-alpha) * cooccur_freq(r, r')

where alpha=0.5 by default, E_r is the set of evidence sentence IDs for the
positive relation, and cooccur_freq is normalised co-occurrence frequency.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Type aliases for clarity
DocLabel  = Tuple[int, int, List[int]]       # (h_idx, t_idx, relation_ids)
DocEvid   = Tuple[int, int, int, List[int]]  # (h_idx, t_idx, r_id, evidence_sentence_ids)
PairKey   = Tuple[int, int]                  # (h_idx, t_idx)

_EPS = 1e-8


def docred_collate_item_to_miner_format(
    labels: Tensor,
    evidence: Dict[Tuple[int, int, int], List[int]],
) -> Tuple[List[DocLabel], List[DocEvid]]:
    """Convert :class:`DocREDDataset` / ``docred_collate_fn`` fields to the
    structures expected by :meth:`EvidenceNegativeMiner.update_statistics`.

    Parameters
    ----------
    labels
        FloatTensor ``[E, E, num_relations]`` with positive cells set to 1.
    evidence
        Mapping ``(h_idx, t_idx, r_id) -> supporting sentence indices``.

    Returns
    -------
    doc_labels
        List of ``(h, t, [r, ...])`` for each entity pair with a positive label.
    doc_evidence
        List of ``(h, t, r_id, evidence_sentence_ids)`` per annotated triple.
    """
    e = int(labels.size(0))
    doc_labels: List[DocLabel] = []
    for h in range(e):
        for t in range(e):
            if h == t:
                continue
            rels = (labels[h, t] > 0.5).nonzero(as_tuple=False).squeeze(-1).tolist()
            if rels:
                doc_labels.append((h, t, rels))
    doc_evidence: List[DocEvid] = [
        (h, t, r_id, list(sents)) for (h, t, r_id), sents in evidence.items()
    ]
    return doc_labels, doc_evidence


class EvidenceNegativeMiner:
    """
    Evidence-aware hard negative mining for DocRED relation extraction.

    Maintains running statistics (co-occurrence counts, evidence overlap)
    accumulated over training batches, and uses them to tier negatives into
    hard / medium / easy categories.

    Parameters
    ----------
    num_relations : int
        Total number of relation types (96 labelled + 1 NA = 97 by convention).
    evidence_overlap_threshold : float
        Minimum Jaccard evidence overlap to consider a candidate as hard.
    topk_hard : int
        Maximum number of hard negatives to return per anchor.
    hardness_alpha : float
        Interpolation weight between evidence overlap and co-occurrence in the
        hardness score: score = alpha * overlap + (1-alpha) * cooccur.
    """

    def __init__(
        self,
        num_relations: int = 97,
        evidence_overlap_threshold: float = 0.3,
        topk_hard: int = 10,
        hardness_alpha: float = 0.5,
    ) -> None:
        self.num_relations = num_relations
        self.evidence_overlap_threshold = evidence_overlap_threshold
        self.topk_hard = topk_hard
        self.hardness_alpha = hardness_alpha

        # Co-occurrence count matrix: cooccurrence_matrix[r1, r2] = # of times
        # r1 and r2 were both positive for the same entity pair in training.
        # Shape: [num_relations, num_relations]
        self.cooccurrence_matrix: Tensor = torch.zeros(
            num_relations, num_relations, dtype=torch.float32
        )

        # Cache: (r1, r2) → cumulative evidence overlap (Jaccard) sum and count,
        # used to compute running average.
        # evidence_overlap_cache[(r1, r2)] = [overlap_sum, count]
        self.evidence_overlap_cache: Dict[Tuple[int, int], List[float]] = defaultdict(
            lambda: [0.0, 0]
        )

    # ------------------------------------------------------------------
    # Statistics accumulation
    # ------------------------------------------------------------------

    def update_statistics(
        self,
        batch_labels: List[List[DocLabel]],
        batch_evidence: List[List[DocEvid]],
    ) -> None:
        """
        Update co-occurrence and evidence-overlap statistics from a training batch.

        Parameters
        ----------
        batch_labels : List[List[DocLabel]]
            Per-document list of (h_idx, t_idx, relation_ids) triples.
            Outer list = documents, inner list = labelled triples per document.
        batch_evidence : List[List[DocEvid]]
            Per-document list of (h_idx, t_idx, r_id, evidence_sentence_ids).
        """
        for doc_labels, doc_evidence in zip(batch_labels, batch_evidence):
            # --- Co-occurrence matrix update ---
            # Group relations by entity pair
            pair_to_relations: Dict[PairKey, List[int]] = defaultdict(list)
            for h_idx, t_idx, relation_ids in doc_labels:
                for r in relation_ids:
                    if r < self.num_relations:
                        pair_to_relations[(h_idx, t_idx)].append(r)

            for (h_idx, t_idx), rels in pair_to_relations.items():
                for i, r1 in enumerate(rels):
                    for r2 in rels[i:]:  # includes r1==r2 diagonal (self-co-occurrence)
                        self.cooccurrence_matrix[r1, r2] += 1.0
                        if r1 != r2:
                            self.cooccurrence_matrix[r2, r1] += 1.0

            # --- Evidence overlap cache update ---
            # Build per-pair-relation → evidence set mapping
            evid_map: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
            for h_idx, t_idx, r_id, evid_sents in doc_evidence:
                evid_map[(h_idx, t_idx, r_id)].update(evid_sents)

            # For every pair of relations sharing an entity pair, compute Jaccard
            for (h_idx, t_idx), rels in pair_to_relations.items():
                for i, r1 in enumerate(rels):
                    for r2 in rels[i + 1:]:
                        e1 = evid_map.get((h_idx, t_idx, r1), set())
                        e2 = evid_map.get((h_idx, t_idx, r2), set())
                        jaccard = _jaccard(e1, e2)
                        self.evidence_overlap_cache[(r1, r2)][0] += jaccard
                        self.evidence_overlap_cache[(r1, r2)][1] += 1
                        # symmetric
                        self.evidence_overlap_cache[(r2, r1)][0] += jaccard
                        self.evidence_overlap_cache[(r2, r1)][1] += 1

    def _get_cooccur_freq(self, r: int) -> Tensor:
        """
        Return normalised co-occurrence frequencies for relation r with all others.

        Parameters
        ----------
        r : int
            Relation index.

        Returns
        -------
        Tensor
            Shape [num_relations], values in [0, 1].
        """
        row = self.cooccurrence_matrix[r].clone()  # [num_relations]
        total = row.sum().clamp(min=_EPS)
        return row / total

    def _get_evidence_overlap(self, r1: int, r2: int) -> float:
        """
        Return the cached average evidence overlap (Jaccard) between r1 and r2.
        Falls back to 0.0 if no statistics have been accumulated.
        """
        entry = self.evidence_overlap_cache.get((r1, r2))
        if entry is None or entry[1] == 0:
            return 0.0
        return entry[0] / entry[1]

    # ------------------------------------------------------------------
    # Hard negative retrieval
    # ------------------------------------------------------------------

    def get_hard_negatives(
        self,
        h_idx: int,
        t_idx: int,
        positive_relations: List[int],
        evidence_sets: Dict[int, Set[int]],
        all_pair_embs: Optional[Tensor],
        all_relation_ids: Optional[List[int]],
    ) -> Dict[str, object]:
        """
        Retrieve hard / medium / easy negatives for an entity pair.

        Hardness is scored as:
            score(r') = alpha * evidence_overlap(r, r') + (1-alpha) * cooccur(r, r')

        averaged over all positive relations r.

        Parameters
        ----------
        h_idx : int
            Head entity index.
        t_idx : int
            Tail entity index.
        positive_relations : List[int]
            Known positive relation IDs for this (h, t) pair.
        evidence_sets : Dict[int, Set[int]]
            Mapping r_id → set of evidence sentence IDs for this pair.
        all_pair_embs : Optional[Tensor]
            Embeddings for *all* (h, t, r) triples in the batch, shape [B, dim].
            Used to incorporate embedding similarity into the hardness score.
            If None, only statistics-based scoring is used.
        all_relation_ids : Optional[List[int]]
            Relation IDs corresponding to each row in ``all_pair_embs``.

        Returns
        -------
        dict with keys:
            - "hard_neg_indices"   : List[int] — indices in all_relation_ids
            - "hard_neg_scores"    : List[float] — hardness in [0, 1]
            - "medium_neg_indices" : List[int]
            - "easy_neg_indices"   : List[int]
        """
        positive_set = set(positive_relations)
        all_relation_range = list(range(self.num_relations))

        # Candidates: any relation not in the positive set
        candidate_ids = [r for r in all_relation_range if r not in positive_set]

        if not candidate_ids:
            return {
                "hard_neg_indices": [],
                "hard_neg_scores": [],
                "medium_neg_indices": [],
                "easy_neg_indices": [],
            }

        # --- Score each candidate ---
        scores: Dict[int, float] = {}
        for r_neg in candidate_ids:
            # Average over positive relations
            overlap_score = 0.0
            cooccur_score = 0.0
            emb_score     = 0.0

            for r_pos in positive_relations:
                overlap_score += self._get_evidence_overlap(r_pos, r_neg)
                cooccur_freq = self._get_cooccur_freq(r_pos)
                cooccur_score += float(cooccur_freq[r_neg].item())

            if positive_relations:
                overlap_score /= len(positive_relations)
                cooccur_score /= len(positive_relations)

            # Statistics-based hardness
            stat_score = (
                self.hardness_alpha * overlap_score
                + (1.0 - self.hardness_alpha) * cooccur_score
            )
            scores[r_neg] = stat_score

        # Optionally incorporate embedding similarity
        if all_pair_embs is not None and all_relation_ids is not None:
            # Find positive anchor embeddings
            pos_indices = [
                i for i, rid in enumerate(all_relation_ids) if rid in positive_set
            ]
            neg_indices = [
                i for i, rid in enumerate(all_relation_ids) if rid not in positive_set
            ]
            if pos_indices and neg_indices:
                pos_embs = all_pair_embs[pos_indices]  # [P, dim]
                neg_embs = all_pair_embs[neg_indices]  # [Q, dim]
                pos_norm = F.normalize(pos_embs.float(), p=2, dim=-1)
                neg_norm = F.normalize(neg_embs.float(), p=2, dim=-1)
                # [P, Q] → mean over positives → [Q]
                cos_sim = torch.mm(pos_norm, neg_norm.t()).mean(dim=0)  # [Q]
                # Map [-1,1] → [0,1]
                cos_sim_01 = ((cos_sim + 1.0) / 2.0).tolist()

                # Blend embedding score into hardness (50/50 with stat_score)
                for batch_idx, rel_id in [
                    (neg_indices[i], all_relation_ids[neg_indices[i]])
                    for i in range(len(neg_indices))
                ]:
                    if rel_id in scores:
                        emb_score = cos_sim_01[neg_indices.index(batch_idx)]
                        scores[rel_id] = 0.5 * scores[rel_id] + 0.5 * emb_score

        # --- Tier into hard / medium / easy ---
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Map relation_id → batch index (uses all_relation_ids if provided)
        rel_to_batch_idx: Dict[int, int] = {}
        if all_relation_ids is not None:
            for i, rid in enumerate(all_relation_ids):
                rel_to_batch_idx[rid] = i

        hard_neg_indices:   List[int] = []
        hard_neg_scores:    List[float] = []
        medium_neg_indices: List[int] = []
        easy_neg_indices:   List[int] = []

        for r_neg, score in sorted_candidates:
            batch_idx = rel_to_batch_idx.get(r_neg, r_neg)  # fallback to relation id itself
            if score >= self.evidence_overlap_threshold:
                if len(hard_neg_indices) < self.topk_hard:
                    hard_neg_indices.append(batch_idx)
                    hard_neg_scores.append(score)
                else:
                    medium_neg_indices.append(batch_idx)
            elif score > 0.0:
                medium_neg_indices.append(batch_idx)
            else:
                easy_neg_indices.append(batch_idx)

        return {
            "hard_neg_indices":   hard_neg_indices,
            "hard_neg_scores":    hard_neg_scores,
            "medium_neg_indices": medium_neg_indices,
            "easy_neg_indices":   easy_neg_indices,
        }

    def sample_negatives(
        self,
        anchor_idx: int,
        batch_data: Dict[str, object],
        num_hard: int = 5,
        num_medium: int = 5,
        num_easy: int = 5,
    ) -> Dict[str, object]:
        """
        Sample a fixed number of negatives (hard / medium / easy) for one anchor.

        Combines evidence-aware statistics with in-batch random sampling so that
        the total budget ``num_hard + num_medium + num_easy`` is always met, filling
        from lower tiers if higher ones are exhausted.

        Parameters
        ----------
        anchor_idx : int
            Index of the anchor triple in the batch.
        batch_data : dict
            Must contain:
            - "relation_labels"  : Tensor [B, num_relations] or List[List[int]]
            - "pair_embs"        : Tensor [B, dim]
            - "entity_pair_ids"  : List[Tuple[int,int]]  (h_idx, t_idx) per sample
            - "evidence_sets"    : List[Dict[int, Set[int]]]  per sample

        Returns
        -------
        dict with keys:
            - "indices"  : Tensor[int]  sampled batch indices
            - "weights"  : Tensor[float]  hardness weights (higher = harder)
            - "tiers"    : List[str]  "hard" / "medium" / "easy" per sample
        """
        pair_embs       = batch_data.get("pair_embs")           # [B, dim]
        entity_pair_ids = batch_data.get("entity_pair_ids", []) # List[(h,t)]
        evidence_sets   = batch_data.get("evidence_sets",   []) # List[Dict]
        relation_labels = batch_data.get("relation_labels")

        B = len(entity_pair_ids)
        if B == 0:
            return {"indices": torch.tensor([], dtype=torch.long), "weights": torch.tensor([]), "tiers": []}

        # Extract anchor info
        h_idx, t_idx = entity_pair_ids[anchor_idx] if anchor_idx < len(entity_pair_ids) else (0, 0)
        anchor_evidence = evidence_sets[anchor_idx] if anchor_idx < len(evidence_sets) else {}

        # Determine positive relations for anchor
        if relation_labels is not None:
            if isinstance(relation_labels, Tensor):
                pos_rels = relation_labels[anchor_idx].nonzero(as_tuple=False).squeeze(-1).tolist()
            else:
                pos_rels = relation_labels[anchor_idx] if anchor_idx < len(relation_labels) else []
        else:
            pos_rels = []

        # All indices except the anchor
        all_indices = [i for i in range(B) if i != anchor_idx]
        if not all_indices:
            return {"indices": torch.tensor([], dtype=torch.long), "weights": torch.tensor([]), "tiers": []}

        # Build per-batch-item relation ids for get_hard_negatives
        batch_relation_ids: List[int] = []
        for i in all_indices:
            if relation_labels is not None:
                if isinstance(relation_labels, Tensor):
                    r_ids = relation_labels[i].nonzero(as_tuple=False).squeeze(-1).tolist()
                else:
                    r_ids = relation_labels[i] if i < len(relation_labels) else []
                r_id = r_ids[0] if r_ids else 0
            else:
                r_id = i % self.num_relations
            batch_relation_ids.append(r_id)

        # Get hardness tiers
        neg_info = self.get_hard_negatives(
            h_idx=h_idx,
            t_idx=t_idx,
            positive_relations=pos_rels,
            evidence_sets=anchor_evidence,
            all_pair_embs=(pair_embs[all_indices] if pair_embs is not None else None),
            all_relation_ids=batch_relation_ids,
        )

        # Map batch_indices references back to full-batch indices
        # neg_info indices refer to positions in all_indices
        def _remap(local_idxs: List[int]) -> List[int]:
            return [all_indices[i] for i in local_idxs if i < len(all_indices)]

        hard_pool   = _remap(neg_info["hard_neg_indices"])
        medium_pool = _remap(neg_info["medium_neg_indices"])
        easy_pool   = _remap(neg_info["easy_neg_indices"])

        # Sample from pools
        sampled_indices: List[int] = []
        sampled_tiers:   List[str] = []

        def _sample(pool: List[int], n: int, tier: str) -> None:
            chosen = random.sample(pool, min(n, len(pool)))
            sampled_indices.extend(chosen)
            sampled_tiers.extend([tier] * len(chosen))

        _sample(hard_pool,   num_hard,   "hard")
        _sample(medium_pool, num_medium, "medium")
        _sample(easy_pool,   num_easy,   "easy")

        # Fill shortage from lower tiers upward
        shortage = (num_hard + num_medium + num_easy) - len(sampled_indices)
        if shortage > 0:
            remaining = [i for i in all_indices if i not in sampled_indices]
            fill = random.sample(remaining, min(shortage, len(remaining)))
            sampled_indices.extend(fill)
            sampled_tiers.extend(["easy"] * len(fill))

        # Build hardness weights: hard=1.0, medium=0.6, easy=0.3
        tier_weight_map = {"hard": 1.0, "medium": 0.6, "easy": 0.3}
        weights = [tier_weight_map[t] for t in sampled_tiers]

        return {
            "indices": torch.tensor(sampled_indices, dtype=torch.long),
            "weights": torch.tensor(weights,         dtype=torch.float32),
            "tiers":   sampled_tiers,
        }


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _jaccard(a: Set[int], b: Set[int]) -> float:
    """Compute Jaccard similarity between two integer sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / (union + _EPS)
