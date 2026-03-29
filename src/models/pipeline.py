"""
pipeline.py – End-to-end DocRED Relation Extraction Pipeline
=============================================================
``DocREDPipeline`` wires together all four sub-modules into a single
``nn.Module`` with a clean ``forward`` interface.

Forward pass::

    batch  ──►  DocumentEncoder       (PLM tokenisation → entity embeddings)
            ──►  DocGraphBuilder      (build HeteroData per doc)
            ──►  DocGraphReasoner     (GNN reasoning → refined entity embs)
            ──►  TripleHead × pairs   (relation logits per entity pair)
            ──►  output dict

The pipeline also exposes:
  - ``get_contrastive_outputs`` for computing BMM-reweighted GCL loss.
  - ``predict`` for inference with adaptive thresholding.

Configuration dict keys
-----------------------
All keys with defaults::

    plm_name           : str   = "microsoft/deberta-v3-large"
    use_lora           : bool  = False
    lora_rank          : int   = 8
    add_sentence_nodes : bool  = True
    gnn_hidden_dim     : int   = 256
    gnn_out_dim        : int   = 256
    gnn_layers         : int   = 3
    gnn_heads          : int   = 4
    gnn_bases          : int   = 4
    gnn_dropout        : float = 0.1
    rel_dim            : int   = 128
    num_relations      : int   = 97
    triple_dim         : int   = 512
    contrastive_dim    : int   = 256
    head_dropout       : float = 0.1
    max_pairs_per_doc  : int   = -1   # -1 = no limit
"""

from __future__ import annotations

import logging
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import DocumentEncoder
from src.models.graph_builder import DocGraphBuilder
from src.models.gnn import DocGraphReasoner
from src.models.triple_head import TripleHead

try:
    from torch_geometric.data import HeteroData  # type: ignore
except ImportError as exc:
    raise ImportError(
        "torch_geometric is required. Install with: pip install torch-geometric"
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cfg(cfg: Dict, key: str, default):
    """Safe config getter with default."""
    return cfg.get(key, default)


# ---------------------------------------------------------------------------
# DocREDPipeline
# ---------------------------------------------------------------------------

class DocREDPipeline(nn.Module):
    """
    Full DocRED relation-extraction pipeline.

    Parameters
    ----------
    config : dict
        Hyper-parameter dictionary.  See module docstring for all keys.

    Examples
    --------
    Minimal construction::

        pipe = DocREDPipeline({"plm_name": "roberta-base", "num_relations": 97})

    Forward call::

        out = pipe(batch)
        loss = criterion(out["logits"], out["labels"])
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config

        # ----------------------------------------------------------------
        # 1. Document Encoder
        # ----------------------------------------------------------------
        plm_name: str = _get_cfg(config, "plm_name", "microsoft/deberta-v3-large")
        use_lora: bool = _get_cfg(config, "use_lora", False)
        lora_rank: int = _get_cfg(config, "lora_rank", 8)

        self.encoder = DocumentEncoder(
            plm_name=plm_name,
            use_lora=use_lora,
            lora_rank=lora_rank,
        )
        plm_hidden: int = self.encoder.get_hidden_dim()

        # ----------------------------------------------------------------
        # 2. Document Graph Builder (stateless, not nn.Module)
        # ----------------------------------------------------------------
        add_sent: bool = _get_cfg(config, "add_sentence_nodes", True)
        self.graph_builder = DocGraphBuilder(add_sentence_nodes=add_sent)

        # ----------------------------------------------------------------
        # 3. GNN Reasoner
        # ----------------------------------------------------------------
        gnn_hidden: int = _get_cfg(config, "gnn_hidden_dim", 256)
        gnn_out: int = _get_cfg(config, "gnn_out_dim", 256)
        gnn_layers: int = _get_cfg(config, "gnn_layers", 3)
        gnn_heads: int = _get_cfg(config, "gnn_heads", 4)
        gnn_bases: int = _get_cfg(config, "gnn_bases", 4)
        gnn_dropout: float = _get_cfg(config, "gnn_dropout", 0.1)

        self.gnn = DocGraphReasoner(
            in_dim=plm_hidden,
            hidden_dim=gnn_hidden,
            out_dim=gnn_out,
            num_layers=gnn_layers,
            num_heads=gnn_heads,
            num_bases=gnn_bases,
            dropout=gnn_dropout,
        )

        # ----------------------------------------------------------------
        # 4. Triple Head / Relation Classifier
        # ----------------------------------------------------------------
        rel_dim: int = _get_cfg(config, "rel_dim", 128)
        num_relations: int = _get_cfg(config, "num_relations", 97)
        triple_dim: int = _get_cfg(config, "triple_dim", 512)
        contrastive_dim: int = _get_cfg(config, "contrastive_dim", 256)
        head_dropout: float = _get_cfg(config, "head_dropout", 0.1)

        self.triple_head = TripleHead(
            entity_dim=gnn_out,
            rel_dim=rel_dim,
            num_relations=num_relations,
            triple_dim=triple_dim,
            contrastive_dim=contrastive_dim,
            dropout=head_dropout,
        )

        # ----------------------------------------------------------------
        # Sentence-level mean pooling (for context embeddings)
        # ----------------------------------------------------------------
        # Linear that projects PLM hidden → GNN out_dim for context
        self.sent_context_proj = nn.Linear(plm_hidden, gnn_out, bias=False)

        # Misc
        self.max_pairs_per_doc: int = _get_cfg(config, "max_pairs_per_doc", -1)
        self.num_relations = num_relations
        self.gnn_out = gnn_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Dict) -> Dict:
        """
        End-to-end forward pass.

        Parameters
        ----------
        batch : dict
            Expected keys:

            ``"input_ids"`` : LongTensor [batch_size, seq_len]
            ``"attention_mask"`` : LongTensor [batch_size, seq_len]
            ``"entity_spans"`` : list[list[list[tuple]]]
                entity_spans[b][e] = list of (start, end) token spans for entity e in doc b.
            ``"mention_to_entity"`` : list[list[int]]
                mention_to_entity[b] = flat list mapping mention → entity index.
            ``"sentences"`` : list[list[tuple]]
                sentences[b] = list of (sent_start, sent_end) token boundaries.
            ``"labels"`` : optional LongTensor [total_pairs, num_relations]
                Multi-hot relation labels per entity pair.
                If absent, an empty tensor is returned.
            ``"coreference_chains"`` : optional list[list[list[int]]]

        Returns
        -------
        dict with keys
            ``"logits"``         : Tensor [total_pairs, num_relations]
            ``"pair_embs"``      : Tensor [total_pairs, triple_dim]
            ``"labels"``         : Tensor [total_pairs, num_relations]  (pass-through)
            ``"entity_pair_ids"``  : list of (doc_idx, h_idx, t_idx) per pair
            ``"evidence_sets"``  : list of set[int] (evidence sentence indices per pair)
        """
        device = batch["input_ids"].device
        batch_size = batch["input_ids"].size(0)

        # ----------------------------------------------------------------
        # Step 1 – Encode documents
        # ----------------------------------------------------------------
        enc_out = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            entity_spans=batch["entity_spans"],
            mention_to_entity=batch.get("mention_to_entity"),
        )
        token_embs: torch.Tensor = enc_out["token_embeddings"]   # [B, seq_len, H]
        mention_embs_list: List[torch.Tensor] = enc_out["mention_embeddings"]
        entity_embs_list: List[torch.Tensor] = enc_out["entity_embeddings"]

        # ----------------------------------------------------------------
        # Step 2 – Build document graphs + Step 3 – GNN reasoning
        # ----------------------------------------------------------------
        all_logits: List[torch.Tensor] = []
        all_pair_embs: List[torch.Tensor] = []
        all_pair_ids: List[Tuple[int, int, int]] = []
        all_evidence_sets: List[object] = []

        all_pair_label_chunks: List[torch.Tensor] = []
        labels_per_doc = batch.get("labels_per_doc")

        for doc_idx in range(batch_size):
            doc_entity_embs = entity_embs_list[doc_idx]   # [num_ent, H]
            doc_mention_embs = mention_embs_list[doc_idx]  # [num_men, H]

            num_entities = doc_entity_embs.size(0)
            if num_entities < 2:
                # Need at least 2 entities to form a pair
                continue

            sentences: List[Tuple[int, int]] = (
                batch["sentences"][doc_idx]
                if "sentences" in batch else []
            )

            # ---- Sentence embeddings: mean-pool token embs per sentence
            sent_embs = self._compute_sentence_embeddings(
                token_embs[doc_idx], sentences
            )  # [num_sents, H]

            # ---- Mention-to-entity mapping
            mention_to_entity: List[int] = (
                batch["mention_to_entity"][doc_idx]
                if "mention_to_entity" in batch
                else self._derive_mention_to_entity(batch["entity_spans"][doc_idx])
            )

            # ---- Build doc_info for graph builder
            doc_info = {
                "entity_embeddings": doc_entity_embs,
                "mention_embeddings": doc_mention_embs,
                "sentence_embeddings": sent_embs,
                "entity_spans": batch["entity_spans"][doc_idx],
                "sentences": sentences,
                "mention_to_entity": mention_to_entity,
                "coreference_chains": (
                    batch["coreference_chains"][doc_idx]
                    if "coreference_chains" in batch else None
                ),
            }

            # ---- Build heterogeneous graph
            hetero_graph: HeteroData = self.graph_builder.build_graph(doc_info)
            # Move graph tensors to device
            hetero_graph = hetero_graph.to(device)

            # ---- GNN reasoning → refined entity embeddings
            gnn_out = self.gnn(hetero_graph)
            refined_entity_embs = gnn_out.get("entity")  # [num_ent, gnn_out]
            if refined_entity_embs is None or refined_entity_embs.numel() == 0:
                refined_entity_embs = doc_entity_embs  # fallback: use encoder output
                # Project to GNN output dimension if needed
                if refined_entity_embs.size(-1) != self.gnn_out:
                    logger.warning(
                        "GNN returned empty entity embeddings for doc %d; "
                        "using encoder output (may have dim mismatch).",
                        doc_idx,
                    )

            # ---- Step 4 – Enumerate entity pairs and compute triple head
            pair_list = self._pair_list_for_doc(num_entities)

            doc_logits, doc_pair_embs, doc_pair_ids, doc_evidence = (
                self._compute_pairs(
                    doc_idx=doc_idx,
                    entity_embs=refined_entity_embs,
                    sent_embs_gnn=gnn_out.get("sentence"),
                    evidence_map=(
                        batch.get("evidence_map", [None] * batch_size)[doc_idx]
                    ),
                )
            )

            if labels_per_doc is not None:
                if (
                    doc_idx < len(labels_per_doc)
                    and labels_per_doc[doc_idx] is not None
                ):
                    lab_3d = labels_per_doc[doc_idx]
                    if lab_3d.dim() == 3 and lab_3d.size(0) >= num_entities:
                        all_pair_label_chunks.append(
                            self._gather_pair_labels(lab_3d, pair_list)
                        )
                    else:
                        all_pair_label_chunks.append(
                            doc_logits.new_zeros(
                                doc_logits.size(0), self.num_relations
                            )
                        )
                else:
                    all_pair_label_chunks.append(
                        doc_logits.new_zeros(
                            doc_logits.size(0), self.num_relations
                        )
                    )

            all_logits.append(doc_logits)
            all_pair_embs.append(doc_pair_embs)
            all_pair_ids.extend(doc_pair_ids)
            all_evidence_sets.extend(doc_evidence)

        # ----------------------------------------------------------------
        # Step 5 – Collect outputs
        # ----------------------------------------------------------------
        if all_logits:
            logits = torch.cat(all_logits, dim=0)      # [total_pairs, num_relations]
            pair_embs = torch.cat(all_pair_embs, dim=0)  # [total_pairs, triple_dim]
        else:
            logits = batch["input_ids"].new_zeros(0, self.num_relations).float()
            pair_embs = batch["input_ids"].new_zeros(
                0, self.triple_head.triple_dim
            ).float()

        # Labels aligned with logits / pair_embs (same pair order & truncation as triple head)
        if all_pair_label_chunks:
            labels = torch.cat(all_pair_label_chunks, dim=0)
        else:
            flat = batch.get("labels")
            if isinstance(flat, torch.Tensor) and flat.dim() == 2:
                if flat.size(0) == logits.size(0):
                    labels = flat
                else:
                    logger.warning(
                        "Flat ``labels`` rows (%d) != logits rows (%d); "
                        "pass ``labels_per_doc`` from training. Using zeros.",
                        flat.size(0),
                        logits.size(0),
                    )
                    labels = logits.new_zeros(logits.size(0), self.num_relations)
            else:
                labels = logits.new_zeros(logits.size(0), self.num_relations)

        return {
            "logits": logits,               # [total_pairs, num_relations]
            "pair_embs": pair_embs,         # [total_pairs, triple_dim]
            "labels": labels,               # [total_pairs, num_relations]
            "entity_pair_ids": all_pair_ids,  # list of (doc_idx, h_idx, t_idx)
            "evidence_sets": all_evidence_sets,
        }

    # ------------------------------------------------------------------
    # Contrastive outputs
    # ------------------------------------------------------------------

    def get_contrastive_outputs(
        self,
        pair_embs: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute triple embeddings and contrastive projections for GCL loss.

        Parameters
        ----------
        pair_embs : Tensor [batch_pairs, triple_dim]
            From ``forward()["pair_embs"]``.
        relation_ids : LongTensor [batch_pairs]
            Relation index per pair (use the primary positive relation).

        Returns
        -------
        dict with keys
            ``"triple_embs"``      : Tensor [batch_pairs, triple_dim]
            ``"contrastive_embs"`` : Tensor [batch_pairs, contrastive_dim]
        """
        triple_embs = self.triple_head.get_triple_emb(pair_embs, relation_ids)
        contrastive_embs = self.triple_head.get_contrastive_emb(triple_embs)
        return {
            "triple_embs": triple_embs,
            "contrastive_embs": contrastive_embs,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: Dict) -> Dict:
        """
        Run inference with adaptive thresholding.

        Returns the same dict as ``forward`` plus:
            ``"predictions"`` : BoolTensor [total_pairs, num_relations]
            ``"thresholds"``  : Tensor [total_pairs, num_relations]
        """
        out = self.forward(batch)
        pred_out = self.triple_head.adaptive_threshold.predict(
            out["logits"], out["pair_embs"]
        )
        out["predictions"] = pred_out
        out["thresholds"] = self.triple_head.adaptive_threshold(out["pair_embs"])
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pair_list_for_doc(self, num_entities: int) -> List[Tuple[int, int]]:
        """Ordered (h, t) pairs, identical to :meth:`_compute_pairs` (incl. cap)."""
        pair_list = [
            (h, t)
            for h in range(num_entities)
            for t in range(num_entities)
            if h != t
        ]
        if self.max_pairs_per_doc > 0:
            pair_list = pair_list[: self.max_pairs_per_doc]
        return pair_list

    @staticmethod
    def _gather_pair_labels(
        labels_3d: torch.Tensor,
        pair_list: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """``labels_3d`` [E, E, R] → rows [num_pairs, R] for the given pairs."""
        if not pair_list:
            return labels_3d.new_zeros(0, labels_3d.size(-1))
        rows = [labels_3d[h, t] for h, t in pair_list]
        return torch.stack(rows, dim=0)

    def _compute_pairs(
        self,
        doc_idx: int,
        entity_embs: torch.Tensor,
        sent_embs_gnn: Optional[torch.Tensor],
        evidence_map: Optional[Dict[Tuple[int, int], List[int]]],
    ) -> Tuple[
        torch.Tensor,             # logits [num_pairs, num_relations]
        torch.Tensor,             # pair_embs [num_pairs, triple_dim]
        List[Tuple[int, int, int]],  # (doc_idx, h_idx, t_idx)
        List[object],             # evidence sets
    ]:
        """
        Enumerate all directed entity pairs, build context embeddings,
        and run TripleHead.

        Parameters
        ----------
        doc_idx : int
        entity_embs : Tensor [num_entities, gnn_out]
        sent_embs_gnn : optional Tensor [num_sentences, hidden_dim]
        evidence_map : optional dict mapping (h, t) → list of sent indices

        Returns
        -------
        (logits, pair_embs, pair_ids, evidence_sets)
        """
        num_entities = entity_embs.size(0)
        pair_list = self._pair_list_for_doc(num_entities)

        if not pair_list:
            device = entity_embs.device
            empty_logits = entity_embs.new_zeros(0, self.num_relations)
            empty_pair_embs = entity_embs.new_zeros(0, self.triple_head.triple_dim)
            return empty_logits, empty_pair_embs, [], []

        h_indices = torch.tensor([p[0] for p in pair_list], dtype=torch.long,
                                  device=entity_embs.device)
        t_indices = torch.tensor([p[1] for p in pair_list], dtype=torch.long,
                                  device=entity_embs.device)

        h_embs = entity_embs[h_indices]  # [num_pairs, gnn_out]
        t_embs = entity_embs[t_indices]  # [num_pairs, gnn_out]

        # ---- Context embeddings from evidence sentences ----------------
        context_embs = self._build_context_embs(
            pair_list, sent_embs_gnn, evidence_map, entity_embs.device
        )  # [num_pairs, gnn_out] or None

        # ---- Triple head forward ---------------------------------------
        head_out = self.triple_head(h_embs, t_embs, context_embs)

        logits: torch.Tensor = head_out["logits"]      # [num_pairs, num_relations]
        pair_embs: torch.Tensor = head_out["pair_emb"]  # [num_pairs, triple_dim]

        # ---- Build pair ID list and evidence sets ----------------------
        pair_ids = [(doc_idx, h, t) for (h, t) in pair_list]
        evidence_sets: List[object] = []
        for h, t in pair_list:
            if evidence_map is not None and (h, t) in evidence_map:
                evidence_sets.append(set(evidence_map[(h, t)]))
            else:
                evidence_sets.append(set())

        return logits, pair_embs, pair_ids, evidence_sets

    def _build_context_embs(
        self,
        pair_list: List[Tuple[int, int]],
        sent_embs_gnn: Optional[torch.Tensor],
        evidence_map: Optional[Dict[Tuple[int, int], List[int]]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Build context embeddings for each entity pair by mean-pooling the
        evidence sentence embeddings (if available).

        Returns
        -------
        Tensor [num_pairs, gnn_out] or None if no sentence embeddings exist.
        """
        if sent_embs_gnn is None or sent_embs_gnn.numel() == 0:
            return None

        num_pairs = len(pair_list)
        context = sent_embs_gnn.new_zeros(num_pairs, sent_embs_gnn.size(-1))

        # Project sentence embeddings to gnn_out if they differ in dim
        if sent_embs_gnn.size(-1) != self.gnn_out:
            # Lazy projection
            if not hasattr(self, "_sent_proj_lazy"):
                self._sent_proj_lazy = nn.Linear(
                    sent_embs_gnn.size(-1), self.gnn_out, bias=False
                ).to(device)
            sent_embs_gnn = self._sent_proj_lazy(sent_embs_gnn)

        for i, (h, t) in enumerate(pair_list):
            if evidence_map is not None and (h, t) in evidence_map:
                ev_sents = evidence_map[(h, t)]
                valid_ev = [
                    s for s in ev_sents if 0 <= s < sent_embs_gnn.size(0)
                ]
                if valid_ev:
                    ev_idx = torch.tensor(valid_ev, dtype=torch.long, device=device)
                    context[i] = sent_embs_gnn[ev_idx].mean(0)
            # Fallback: use average of all sentence embeddings as context
            else:
                context[i] = sent_embs_gnn.mean(0)

        return context

    @staticmethod
    def _compute_sentence_embeddings(
        token_embs: torch.Tensor,
        sentences: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Compute sentence-level embeddings by mean-pooling token embeddings.

        Parameters
        ----------
        token_embs : Tensor [seq_len, hidden_dim]
        sentences : list of (start, end) token index tuples (half-open)

        Returns
        -------
        Tensor [num_sentences, hidden_dim]
        """
        if not sentences:
            return token_embs.new_zeros(0, token_embs.size(-1))

        hidden_dim = token_embs.size(-1)
        sent_embs = token_embs.new_zeros(len(sentences), hidden_dim)

        for s_idx, (s_start, s_end) in enumerate(sentences):
            s_start = max(0, s_start)
            s_end = min(token_embs.size(0), s_end)
            if s_start < s_end:
                sent_embs[s_idx] = token_embs[s_start:s_end].mean(0)

        return sent_embs

    @staticmethod
    def _derive_mention_to_entity(
        entity_spans: List[List[Tuple[int, int]]],
    ) -> List[int]:
        """
        Derive flat mention→entity mapping from nested entity_spans.

        Parameters
        ----------
        entity_spans : entity_spans[e] = list of (start, end) for entity e

        Returns
        -------
        list[int] of length = total_mentions
        """
        mapping: List[int] = []
        for e_idx, spans in enumerate(entity_spans):
            for _ in spans:
                mapping.append(e_idx)
        return mapping
