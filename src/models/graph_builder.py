"""
graph_builder.py - Heterogeneous Document Graph Constructor
============================================================
Builds a PyTorch Geometric ``HeteroData`` object from a single document's
encoder outputs and linguistic annotations.

Node types
----------
  - ``"mention"``  : one node per entity mention span
  - ``"entity"``   : one node per unique entity (aggregated from mentions)
  - ``"sentence"`` : one node per sentence (optional, enabled by default)

Edge types
----------
  - (mention, in_sentence,   sentence) : mention → sentence it lives in
  - (sentence, contains,     mention)  : reverse of above
  - (mention,  coref,        mention)  : coreference links (same entity)
  - (entity,   has_mention,  mention)  : entity → all its mentions
  - (mention,  belongs_to,   entity)   : reverse of above
  - (sentence, adjacent,     sentence) : consecutive-sentence chain
  - (mention,  same_sent,    mention)  : two mentions share a sentence

All edges are stored as directed; the GNN can optionally make them
bidirectional by adding reverse edges.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import torch

try:
    from torch_geometric.data import HeteroData  # type: ignore
except ImportError as exc:
    raise ImportError(
        "torch_geometric is required. Install with: "
        "pip install torch-geometric"
    ) from exc

logger = logging.getLogger(__name__)


class DocGraphBuilder:
    """
    Build a heterogeneous document graph for one document at a time.

    Parameters
    ----------
    add_sentence_nodes : bool
        Whether to include sentence nodes and sentence-related edges.
        Disable to reduce graph size during ablation studies.
    add_self_loops : bool
        Whether to add self-loops to every node type.
    max_coref_mention_pairs_per_entity : int
        Cap undirected mention–mention pairs per entity when building coref
        cliques.  ``0`` means no cap (full clique).
    max_same_sent_mention_pairs_per_sentence : int
        Cap undirected pairs per sentence for ``same_sent`` edges.  ``0`` = no cap.
    """

    def __init__(
        self,
        add_sentence_nodes: bool = True,
        add_self_loops: bool = True,
        max_coref_mention_pairs_per_entity: int = 0,
        max_same_sent_mention_pairs_per_sentence: int = 0,
    ) -> None:
        self.add_sentence_nodes = add_sentence_nodes
        self.add_self_loops = add_self_loops
        self.max_coref_mention_pairs_per_entity = max_coref_mention_pairs_per_entity
        self.max_same_sent_mention_pairs_per_sentence = (
            max_same_sent_mention_pairs_per_sentence
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graph(self, doc_info: Dict) -> HeteroData:
        """
        Construct a HeteroData graph for one document.

        Parameters
        ----------
        doc_info : dict
            Expected keys:

            ``"entity_embeddings"`` : Tensor [num_entities, hidden_dim]
                Entity-level embeddings from the encoder.
            ``"mention_embeddings"`` : Tensor [num_mentions, hidden_dim]
                Mention-level embeddings from the encoder.
            ``"sentence_embeddings"`` : Tensor [num_sentences, hidden_dim]  (optional)
                Sentence-level embeddings (e.g. mean-pooled from encoder).
                If absent and ``add_sentence_nodes=True``, dummy zero vectors are used.
            ``"entity_spans"`` : list of list of (start, end)
                ``entity_spans[i]`` is the list of token-span tuples for entity *i*.
            ``"sentences"`` : list of (sent_start, sent_end) token-index tuples
                Sentence boundaries in the tokenized document.
            ``"mention_to_entity"`` : list[int]
                Flat list mapping mention index → entity index.
                Length == total number of mentions across all entities.
            ``"coreference_chains"`` : list of list[int]  (optional)
                Each inner list is a cluster of mention indices that are
                coreferent.  When omitted, coreference edges are derived
                purely from ``mention_to_entity`` (all mentions of the same
                entity are linked).

        Returns
        -------
        HeteroData
        """
        # ----------------------------------------------------------------
        # Unpack doc_info
        # ----------------------------------------------------------------
        entity_embs: torch.Tensor = doc_info["entity_embeddings"]
        mention_embs: torch.Tensor = doc_info["mention_embeddings"]
        entity_spans: List[List[Tuple[int, int]]] = doc_info["entity_spans"]
        sentences: List[Tuple[int, int]] = doc_info.get("sentences", [])
        mention_to_entity: List[int] = doc_info["mention_to_entity"]
        coref_chains: Optional[List[List[int]]] = doc_info.get("coreference_chains", None)

        num_entities = entity_embs.size(0)
        num_mentions = mention_embs.size(0)
        num_sentences = len(sentences)

        # ----------------------------------------------------------------
        # Derive sentence embeddings if not provided
        # ----------------------------------------------------------------
        if "sentence_embeddings" in doc_info:
            sent_embs: torch.Tensor = doc_info["sentence_embeddings"]
        else:
            hidden_dim = entity_embs.size(1) if entity_embs.numel() > 0 else mention_embs.size(1)
            sent_embs = entity_embs.new_zeros(max(num_sentences, 0), hidden_dim)

        # ----------------------------------------------------------------
        # Flatten mention spans
        # ----------------------------------------------------------------
        flat_spans: List[Tuple[int, int]] = []
        for ent_spans in entity_spans:
            for span in ent_spans:
                flat_spans.append(span)

        assert len(flat_spans) == num_mentions, (
            f"flat_spans length {len(flat_spans)} != num_mentions {num_mentions}"
        )

        # ----------------------------------------------------------------
        # Build HeteroData
        # ----------------------------------------------------------------
        data = HeteroData()

        # ---- Node features ---------------------------------------------
        data["mention"].x = mention_embs       # [num_mentions, hidden_dim]
        data["entity"].x = entity_embs         # [num_entities, hidden_dim]
        if self.add_sentence_nodes and num_sentences > 0:
            data["sentence"].x = sent_embs     # [num_sentences, hidden_dim]

        # ---- (entity, has_mention, mention) and reverse ----------------
        if num_mentions > 0 and num_entities > 0:
            ent_src = torch.tensor(mention_to_entity, dtype=torch.long)  # [num_mentions]
            ment_dst = torch.arange(num_mentions, dtype=torch.long)
            data["entity", "has_mention", "mention"].edge_index = torch.stack(
                [ent_src, ment_dst], dim=0
            )  # [2, num_mentions]
            data["mention", "belongs_to", "entity"].edge_index = torch.stack(
                [ment_dst, ent_src], dim=0
            )  # [2, num_mentions]

        # ---- Coreference edges: (mention, coref, mention) -------------
        if num_mentions > 0:
            coref_src: List[int] = []
            coref_dst: List[int] = []

            if coref_chains is not None:
                for chain in coref_chains:
                    cs, cd = self._limited_clique_edges(
                        chain,
                        self.max_coref_mention_pairs_per_entity,
                    )
                    coref_src.extend(cs)
                    coref_dst.extend(cd)
            else:
                # Use mention_to_entity to infer coreferent pairs
                from collections import defaultdict
                entity_to_mentions: Dict[int, List[int]] = defaultdict(list)
                for m_idx, e_idx in enumerate(mention_to_entity):
                    entity_to_mentions[e_idx].append(m_idx)
                for e_idx, m_list in entity_to_mentions.items():
                    cs, cd = self._limited_clique_edges(
                        m_list,
                        self.max_coref_mention_pairs_per_entity,
                    )
                    coref_src.extend(cs)
                    coref_dst.extend(cd)

            if coref_src:
                data["mention", "coref", "mention"].edge_index = torch.tensor(
                    [coref_src, coref_dst], dtype=torch.long
                )  # [2, num_coref_edges]
            else:
                data["mention", "coref", "mention"].edge_index = torch.zeros(
                    2, 0, dtype=torch.long
                )

        # ---- Sentence-related edges ------------------------------------
        if self.add_sentence_nodes and num_sentences > 0:
            # Derive sentence boundary list: [(sent_start, sent_end), ...]
            sentence_boundaries = sentences  # already list of (start, end)

            # (mention, in_sentence, sentence) and reverse
            m_in_s_src: List[int] = []
            m_in_s_dst: List[int] = []
            for m_idx, (m_start, _m_end) in enumerate(flat_spans):
                s_idx = self._get_sentence_idx(m_start, sentence_boundaries)
                if s_idx >= 0:
                    m_in_s_src.append(m_idx)
                    m_in_s_dst.append(s_idx)

            if m_in_s_src:
                data["mention", "in_sentence", "sentence"].edge_index = torch.tensor(
                    [m_in_s_src, m_in_s_dst], dtype=torch.long
                )
                data["sentence", "contains", "mention"].edge_index = torch.tensor(
                    [m_in_s_dst, m_in_s_src], dtype=torch.long
                )
            else:
                data["mention", "in_sentence", "sentence"].edge_index = torch.zeros(2, 0, dtype=torch.long)
                data["sentence", "contains", "mention"].edge_index = torch.zeros(2, 0, dtype=torch.long)

            # (sentence, adjacent, sentence) – consecutive chain
            if num_sentences > 1:
                adj_src = list(range(num_sentences - 1))
                adj_dst = list(range(1, num_sentences))
                # Bidirectional
                adj_src_bi = adj_src + adj_dst
                adj_dst_bi = adj_dst + adj_src
                data["sentence", "adjacent", "sentence"].edge_index = torch.tensor(
                    [adj_src_bi, adj_dst_bi], dtype=torch.long
                )
            else:
                data["sentence", "adjacent", "sentence"].edge_index = torch.zeros(2, 0, dtype=torch.long)

            # (mention, same_sent, mention) – two mentions share a sentence
            # Build sentence → mention list first
            from collections import defaultdict
            sent_to_mentions: Dict[int, List[int]] = defaultdict(list)
            for m_idx, s_idx in zip(m_in_s_src, m_in_s_dst):
                sent_to_mentions[s_idx].append(m_idx)

            ss_src: List[int] = []
            ss_dst: List[int] = []
            for s_idx, m_list in sent_to_mentions.items():
                cs, cd = self._limited_clique_edges(
                    m_list,
                    self.max_same_sent_mention_pairs_per_sentence,
                )
                ss_src.extend(cs)
                ss_dst.extend(cd)

            if ss_src:
                data["mention", "same_sent", "mention"].edge_index = torch.tensor(
                    [ss_src, ss_dst], dtype=torch.long
                )
            else:
                data["mention", "same_sent", "mention"].edge_index = torch.zeros(
                    2, 0, dtype=torch.long
                )

        # ---- Optional self-loops (on entity nodes for stability) ------
        if self.add_self_loops and num_entities > 0:
            se = torch.arange(num_entities, dtype=torch.long)
            data["entity", "self_loop", "entity"].edge_index = torch.stack([se, se], dim=0)

        if self.add_self_loops and num_mentions > 0:
            sm = torch.arange(num_mentions, dtype=torch.long)
            data["mention", "self_loop", "mention"].edge_index = torch.stack([sm, sm], dim=0)

        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _limited_clique_edges(
        mention_indices: List[int],
        max_undirected_pairs: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Build bidirectional edges for an undirected clique over ``mention_indices``,
        optionally subsampling undirected pairs when the clique is large.
        """
        m_list = list(mention_indices)
        pairs: List[Tuple[int, int]] = [
            (m_list[i], m_list[j])
            for i in range(len(m_list))
            for j in range(i + 1, len(m_list))
        ]
        if max_undirected_pairs > 0 and len(pairs) > max_undirected_pairs:
            pairs = random.sample(pairs, max_undirected_pairs)
        src: List[int] = []
        dst: List[int] = []
        for mi, mj in pairs:
            src.extend([mi, mj])
            dst.extend([mj, mi])
        return src, dst

    @staticmethod
    def _get_sentence_idx(
        mention_start: int,
        sentence_boundaries: List[Tuple[int, int]],
    ) -> int:
        """
        Return the 0-based sentence index that contains ``mention_start``.

        Parameters
        ----------
        mention_start : int
            Token index of the mention's first token.
        sentence_boundaries : list of (sent_start, sent_end)
            Each tuple is the half-open [start, end) token range of a sentence.

        Returns
        -------
        int
            Sentence index, or ``-1`` if no sentence boundary contains the token.
        """
        for s_idx, (s_start, s_end) in enumerate(sentence_boundaries):
            if s_start <= mention_start < s_end:
                return s_idx
        # Fallback: assign to the last sentence if slightly out of range
        if sentence_boundaries:
            return len(sentence_boundaries) - 1
        return -1

    # ------------------------------------------------------------------
    # Batch interface
    # ------------------------------------------------------------------

    def build_graphs(self, doc_infos: List[Dict]) -> List[HeteroData]:
        """Build a graph for each document in a list.

        Parameters
        ----------
        doc_infos : list[dict]
            Each element is a ``doc_info`` dict as expected by :meth:`build_graph`.

        Returns
        -------
        list[HeteroData]
        """
        return [self.build_graph(di) for di in doc_infos]
