"""
Triple-Level Graph Builder for Graph Contrastive Learning.

Constructs a node-per-entity-pair graph used in the GCL stage of the pipeline.
Each node represents an entity pair (h, t) in the current batch.  Edges connect
pairs that are "structurally related" via one or more of:

  1. High embedding cosine similarity (> threshold)
  2. Shared entity (same head or same tail)
  3. Short graph-structural distance in the document graph

Edge weights are a composite of these three similarity signals.

The output is a ``torch_geometric.data.Data`` object ready for message-passing.
If PyTorch Geometric is not installed, a lightweight fallback adjacency-list
representation is provided as a plain dict.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Try importing PyG; fall back gracefully
try:
    from torch_geometric.data import Data as PyGData  # type: ignore

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    logger.warning(
        "torch_geometric not found. TripleGraphBuilder will return a plain dict "
        "instead of a PyG Data object."
    )

_EPS = 1e-8

PairID = Tuple[int, int]  # (h_entity_idx, t_entity_idx)


class TripleGraphBuilder:
    """
    Builds a triple-level graph (one node per entity pair) for GCL.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity between pair embeddings to connect two nodes
        via an "embedding similarity" edge.
    max_neighbors : int
        Maximum number of neighbors per node (retains top-k by edge weight).
    emb_weight : float
        Contribution of embedding similarity to composite edge weight.
    shared_entity_weight : float
        Contribution of shared-entity indicator to composite edge weight.
    graph_dist_weight : float
        Contribution of graph-proximity signal to composite edge weight.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        max_neighbors: int = 20,
        emb_weight: float = 0.5,
        shared_entity_weight: float = 0.3,
        graph_dist_weight: float = 0.2,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self.emb_weight = emb_weight
        self.shared_entity_weight = shared_entity_weight
        self.graph_dist_weight = graph_dist_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_triple_graph(
        self,
        pair_embs: Tensor,
        entity_pair_ids: List[PairID],
        doc_graph_distances: Optional[Dict[Tuple[int, int], float]] = None,
        relation_labels: Optional[Tensor] = None,
    ) -> "PyGData | dict":
        """
        Build the triple-level graph.

        Parameters
        ----------
        pair_embs : Tensor
            Node (entity-pair) feature vectors, shape [N, dim].
        entity_pair_ids : List[PairID]
            List of (h_idx, t_idx) for each node, length N.
        doc_graph_distances : Optional[Dict[Tuple[int,int], float]]
            Pre-computed shortest-path distances between entity-pairs in the
            document graph.  Keys are (pair_i, pair_j) where pair_i < pair_j.
            If None, the graph-distance term is omitted.
        relation_labels : Optional[Tensor]
            Multi-hot relation labels per pair, shape [N, num_relations].
            Currently unused in edge computation but stored as node attribute.

        Returns
        -------
        PyGData | dict
            PyG Data with:
                - x           : node features [N, dim]
                - edge_index  : [2, E]
                - edge_weight : [E]
            Or plain dict with the same keys if PyG is unavailable.
        """
        N = pair_embs.shape[0]
        device = pair_embs.device

        if N == 0:
            return self._build_empty_graph(pair_embs, device)

        # ---- 1. Compute pairwise cosine similarity ----
        emb_norm = F.normalize(pair_embs.float(), p=2, dim=-1)  # [N, dim]
        cos_sim   = torch.mm(emb_norm, emb_norm.t())             # [N, N]
        # Zero out diagonal (self-loops) — we add them separately if needed
        cos_sim.fill_diagonal_(0.0)

        # ---- 2. Shared-entity adjacency matrix ----
        # shared_entity[i, j] = 1 if pairs i and j share head or tail
        shared_entity = self._compute_shared_entity_matrix(entity_pair_ids, device)  # [N, N]

        # ---- 3. Graph-distance proximity ----
        if doc_graph_distances is not None:
            graph_prox = self._compute_graph_proximity_matrix(
                entity_pair_ids, doc_graph_distances, device
            )  # [N, N], values in [0, 1]
        else:
            graph_prox = torch.zeros(N, N, device=device)

        # ---- 4. Composite edge weight ----
        # Map cos_sim from [-1,1] → [0,1]
        cos_sim_01 = (cos_sim + 1.0) / 2.0  # [N, N]

        composite = (
            self.emb_weight          * cos_sim_01
            + self.shared_entity_weight * shared_entity
            + self.graph_dist_weight    * graph_prox
        )  # [N, N]

        # ---- 5. Threshold and select top-k neighbors ----
        # Only keep edges where embedding similarity exceeds threshold
        sim_mask = cos_sim_01 >= self.similarity_threshold                  # [N, N]
        # Always keep shared-entity edges (they are structurally informative)
        struct_mask = shared_entity > 0                                     # [N, N]
        edge_mask = sim_mask | struct_mask                                  # [N, N]

        # Apply mask
        composite = composite * edge_mask.float()

        # Top-k per row
        composite = self._apply_topk(composite, k=self.max_neighbors)      # [N, N]

        # Remove zero-weight edges
        edge_indices = composite.nonzero(as_tuple=False)                    # [E, 2]
        if edge_indices.numel() == 0:
            return self._build_empty_graph(pair_embs, device)

        src = edge_indices[:, 0]                                            # [E]
        dst = edge_indices[:, 1]                                            # [E]
        weights = composite[src, dst]                                       # [E]

        edge_index  = torch.stack([src, dst], dim=0)                        # [2, E]
        edge_weight = weights                                               # [E]

        return self._pack_graph(
            pair_embs, edge_index, edge_weight, relation_labels, device
        )

    def compute_doc_graph_distances(
        self,
        entity_pair_ids: List[PairID],
        doc_graph: object,
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute shortest-path distances between entity pairs using the document graph.

        The document graph is expected to be a mapping from entity index to its
        direct neighbours (adjacency list), or a NetworkX-style graph object.
        This implementation uses BFS for each entity to find distances, then
        maps those to entity-pair distances as:

            dist(pair_i, pair_j) = min(dist(h_i, h_j), dist(h_i, t_j),
                                       dist(t_i, h_j), dist(t_i, t_j))

        Parameters
        ----------
        entity_pair_ids : List[PairID]
            (h_idx, t_idx) for each entity pair node.
        doc_graph : object
            Either:
            - A dict of {entity_idx: List[entity_idx]} (adjacency list), or
            - A networkx.Graph object (if networkx is available).

        Returns
        -------
        Dict[Tuple[int, int], float]
            Maps (pair_i, pair_j) with i < j → normalised distance in [0, 1].
        """
        import sys

        # Try to interpret doc_graph as adjacency list or networkx graph
        adj: Dict[int, List[int]]
        if isinstance(doc_graph, dict):
            adj = doc_graph
        else:
            # Assume networkx-compatible object
            try:
                adj = {n: list(doc_graph.neighbors(n)) for n in doc_graph.nodes()}
            except AttributeError:
                logger.warning(
                    "doc_graph type %s not recognised. Returning empty distances.",
                    type(doc_graph),
                )
                return {}

        # BFS shortest-path distances between all entities mentioned in pairs
        all_entities = set()
        for h, t in entity_pair_ids:
            all_entities.add(h)
            all_entities.add(t)

        entity_dists: Dict[Tuple[int, int], float] = {}
        for src_ent in all_entities:
            dists = _bfs_distances(src_ent, adj)
            for tgt_ent in all_entities:
                d = dists.get(tgt_ent, float("inf"))
                entity_dists[(src_ent, tgt_ent)] = d

        # Determine max finite distance for normalisation
        finite_dists = [v for v in entity_dists.values() if v != float("inf")]
        max_dist = max(finite_dists) if finite_dists else 1.0

        # Map entity distances to pair distances
        pair_dists: Dict[Tuple[int, int], float] = {}
        for i, (hi, ti) in enumerate(entity_pair_ids):
            for j, (hj, tj) in enumerate(entity_pair_ids):
                if i >= j:
                    continue
                d = min(
                    entity_dists.get((hi, hj), float("inf")),
                    entity_dists.get((hi, tj), float("inf")),
                    entity_dists.get((ti, hj), float("inf")),
                    entity_dists.get((ti, tj), float("inf")),
                )
                # Normalise to [0, 1]; closer → higher proximity
                if d == float("inf"):
                    pair_dists[(i, j)] = 0.0
                else:
                    pair_dists[(i, j)] = 1.0 - d / (max_dist + _EPS)

        return pair_dists

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_shared_entity_matrix(
        self,
        entity_pair_ids: List[PairID],
        device: torch.device,
    ) -> Tensor:
        """
        Build binary matrix where M[i, j] = 1 iff pairs i and j share head or tail.

        Shape: [N, N].
        """
        N = len(entity_pair_ids)
        mat = torch.zeros(N, N, device=device)
        for i, (hi, ti) in enumerate(entity_pair_ids):
            for j, (hj, tj) in enumerate(entity_pair_ids):
                if i != j and (hi == hj or ti == tj or hi == tj or ti == hj):
                    mat[i, j] = 1.0
        return mat

    def _compute_graph_proximity_matrix(
        self,
        entity_pair_ids: List[PairID],
        doc_graph_distances: Dict[Tuple[int, int], float],
        device: torch.device,
    ) -> Tensor:
        """
        Build proximity matrix from pre-computed pair distances.

        Values in [0, 1] where 1 = identical position, 0 = unreachable.

        Shape: [N, N].
        """
        N = len(entity_pair_ids)
        mat = torch.zeros(N, N, device=device)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                prox = doc_graph_distances.get(key, 0.0)
                mat[i, j] = prox
        return mat

    def _apply_topk(self, weight_matrix: Tensor, k: int) -> Tensor:
        """
        For each row, zero out all but the top-k weights.

        Parameters
        ----------
        weight_matrix : Tensor
            [N, N] weight matrix.
        k : int
            Number of neighbors to retain.

        Returns
        -------
        Tensor
            [N, N] sparse weight matrix.
        """
        N = weight_matrix.shape[0]
        if k >= N:
            return weight_matrix
        topk_vals, _ = torch.topk(weight_matrix, k=min(k, N - 1), dim=1)
        threshold = topk_vals[:, -1].unsqueeze(1)  # [N, 1]
        mask = weight_matrix >= threshold           # [N, N]
        return weight_matrix * mask.float()

    def _pack_graph(
        self,
        pair_embs: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        relation_labels: Optional[Tensor],
        device: torch.device,
    ) -> "PyGData | dict":
        """Wrap tensors into a PyG Data object or plain dict."""
        if _HAS_PYG:
            data = PyGData(
                x=pair_embs.to(device),
                edge_index=edge_index.to(device),
                edge_weight=edge_weight.to(device),
            )
            if relation_labels is not None:
                data.y = relation_labels.to(device)
            return data
        else:
            result = {
                "x":           pair_embs.to(device),
                "edge_index":  edge_index.to(device),
                "edge_weight": edge_weight.to(device),
            }
            if relation_labels is not None:
                result["y"] = relation_labels.to(device)
            return result

    def _build_empty_graph(
        self, pair_embs: Tensor, device: torch.device
    ) -> "PyGData | dict":
        """Return an empty graph structure for degenerate inputs."""
        empty_edges  = torch.zeros(2, 0, dtype=torch.long, device=device)
        empty_weight = torch.zeros(0, device=device)
        return self._pack_graph(pair_embs, empty_edges, empty_weight, None, device)


# ------------------------------------------------------------------
# BFS utility
# ------------------------------------------------------------------

def _bfs_distances(
    src: int, adj: Dict[int, List[int]]
) -> Dict[int, float]:
    """
    BFS from ``src`` in adjacency list ``adj``.

    Returns
    -------
    Dict[int, float]
        Mapping node → shortest-path distance from src.
    """
    from collections import deque

    dist: Dict[int, float] = {src: 0.0}
    queue: deque = deque([src])
    while queue:
        node = queue.popleft()
        for neighbour in adj.get(node, []):
            if neighbour not in dist:
                dist[neighbour] = dist[node] + 1.0
                queue.append(neighbour)
    return dist
