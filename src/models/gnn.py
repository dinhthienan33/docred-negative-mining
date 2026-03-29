"""
gnn.py – Relational GCN Reasoning over Heterogeneous Document Graph
====================================================================
Implements two classes:

1. ``RGCNLayer``
   A single layer of heterogeneous message-passing that uses
   ``torch_geometric.nn.RGCNConv`` (with basis decomposition) for each
   relation type, wrapped in a ``HeteroConv`` container.
   After aggregation: ReLU → Dropout → Residual addition → LayerNorm.

2. ``DocGraphReasoner``
   Stacks ``num_layers`` of ``RGCNLayer`` to refine all node-type
   embeddings.  The final entity-node embeddings are the primary output
   for downstream triple scoring.

Basis decomposition rationale
------------------------------
With many relation types, having one full weight matrix per relation
blows up parameter count.  Basis decomposition expresses each relation's
weight matrix as a linear combination of ``num_bases`` shared basis
matrices, reducing parameters from O(R * d²) to O((R + num_bases) * d²)
while preserving expressiveness.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData  # type: ignore
    from torch_geometric.nn import HeteroConv, RGCNConv  # type: ignore
except ImportError as exc:
    raise ImportError(
        "torch_geometric is required. "
        "Install with: pip install torch-geometric"
    ) from exc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default edge types matching DocGraphBuilder output
# ---------------------------------------------------------------------------

DEFAULT_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("mention",   "in_sentence",  "sentence"),
    ("sentence",  "contains",     "mention"),
    ("mention",   "coref",        "mention"),
    ("entity",    "has_mention",  "mention"),
    ("mention",   "belongs_to",   "entity"),
    ("sentence",  "adjacent",     "sentence"),
    ("mention",   "same_sent",    "mention"),
    ("entity",    "self_loop",    "entity"),
    ("mention",   "self_loop",    "mention"),
]


class _RGCNConvWithEdgeType(nn.Module):
    """Single-relation :class:`RGCNConv` for use inside :class:`HeteroConv`.

    PyTorch Geometric 2.4+ requires ``edge_type`` when ``edge_index`` is a
    dense tensor. :class:`HeteroConv` only passes ``(x, edge_index)``; for
    ``num_relations=1`` we supply relation id 0 for every edge.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_bases: int,
        aggr: str = "add",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.conv = RGCNConv(
            in_channels=in_dim,
            out_channels=out_dim,
            num_relations=1,
            num_bases=num_bases,
            aggr=aggr,
            bias=bias,
        )

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            if isinstance(x, tuple):
                x_r = x[1]
                assert x_r is not None
                return x_r.new_zeros(x_r.size(0), self.out_dim)
            return x.new_zeros(x.size(0), self.out_dim)
        edge_type = torch.zeros(
            edge_index.size(1), dtype=torch.long, device=edge_index.device
        )
        return self.conv(x, edge_index, edge_type)


# ---------------------------------------------------------------------------
# RGCNLayer
# ---------------------------------------------------------------------------

class RGCNLayer(nn.Module):
    """
    One layer of relational message-passing over a heterogeneous graph.

    For each edge type ``(src_type, rel_type, dst_type)`` a separate
    ``RGCNConv`` is created; ``HeteroConv`` orchestrates the forward pass
    and aggregates messages from different source types by summing.

    Post-aggregation per node type::

        h' = LayerNorm( Dropout( ReLU(agg) ) + residual_proj(h) )

    Parameters
    ----------
    in_dim : int
        Input feature dimension (same for all node types for simplicity).
    out_dim : int
        Output feature dimension.
    edge_types : list of (src, rel, dst) tuples
        Which edge types to create convolution operators for.
    num_bases : int
        Number of basis matrices for basis decomposition in RGCNConv.
    dropout : float
        Dropout probability applied after ReLU.
    aggr : str
        Aggregation scheme for ``HeteroConv`` (``"sum"``, ``"mean"``, ``"max"``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        num_bases: int = 4,
        dropout: float = 0.1,
        aggr: str = "sum",
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        _edge_types = edge_types if edge_types is not None else DEFAULT_EDGE_TYPES

        # ----------------------------------------------------------------
        # Build one RGCNConv per edge type
        # ----------------------------------------------------------------
        conv_dict: Dict[Tuple[str, str, str], nn.Module] = {}
        for edge_type in _edge_types:
            conv_dict[edge_type] = _RGCNConvWithEdgeType(
                in_dim=in_dim,
                out_dim=out_dim,
                num_bases=num_bases,
                aggr="add",
                bias=True,
            )
        self.hetero_conv = HeteroConv(conv_dict, aggr=aggr)

        # ----------------------------------------------------------------
        # Residual projection (if in_dim != out_dim, project; else identity)
        # ----------------------------------------------------------------
        # We build one linear per possible node type that appears in edge_types
        node_types: Set[str] = set()
        for src, _, dst in _edge_types:
            node_types.add(src)
            node_types.add(dst)

        self.residual_projs = nn.ModuleDict()
        for nt in sorted(node_types):
            if in_dim != out_dim:
                self.residual_projs[nt] = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.residual_projs[nt] = nn.Identity()

        # ----------------------------------------------------------------
        # LayerNorm per node type
        # ----------------------------------------------------------------
        self.layer_norms = nn.ModuleDict(
            {nt: nn.LayerNorm(out_dim) for nt in sorted(node_types)}
        )

        self.dropout = nn.Dropout(dropout)
        self._node_types = sorted(node_types)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Run one round of heterogeneous message passing.

        Parameters
        ----------
        x_dict : dict[str, Tensor]
            Node features per type, each [num_nodes_t, in_dim].
        edge_index_dict : dict[(src, rel, dst), Tensor]
            Edge indices per edge type, each [2, num_edges].

        Returns
        -------
        dict[str, Tensor]
            Updated node features per type, each [num_nodes_t, out_dim].
        """
        # Filter edge_index_dict to only include edges that have at least
        # one entry and whose node types exist in x_dict
        filtered_edge_index: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for et, ei in edge_index_dict.items():
            src_type, _, dst_type = et
            if (
                src_type in x_dict
                and dst_type in x_dict
                and ei.numel() > 0
            ):
                filtered_edge_index[et] = ei

        # HeteroConv forward
        out_dict: Dict[str, torch.Tensor] = self.hetero_conv(
            x_dict, filtered_edge_index
        )

        # Residual + norm for all node types present in x_dict
        result: Dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt in out_dict and out_dict[nt] is not None:
                h_new = self.dropout(F.relu(out_dict[nt]))
                # Residual
                if nt in self.residual_projs:
                    residual = self.residual_projs[nt](x)
                else:
                    residual = x if self.in_dim == self.out_dim else torch.zeros_like(h_new)
                h_new = h_new + residual
                # LayerNorm
                if nt in self.layer_norms:
                    h_new = self.layer_norms[nt](h_new)
                result[nt] = h_new
            else:
                # Node type received no messages: apply residual projection only
                if nt in self.residual_projs:
                    result[nt] = self.residual_projs[nt](x)
                else:
                    result[nt] = x

        return result


# ---------------------------------------------------------------------------
# DocGraphReasoner
# ---------------------------------------------------------------------------

class DocGraphReasoner(nn.Module):
    """
    Multi-layer R-GCN reasoning module over a heterogeneous document graph.

    Architecture::

        input x_dict
            ↓
        [input_proj per node type]  (in_dim → hidden_dim)
            ↓
        [RGCNLayer × num_layers]
            ↓
        [output_proj for entity nodes]  (hidden_dim → out_dim)
            ↓
        refined entity embeddings

    Parameters
    ----------
    in_dim : int
        Dimensionality of input node embeddings (PLM hidden size, e.g. 1024).
    hidden_dim : int
        GNN hidden dimension (e.g. 256).
    out_dim : int
        Final entity embedding dimension; passed to TripleHead.
    num_layers : int
        Number of stacked ``RGCNLayer`` layers.
    num_heads : int
        Reserved for multi-head attention extensions (stored but not used in
        current RGCNConv-based implementation which already handles this via
        basis decomposition; future Graph Transformer variant can use this).
    num_bases : int
        Basis matrices per ``RGCNConv`` layer.
    dropout : float
        Dropout probability within each layer.
    edge_types : optional list of (src, rel, dst)
        Override the default edge type set.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        num_bases: int = 4,
        dropout: float = 0.1,
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads  # stored for potential future use

        _edge_types = edge_types if edge_types is not None else DEFAULT_EDGE_TYPES

        # Collect all node types
        _node_types: Set[str] = set()
        for src, _, dst in _edge_types:
            _node_types.add(src)
            _node_types.add(dst)
        self._node_types = sorted(_node_types)

        # ----------------------------------------------------------------
        # Input projection: in_dim → hidden_dim per node type
        # ----------------------------------------------------------------
        self.input_projs = nn.ModuleDict(
            {nt: nn.Linear(in_dim, hidden_dim, bias=True) for nt in self._node_types}
        )
        self.input_norms = nn.ModuleDict(
            {nt: nn.LayerNorm(hidden_dim) for nt in self._node_types}
        )

        # ----------------------------------------------------------------
        # Stacked RGCNLayers
        # ----------------------------------------------------------------
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = RGCNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                edge_types=_edge_types,
                num_bases=num_bases,
                dropout=dropout,
                aggr="sum",
            )
            self.layers.append(layer)

        # ----------------------------------------------------------------
        # Output projection for entity nodes: hidden_dim → out_dim
        # ----------------------------------------------------------------
        self.entity_output_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=True),
            nn.LayerNorm(out_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, hetero_data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Run multi-layer GNN reasoning.

        Parameters
        ----------
        hetero_data : HeteroData
            PyG HeteroData object with node features ``x`` and edges.
            Node features must have shape [num_nodes_t, in_dim].

        Returns
        -------
        dict[str, Tensor]
            Refined embeddings per node type:
            - ``"entity"``  : [num_entities, out_dim]  ← primary output
            - ``"mention"`` : [num_mentions, hidden_dim]
            - ``"sentence"``: [num_sentences, hidden_dim]  (if present)
        """
        # ----------------------------------------------------------------
        # Extract node features and edge indices from HeteroData
        # ----------------------------------------------------------------
        x_dict: Dict[str, torch.Tensor] = {}
        for nt in self._node_types:
            if hasattr(hetero_data[nt], "x") and hetero_data[nt].x is not None:
                x = hetero_data[nt].x  # [num_nodes_t, in_dim]
                # Input projection + norm
                if nt in self.input_projs:
                    x = self.input_norms[nt](self.input_projs[nt](x))
                    x = F.relu(x)
                x_dict[nt] = x

        if not x_dict:
            return {}

        # ----------------------------------------------------------------
        # Collect edge indices
        # ----------------------------------------------------------------
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for store in hetero_data.edge_stores:
            et = store._key  # (src, rel, dst) tuple
            if et is not None and hasattr(store, "edge_index"):
                edge_index_dict[et] = store.edge_index

        # ----------------------------------------------------------------
        # Forward through stacked layers
        # ----------------------------------------------------------------
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)

        # ----------------------------------------------------------------
        # Output projection for entity nodes
        # ----------------------------------------------------------------
        result: Dict[str, torch.Tensor] = dict(x_dict)  # shallow copy
        if "entity" in x_dict:
            result["entity"] = self.entity_output_proj(x_dict["entity"])

        return result

    def get_entity_embeddings(self, hetero_data: HeteroData) -> torch.Tensor:
        """
        Convenience method: run forward and return only entity embeddings.

        Parameters
        ----------
        hetero_data : HeteroData

        Returns
        -------
        Tensor [num_entities, out_dim]
        """
        out = self.forward(hetero_data)
        return out.get("entity", torch.zeros(0, self.out_dim))
