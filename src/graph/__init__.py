"""
Graph modules for the DocRED SOTA pipeline.

Exports
-------
TripleGraphBuilder
    Constructs a triple-level graph (one node per entity pair) used in the
    graph contrastive learning stage.  Edges are formed by:
    - Embedding cosine similarity above a configurable threshold
    - Shared head or tail entity between pairs
    - Graph-structural proximity in the document graph
    The output is a PyG ``Data`` object (or a plain dict fallback if
    ``torch_geometric`` is not installed).
"""

from .triple_graph import TripleGraphBuilder

__all__ = [
    "TripleGraphBuilder",
]
