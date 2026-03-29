"""
Model modules for DocRED SOTA pipeline.

Modules
-------
encoder        : DocumentEncoder  – PLM backbone with mention/entity pooling
graph_builder  : DocGraphBuilder  – heterogeneous document graph (PyG HeteroData)
gnn            : RGCNLayer, DocGraphReasoner – relational GCN message-passing
triple_head    : TripleHead, AdaptiveThreshold – pair representation + classifier
pipeline       : DocREDPipeline   – end-to-end forward pass
"""

from src.models.encoder import DocumentEncoder
from src.models.graph_builder import DocGraphBuilder
from src.models.gnn import RGCNLayer, DocGraphReasoner
from src.models.triple_head import TripleHead, AdaptiveThreshold
from src.models.pipeline import DocREDPipeline

__all__ = [
    "DocumentEncoder",
    "DocGraphBuilder",
    "RGCNLayer",
    "DocGraphReasoner",
    "TripleHead",
    "AdaptiveThreshold",
    "DocREDPipeline",
]
