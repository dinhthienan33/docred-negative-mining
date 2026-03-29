"""
DocRED SOTA Pipeline: LLM + GNN + Debiased GCL
===============================================
Document-level relation extraction pipeline combining:
  - Pretrained Language Model encoder (DeBERTa / RoBERTa) with optional LoRA
  - Heterogeneous document graph (mention / entity / sentence nodes)
  - Relational GCN reasoning over the graph
  - Triple representation head with ATLOP-style adaptive thresholding
  - BMM-reweighted graph contrastive learning losses
"""

# Top-level imports are guarded to avoid hard-failing when optional
# dependencies (e.g. torch_geometric) are not installed.  Import the
# submodules directly when only specific components are needed.
try:
    from src.models.encoder import DocumentEncoder
    from src.models.graph_builder import DocGraphBuilder
    from src.models.gnn import DocGraphReasoner, RGCNLayer
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
except ImportError:
    # torch_geometric or other optional deps not installed.
    # Submodule imports (e.g. src.data.docred_dataset) still work fine.
    __all__ = []
