"""DocRED data loading utilities."""

from src.data.docred_dataset import (
    DocREDDataset,
    DocREDRelationInfo,
    docred_collate_fn,
    NUM_RELATIONS,
    ENTITY_START_MARKER,
    ENTITY_END_MARKER,
)

__all__ = [
    "DocREDDataset",
    "DocREDRelationInfo",
    "docred_collate_fn",
    "NUM_RELATIONS",
    "ENTITY_START_MARKER",
    "ENTITY_END_MARKER",
]
