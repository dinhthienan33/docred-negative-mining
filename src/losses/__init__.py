"""
Loss modules for the DocRED SOTA pipeline.

Exports
-------
BetaMixtureModel
    Two-component Beta Mixture Model fitted by EM for estimating per-negative
    hardness probabilities (ProGCL-style).

HardNegativeWeighter
    Wrapper around BetaMixtureModel that computes per-negative importance
    weights during contrastive training, with warm-up support.

EvidenceNegativeMiner
    DocRED-specific hard negative miner that uses evidence sentence overlap
    and relation co-occurrence to tier negatives (hard / medium / easy).

ATLOPLoss
    Adaptive Thresholding Loss for multi-label document-level relation
    classification (from the ATLOP paper).

BMM_InfoNCE
    BMM-reweighted InfoNCE contrastive loss that down-weights suspected
    false negatives.

JointLoss
    Full joint loss: L_total = L_CE + λ_gcl * L_gcl + λ_evidence * L_ev_cl.
    Internally manages ATLOPLoss, BMM_InfoNCE, HardNegativeWeighter, and
    EvidenceNegativeMiner.
"""

from .bmm import BetaMixtureModel, HardNegativeWeighter
from .evidence_negatives import EvidenceNegativeMiner
from .joint_loss import ATLOPLoss, BMM_InfoNCE, JointLoss

__all__ = [
    "BetaMixtureModel",
    "HardNegativeWeighter",
    "EvidenceNegativeMiner",
    "ATLOPLoss",
    "BMM_InfoNCE",
    "JointLoss",
]
