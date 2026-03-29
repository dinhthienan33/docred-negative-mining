"""
triple_head.py – Triple Representation + Relation Classification Head
=======================================================================
Implements:

1. ``AdaptiveThreshold``
   A learnable per-relation threshold layer (ATLOP-style).  A single
   linear layer maps entity-pair features to one threshold per relation;
   during inference the threshold is subtracted from the logits so that
   positive relations have logit > 0.

2. ``TripleHead``
   Full triple representation and relation-classification module:
   - Builds pair embedding from [h; t; h⊙t; context] concatenation.
   - Projects through two-layer MLP → ``pair_emb`` (triple_dim).
   - Computes bilinear relation logits over all ``num_relations``.
   - Provides ``get_triple_emb`` for combining pair + relation embeddings.
   - Provides ``get_contrastive_emb`` for GCL projection to unit sphere.

References
----------
- ATLOP: Zhou et al. 2021 "Document-Level Relation Extraction with Adaptive
  Thresholding and Localized Context Pooling"
  https://arxiv.org/abs/2010.11304
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_hidden_layers: int = 1,
    dropout: float = 0.1,
    activation: nn.Module = None,
) -> nn.Sequential:
    """Build a 2–3-layer MLP with GELU activations and dropout."""
    act = activation if activation is not None else nn.GELU()
    layers: List[nn.Module] = []
    current = in_dim
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(current, hidden_dim), act, nn.Dropout(dropout)]
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# AdaptiveThreshold (ATLOP-style)
# ---------------------------------------------------------------------------

class AdaptiveThreshold(nn.Module):
    """
    Learnable adaptive thresholding for multi-label relation classification.

    ATLOP introduces one extra "threshold" class whose score separates
    positive from negative relations at inference.  This module implements
    the learnable variant: a linear layer that maps the pair representation
    to one threshold score per relation.

    Usage during training
    ----------------------
    Concatenate the threshold score as an extra column in the logit matrix:
    ``logits_with_threshold = [logits | threshold_score]``
    and use a cross-entropy loss where the label for the threshold column
    is 1 for NA pairs and 0 for positive pairs.

    Usage during inference
    ----------------------
    A relation r is predicted positive if ``logits[:, r] > threshold[:, r]``.

    Parameters
    ----------
    entity_dim : int
        Dimensionality of the pair (context) representation fed in.
    num_relations : int
        Number of relation types (excludes NA / threshold column).
    """

    def __init__(self, entity_dim: int, num_relations: int) -> None:
        super().__init__()
        self.linear = nn.Linear(entity_dim, num_relations, bias=True)

    def forward(self, pair_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute per-relation thresholds.

        Parameters
        ----------
        pair_emb : Tensor [batch_pairs, entity_dim]

        Returns
        -------
        Tensor [batch_pairs, num_relations]
            One threshold value per (pair, relation).
        """
        return self.linear(pair_emb)

    def predict(
        self,
        logits: torch.Tensor,
        pair_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary predictions via adaptive thresholding.

        Parameters
        ----------
        logits : Tensor [batch_pairs, num_relations]
        pair_emb : Tensor [batch_pairs, entity_dim]

        Returns
        -------
        Tensor [batch_pairs, num_relations]  – BoolTensor
        """
        thresholds = self.forward(pair_emb)  # [batch_pairs, num_relations]
        return logits > thresholds


# ---------------------------------------------------------------------------
# TripleHead
# ---------------------------------------------------------------------------

class TripleHead(nn.Module):
    """
    Triple representation and relation classification head.

    Given a head entity embedding ``h``, tail entity embedding ``t``, and
    an optional context embedding ``ctx``, this module:

    1. Constructs a **pair embedding** via concatenation and MLP projection::

           pair_input = [h; t; h⊙t; ctx]      (if ctx given)
                      = [h; t; h⊙t; zeros]    (if ctx is None)
           pair_emb   = pair_projector(pair_input)   # [batch, triple_dim]

    2. Scores each relation with a **bilinear scorer**::

           logits_r = h_proj · W_r · t_proj + bias_r   ∀ r

       where ``W_r`` is factorised as ``U_r V_r^T`` (low-rank bilinear).

    3. Provides ``get_triple_emb`` to combine ``pair_emb`` with a specific
       relation embedding (for contrastive learning).

    4. Provides ``get_contrastive_emb`` to project to the unit sphere.

    5. Hosts ``AdaptiveThreshold`` for ATLOP-style inference.

    Parameters
    ----------
    entity_dim : int
        Dimensionality of entity embeddings from the GNN (``out_dim``).
    rel_dim : int
        Dimensionality of the learnable relation embedding table.
    num_relations : int
        Number of relation types (DocRED: 96 named + 1 NA = 97).
    triple_dim : int
        Dimensionality of the pair / triple representation space.
    contrastive_dim : int
        Dimensionality of the contrastive projection space.
    dropout : float
        Dropout in the MLP layers.
    """

    def __init__(
        self,
        entity_dim: int,
        rel_dim: int = 128,
        num_relations: int = 97,
        triple_dim: int = 512,
        contrastive_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.entity_dim = entity_dim
        self.rel_dim = rel_dim
        self.num_relations = num_relations
        self.triple_dim = triple_dim
        self.contrastive_dim = contrastive_dim

        # ----------------------------------------------------------------
        # Relation embedding table
        # ----------------------------------------------------------------
        self.relation_embeddings = nn.Embedding(num_relations, rel_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # ----------------------------------------------------------------
        # Pair projector: [h; t; h⊙t; ctx] → triple_dim
        # Input size = 4 * entity_dim  (context is projected to entity_dim)
        # ----------------------------------------------------------------
        # Context projection (handles the case where context_emb comes from
        # a different feature space, e.g. sentence embeddings)
        self.context_proj = nn.Linear(entity_dim, entity_dim, bias=False)

        pair_input_dim = 4 * entity_dim  # [h; t; h*t; ctx]
        self.pair_projector = _make_mlp(
            in_dim=pair_input_dim,
            hidden_dim=triple_dim,
            out_dim=triple_dim,
            num_hidden_layers=1,
            dropout=dropout,
        )

        # ----------------------------------------------------------------
        # Triple projector: [pair_emb; rel_emb] → triple_dim
        # ----------------------------------------------------------------
        self.triple_projector = _make_mlp(
            in_dim=triple_dim + rel_dim,
            hidden_dim=triple_dim,
            out_dim=triple_dim,
            num_hidden_layers=1,
            dropout=dropout,
        )

        # ----------------------------------------------------------------
        # Bilinear classifier
        # We use a factored bilinear: h_proj and t_proj are linear
        # projections; then a bilinear weight B ∈ R^{triple_dim × triple_dim × num_relations}
        # would be too large.  Instead we use a linear layer over the pair
        # embedding which is already a compressed representation.
        # ----------------------------------------------------------------
        self.h_proj = nn.Linear(entity_dim, triple_dim // 2, bias=True)
        self.t_proj = nn.Linear(entity_dim, triple_dim // 2, bias=True)
        # Bilinear: (triple_dim/2) × (triple_dim/2) → num_relations
        # We factor as W = U * V^T where U, V ∈ R^{num_relations × triple_dim/2}
        bilinear_dim = triple_dim // 2
        self.bilinear_u = nn.Linear(bilinear_dim, num_relations, bias=False)
        self.bilinear_v = nn.Linear(bilinear_dim, num_relations, bias=False)
        self.classifier_bias = nn.Parameter(torch.zeros(num_relations))

        # Also provide a direct pair_emb → logits linear head as a
        # complementary scorer (added to bilinear logits)
        self.linear_classifier = nn.Linear(triple_dim, num_relations, bias=False)

        # ----------------------------------------------------------------
        # Adaptive threshold (ATLOP)
        # ----------------------------------------------------------------
        self.adaptive_threshold = AdaptiveThreshold(triple_dim, num_relations)

        # ----------------------------------------------------------------
        # Contrastive projector: triple_dim → contrastive_dim → L2-normalise
        # ----------------------------------------------------------------
        self.contrastive_projector = _make_mlp(
            in_dim=triple_dim,
            hidden_dim=triple_dim,
            out_dim=contrastive_dim,
            num_hidden_layers=1,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier/kaiming initialisation for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h_emb: torch.Tensor,
        t_emb: torch.Tensor,
        context_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute relation logits for a batch of entity pairs.

        Parameters
        ----------
        h_emb : Tensor [batch_pairs, entity_dim]
            Head entity embeddings.
        t_emb : Tensor [batch_pairs, entity_dim]
            Tail entity embeddings.
        context_emb : optional Tensor [batch_pairs, entity_dim]
            Localised context embedding (e.g. average of evidence sentence
            embeddings).  If ``None``, replaced by zeros.

        Returns
        -------
        dict with keys
            ``"logits"``   : Tensor [batch_pairs, num_relations]
            ``"pair_emb"`` : Tensor [batch_pairs, triple_dim]
        """
        batch_pairs = h_emb.size(0)

        # ---- Context fallback ------------------------------------------
        if context_emb is None:
            ctx = h_emb.new_zeros(batch_pairs, self.entity_dim)
        else:
            ctx = self.context_proj(context_emb)  # [batch_pairs, entity_dim]

        # ---- Pair representation ---------------------------------------
        # [h; t; h⊙t; ctx] → [batch_pairs, 4*entity_dim]
        pair_input = torch.cat([h_emb, t_emb, h_emb * t_emb, ctx], dim=-1)
        pair_emb = self.pair_projector(pair_input)  # [batch_pairs, triple_dim]

        # ---- Bilinear logits -------------------------------------------
        h_proj = self.h_proj(h_emb)   # [batch_pairs, triple_dim/2]
        t_proj = self.t_proj(t_emb)   # [batch_pairs, triple_dim/2]

        # u_scores[i, r] = h_proj[i] · U_r
        # v_scores[i, r] = t_proj[i] · V_r
        u_scores = self.bilinear_u(h_proj)   # [batch_pairs, num_relations]
        v_scores = self.bilinear_v(t_proj)   # [batch_pairs, num_relations]
        bilinear_logits = u_scores * v_scores + self.classifier_bias.unsqueeze(0)

        # ---- Linear logits from pair_emb --------------------------------
        linear_logits = self.linear_classifier(pair_emb)  # [batch_pairs, num_relations]

        # ---- Combined logits -------------------------------------------
        logits = bilinear_logits + linear_logits  # [batch_pairs, num_relations]

        return {
            "logits": logits,      # [batch_pairs, num_relations]
            "pair_emb": pair_emb,  # [batch_pairs, triple_dim]
        }

    # ------------------------------------------------------------------
    # Triple embedding
    # ------------------------------------------------------------------

    def get_triple_emb(
        self,
        pair_emb: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine pair embedding with a specific relation embedding.

        Used to build triple-level representations z(h, t, r) for
        contrastive learning.

        Parameters
        ----------
        pair_emb : Tensor [batch_pairs, triple_dim]
        relation_ids : LongTensor [batch_pairs]
            Index into the relation embedding table.

        Returns
        -------
        Tensor [batch_pairs, triple_dim]
        """
        rel_emb = self.relation_embeddings(relation_ids)  # [batch_pairs, rel_dim]
        triple_input = torch.cat([pair_emb, rel_emb], dim=-1)  # [batch, triple_dim + rel_dim]
        return self.triple_projector(triple_input)  # [batch_pairs, triple_dim]

    # ------------------------------------------------------------------
    # Contrastive projection
    # ------------------------------------------------------------------

    def get_contrastive_emb(self, triple_emb: torch.Tensor) -> torch.Tensor:
        """
        Project triple embeddings to the contrastive unit sphere.

        Parameters
        ----------
        triple_emb : Tensor [batch_pairs, triple_dim]

        Returns
        -------
        Tensor [batch_pairs, contrastive_dim]
            L2-normalised (unit norm).
        """
        proj = self.contrastive_projector(triple_emb)  # [batch, contrastive_dim]
        return F.normalize(proj, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Inference with adaptive thresholding
    # ------------------------------------------------------------------

    def predict(
        self,
        h_emb: torch.Tensor,
        t_emb: torch.Tensor,
        context_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full inference: logits + binary predictions via adaptive thresholding.

        Parameters
        ----------
        h_emb : Tensor [batch_pairs, entity_dim]
        t_emb : Tensor [batch_pairs, entity_dim]
        context_emb : optional Tensor [batch_pairs, entity_dim]

        Returns
        -------
        dict with keys
            ``"logits"``      : Tensor [batch_pairs, num_relations]
            ``"pair_emb"``    : Tensor [batch_pairs, triple_dim]
            ``"predictions"`` : BoolTensor [batch_pairs, num_relations]
            ``"thresholds"``  : Tensor [batch_pairs, num_relations]
        """
        out = self.forward(h_emb, t_emb, context_emb)
        logits: torch.Tensor = out["logits"]
        pair_emb: torch.Tensor = out["pair_emb"]

        thresholds = self.adaptive_threshold(pair_emb)  # [batch, num_relations]
        predictions = logits > thresholds                # bool [batch, num_relations]

        return {
            "logits": logits,
            "pair_emb": pair_emb,
            "predictions": predictions,
            "thresholds": thresholds,
        }
