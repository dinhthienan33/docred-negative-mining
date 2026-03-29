"""
encoder.py – Document Encoder for DocRED
=========================================
Wraps a HuggingFace pretrained language model (DeBERTa-v3-large, RoBERTa-large, etc.)
and produces three levels of contextual representations:

  1. Token embeddings    : [batch, seq_len, hidden_dim]
  2. Mention embeddings  : [num_mentions, hidden_dim]   (per document)
  3. Entity embeddings   : [num_entities, hidden_dim]   (per document)

Mention pooling uses attention-weighted averaging over the span tokens.
Entity pooling uses logsumexp (log-sum-exp) aggregation across all mentions that
belong to the same entity, following the ATLOP / DocuNet conventions.

Optional LoRA / QLoRA wrapping via the `peft` library.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: safe LoRA import
# ---------------------------------------------------------------------------

def _try_load_peft():
    """Return (get_peft_model, LoraConfig) or raise ImportError with a helpful message."""
    try:
        from peft import get_peft_model, LoraConfig, TaskType  # type: ignore
        return get_peft_model, LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError(
            "peft is required when use_lora=True. Install with: pip install peft"
        ) from exc


# ---------------------------------------------------------------------------
# DocumentEncoder
# ---------------------------------------------------------------------------

class DocumentEncoder(nn.Module):
    """
    Contextual document encoder backed by a pretrained language model.

    Parameters
    ----------
    plm_name : str
        HuggingFace model name or local path, e.g. ``"microsoft/deberta-v3-large"``.
    use_lora : bool
        Whether to wrap the PLM with LoRA adapters (requires ``peft``).
    lora_rank : int
        LoRA rank ``r``; only relevant when ``use_lora=True``.
    lora_alpha : int
        LoRA scaling factor ``alpha``; defaults to ``2 * lora_rank``.
    lora_dropout : float
        Dropout applied inside LoRA layers.
    lora_target_modules : Optional[List[str]]
        Which projection modules to apply LoRA to.  When ``None``, uses
        ``["query_proj", "key_proj", "value_proj"]`` (DeBERTa-v3 style).
    """

    def __init__(
        self,
        plm_name: str,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Load pretrained model
        # ------------------------------------------------------------------
        logger.info("Loading PLM: %s", plm_name)
        config = AutoConfig.from_pretrained(plm_name)
        self.plm = AutoModel.from_pretrained(plm_name, config=config)
        self.hidden_dim: int = config.hidden_size

        # ------------------------------------------------------------------
        # 2. Optional LoRA wrapping
        # ------------------------------------------------------------------
        if use_lora:
            get_peft_model, LoraConfig, TaskType = _try_load_peft()
            _alpha = lora_alpha if lora_alpha is not None else 2 * lora_rank
            _target = lora_target_modules or [
                "query_proj", "key_proj", "value_proj",  # DeBERTa-v3
                "query", "key", "value",                  # RoBERTa / BERT
            ]
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=_alpha,
                lora_dropout=lora_dropout,
                target_modules=_target,
                bias="none",
            )
            self.plm = get_peft_model(self.plm, lora_cfg)
            logger.info(
                "Applied LoRA (rank=%d, alpha=%d) to: %s",
                lora_rank, _alpha, _target,
            )
            self.plm.print_trainable_parameters()

        # ------------------------------------------------------------------
        # 3. Attention projection for mention-span pooling
        #    A single linear layer maps each token embedding to a scalar
        #    attention weight over the span.
        # ------------------------------------------------------------------
        self.span_attn = nn.Linear(self.hidden_dim, 1, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_spans: List[List[List[Tuple[int, int]]]],
        mention_to_entity: Optional[List[List[int]]] = None,
    ) -> Dict[str, object]:
        """
        Encode a batch of documents.

        Parameters
        ----------
        input_ids : Tensor, shape [batch, seq_len]
            Token ids from the PLM tokenizer.
        attention_mask : Tensor, shape [batch, seq_len]
            1 for real tokens, 0 for padding.
        entity_spans : list[list[list[tuple[int,int]]]]
            Outer list → batch; inner list → entities; innermost list → mention
            spans (token-level, half-open [start, end)) for that entity.

            Example for a document with 2 entities::

                [
                  [  # entity 0
                    [(3, 5), (12, 13)],   # two mentions
                  ],
                  [  # entity 1
                    [(7, 8)],
                  ],
                ]

        mention_to_entity : optional list[list[int]]
            Pre-computed flat mention→entity mapping per document.
            When ``None`` it is derived from ``entity_spans``.

        Returns
        -------
        dict with keys
            ``"token_embeddings"`` : Tensor [batch, seq_len, hidden_dim]
            ``"mention_embeddings"`` : list of Tensor [num_mentions_i, hidden_dim]
            ``"entity_embeddings"``  : list of Tensor [num_entities_i, hidden_dim]
        """
        batch_size, seq_len = input_ids.shape

        # ---- PLM forward ------------------------------------------------
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        # Shape: [batch, seq_len, hidden_dim]
        token_embs: torch.Tensor = outputs.last_hidden_state

        # ---- Per-document mention / entity pooling ----------------------
        all_mention_embs: List[torch.Tensor] = []
        all_entity_embs: List[torch.Tensor] = []

        for doc_idx in range(batch_size):
            doc_token_emb = token_embs[doc_idx]  # [seq_len, hidden_dim]
            doc_entity_spans = entity_spans[doc_idx]  # list of entities, each is list of spans

            # Flatten spans into a list of (start, end) and track entity membership
            flat_spans: List[Tuple[int, int]] = []
            flat_mention_to_entity: List[int] = []

            for ent_idx, ent_spans in enumerate(doc_entity_spans):
                for span in ent_spans:
                    flat_spans.append(span)
                    flat_mention_to_entity.append(ent_idx)

            if len(flat_spans) == 0:
                # Edge case: no entities / mentions in document
                empty = doc_token_emb.new_zeros(0, self.hidden_dim)
                all_mention_embs.append(empty)
                all_entity_embs.append(empty)
                continue

            # Pool each mention span → [num_mentions, hidden_dim]
            mention_embs = self._pool_mentions(doc_token_emb, flat_spans)
            all_mention_embs.append(mention_embs)

            # Aggregate mentions → entities via logsumexp
            num_entities = len(doc_entity_spans)
            entity_embs = self._aggregate_entity(
                mention_embs, flat_mention_to_entity, num_entities
            )
            all_entity_embs.append(entity_embs)

        return {
            "token_embeddings": token_embs,               # [batch, seq_len, hidden_dim]
            "mention_embeddings": all_mention_embs,        # list of [num_mentions, hidden_dim]
            "entity_embeddings": all_entity_embs,          # list of [num_entities, hidden_dim]
        }

    # ------------------------------------------------------------------
    # Mention pooling
    # ------------------------------------------------------------------

    def _pool_mentions(
        self,
        token_embs: torch.Tensor,
        spans: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Attention-weighted pooling of token embeddings over each span.

        For span [s, e) the attention weights are::

            α_i = softmax( W_attn · h_i )   for i ∈ [s, e)

        The mention embedding is then Σ_i α_i * h_i.

        Parameters
        ----------
        token_embs : Tensor [seq_len, hidden_dim]
            Token-level contextual representations for a single document.
        spans : list of (start, end) tuples (half-open, token indices)

        Returns
        -------
        Tensor [num_mentions, hidden_dim]
        """
        device = token_embs.device
        num_mentions = len(spans)
        mention_embs = token_embs.new_zeros(num_mentions, self.hidden_dim)

        for m_idx, (start, end) in enumerate(spans):
            # Clamp to valid range
            start = max(0, start)
            end = min(token_embs.size(0), end)
            if start >= end:
                # Degenerate span: use zero vector
                continue

            span_tokens = token_embs[start:end]  # [span_len, hidden_dim]
            # Attention scores: [span_len, 1] → [span_len]
            attn_scores = self.span_attn(span_tokens).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=0)  # [span_len]
            # Weighted sum: [hidden_dim]
            mention_embs[m_idx] = (attn_weights.unsqueeze(-1) * span_tokens).sum(0)

        return mention_embs

    # ------------------------------------------------------------------
    # Entity aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_entity(
        mention_embs: torch.Tensor,
        mention_to_entity: List[int],
        num_entities: int,
    ) -> torch.Tensor:
        """
        Aggregate mention embeddings into entity-level representations using
        logsumexp pooling, following ATLOP / DocuNet.

        For entity *e* with mentions M_e::

            entity_emb_e = log Σ_{m ∈ M_e} exp(mention_emb_m)

        This is numerically stabilised via ``torch.logsumexp``.

        Parameters
        ----------
        mention_embs : Tensor [num_mentions, hidden_dim]
        mention_to_entity : list[int]  length num_mentions
            Maps each mention index to its entity index.
        num_entities : int

        Returns
        -------
        Tensor [num_entities, hidden_dim]
        """
        hidden_dim = mention_embs.size(1)
        device = mention_embs.device

        entity_embs = mention_embs.new_zeros(num_entities, hidden_dim)

        # Group mentions by entity using a scatter approach
        for ent_idx in range(num_entities):
            indices = [
                m_idx
                for m_idx, e_idx in enumerate(mention_to_entity)
                if e_idx == ent_idx
            ]
            if not indices:
                continue
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            ent_mention_embs = mention_embs[idx_tensor]  # [k, hidden_dim]
            entity_embs[ent_idx] = torch.logsumexp(ent_mention_embs, dim=0)

        return entity_embs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_hidden_dim(self) -> int:
        """Return the PLM hidden dimension."""
        return self.hidden_dim

    def freeze_plm(self) -> None:
        """Freeze all PLM parameters (useful for two-stage training)."""
        for param in self.plm.parameters():
            param.requires_grad_(False)

    def unfreeze_plm(self) -> None:
        """Unfreeze all PLM parameters."""
        for param in self.plm.parameters():
            param.requires_grad_(True)
