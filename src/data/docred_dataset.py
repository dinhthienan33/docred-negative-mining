"""
DocRED Dataset Loader.

Handles loading, tokenization, and batching of the DocRED
document-level relation extraction dataset.

DocRED JSON format reference:
  Each document is a dict with keys:
    - "title": str
    - "sents": List[List[str]]  (sentences as word lists)
    - "vertexSet": List[List[dict]]  (entities; each entity is a list of mentions)
        mention dict: {"name": str, "sent_id": int, "pos": [start, end], "type": str}
    - "labels": List[dict]  (relation triples)
        label dict: {"h": int, "t": int, "r": str, "evidence": List[int]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Special marker tokens defined in the vocab as [unusedX]
ENTITY_START_MARKER = "[unused0]"
ENTITY_END_MARKER = "[unused1]"

# DocRED has 96 typed relations + 1 NA relation (id=0)
NUM_RELATIONS = 97


# ---------------------------------------------------------------------------
# Relation Info
# ---------------------------------------------------------------------------

class DocREDRelationInfo:
    """Loads and stores the 96 DocRED relation types from rel_info.json.

    The NA (no-relation) class is assigned id 0.  All 96 DocRED relations
    are mapped to ids 1..96 (alphabetically by Wikidata property ID so that
    the mapping is deterministic even without rel_info.json).

    Attributes:
        rel2id: Mapping from Wikidata property string (e.g. "P17") to int id.
        id2rel: Inverse mapping.
        rel_info: Optional human-readable label per relation, loaded from
            rel_info.json when available.
    """

    # Canonical ordering of the 96 DocRED relations
    DOCRED_RELATIONS: List[str] = [
        "P1001", "P101", "P102", "P103", "P105", "P106", "P108", "P1080",
        "P110", "P112", "P118", "P123", "P127", "P1303", "P131", "P1344",
        "P135", "P136", "P137", "P138", "P140", "P1412", "P1441", "P150",
        "P155", "P156", "P159", "P161", "P162", "P163", "P166", "P17",
        "P170", "P171", "P172", "P175", "P176", "P178", "P179", "P19",
        "P190", "P193", "P194", "P20", "P206", "P21", "P213", "P22",
        "P223", "P225", "P241", "P25", "P264", "P272", "P276", "P279",
        "P27", "P30", "P31", "P36", "P361", "P364", "P37", "P38",
        "P39", "P40", "P403", "P407", "P408", "P410", "P413", "P414",
        "P449", "P463", "P466", "P495", "P527", "P551", "P569", "P570",
        "P571", "P57", "P576", "P577", "P58", "P580", "P582", "P585",
        "P607", "P61", "P641", "P664", "P69", "P710", "P740", "P84",
    ]

    def __init__(self, rel_info_path: Optional[str] = None) -> None:
        """Initialise relation mappings.

        Args:
            rel_info_path: Path to DocRED ``rel_info.json``.  When provided
                the human-readable labels are loaded; the id mapping is always
                built from :attr:`DOCRED_RELATIONS`.
        """
        # NA is id 0; typed relations start from 1
        self.rel2id: Dict[str, int] = {"NA": 0}
        for idx, rel in enumerate(self.DOCRED_RELATIONS, start=1):
            self.rel2id[rel] = idx

        self.id2rel: Dict[int, str] = {v: k for k, v in self.rel2id.items()}

        self.rel_info: Dict[str, str] = {}
        if rel_info_path is not None:
            self._load_rel_info(rel_info_path)

    def _load_rel_info(self, path: str) -> None:
        """Load human-readable relation descriptions from rel_info.json.

        Args:
            path: Filesystem path to ``rel_info.json``.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw: Dict[str, str] = json.load(fh)
            self.rel_info = raw
            logger.info("Loaded rel_info from %s (%d entries)", path, len(raw))
        except FileNotFoundError:
            logger.warning("rel_info.json not found at %s; labels unavailable.", path)

    def get_id(self, rel_name: str) -> int:
        """Return integer id for a relation string, or 0 (NA) if unknown.

        Args:
            rel_name: Wikidata property string, e.g. ``"P17"``.

        Returns:
            Integer relation id.
        """
        return self.rel2id.get(rel_name, 0)

    def get_name(self, rel_id: int) -> str:
        """Return relation string for an integer id.

        Args:
            rel_id: Integer relation id.

        Returns:
            Wikidata property string, or ``"NA"`` for id 0.
        """
        return self.id2rel.get(rel_id, "NA")

    def __len__(self) -> int:
        return len(self.rel2id)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DocREDDataset(Dataset):
    """PyTorch Dataset for the DocRED document-level relation extraction corpus.

    Each sample corresponds to one document.  The dataset tokenizes the
    concatenated sentences, maps entity mention spans from word-level to
    subword token-level offsets, constructs the multi-hot label tensor, and
    extracts evidence sentence indices per relation triple.

    Args:
        data_path: Path to a DocRED JSON file (train_annotated.json, dev.json,
            or test.json).
        tokenizer_name: HuggingFace model name / path for the tokenizer.
        max_length: Maximum number of subword tokens (documents longer than
            this are truncated).
        relation_map_path: Optional path to ``rel_info.json`` for loading
            human-readable relation labels.
        use_entity_markers: If ``True``, insert ``[unused0]`` / ``[unused1]``
            tokens around every entity mention span.  Helps many PLMs to
            locate entity boundaries.
        add_special_tokens: If ``True`` (default), the tokenizer wraps the
            sequence with [CLS] / [SEP] tokens.

    Example::

        dataset = DocREDDataset("data/train_annotated.json", "roberta-base")
        sample = dataset[0]
        # sample["input_ids"]    – LongTensor [seq_len]
        # sample["labels"]       – FloatTensor [E, E, 97]
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_length: int = 1024,
        relation_map_path: Optional[str] = None,
        use_entity_markers: bool = True,
        add_special_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.max_length = max_length
        self.use_entity_markers = use_entity_markers
        self.add_special_tokens = add_special_tokens

        # Load tokenizer ---------------------------------------------------
        logger.info("Loading tokenizer: %s", tokenizer_name)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )

        # Add entity marker tokens if not already in vocabulary ------------
        if use_entity_markers:
            special_tokens: List[str] = []
            for tok in [ENTITY_START_MARKER, ENTITY_END_MARKER]:
                if tok not in self.tokenizer.get_vocab():
                    special_tokens.append(tok)
            if special_tokens:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": special_tokens}
                )
                logger.info("Added entity marker tokens: %s", special_tokens)

        # Relation info ----------------------------------------------------
        self.relation_info = DocREDRelationInfo(relation_map_path)
        self.num_relations = len(self.relation_info)  # 97

        # Load data --------------------------------------------------------
        logger.info("Loading DocRED data from %s", data_path)
        with open(data_path, "r", encoding="utf-8") as fh:
            self.documents: List[Dict[str, Any]] = json.load(fh)
        logger.info("Loaded %d documents", len(self.documents))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return number of documents in this split."""
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single processed document.

        Args:
            idx: Document index.

        Returns:
            A dict with keys:

            ``input_ids`` – LongTensor [seq_len]
                Token ids, possibly truncated to ``max_length``.
            ``attention_mask`` – LongTensor [seq_len]
                1 for real tokens, 0 for padding (no padding here; padding is
                done in :func:`docred_collate_fn`).
            ``entity_spans`` – List[List[Tuple[int,int]]]
                For each entity, a list of ``(start_tok, end_tok)`` pairs
                (end_tok is *exclusive*) in the tokenized sequence.
            ``mention_to_entity`` – List[int]
                ``mention_to_entity[m]`` is the entity index that mention *m*
                belongs to (flattened mention list).
            ``labels`` – FloatTensor [E, E, num_relations]
                Multi-hot label matrix; ``labels[h, t, r] = 1`` iff triple
                (h, t, r) is annotated.
            ``evidence`` – Dict[Tuple[int,int,int], List[int]]
                Maps ``(h_idx, t_idx, r_id)`` to sorted list of supporting
                sentence indices.
            ``hts`` – List[Tuple[int,int]]
                All ordered entity pairs ``(h, t)`` with ``h != t``.
            ``title`` – str
                Document title.
            ``sentence_boundaries`` – List[Tuple[int,int]]
                ``(start_tok, end_tok)`` for each sentence in token space.
        """
        doc = self.documents[idx]
        return self._tokenize_document(doc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a DocRED document and map all spans to token offsets.

        The document's sentences are first joined into a single string (with
        word-level tokens separated by spaces) and then passed to the
        subword tokenizer.  Entity mention positions are tracked through
        this process using a word-to-token offset map.

        Args:
            doc: Raw DocRED document dict.

        Returns:
            Processed sample dict (see :meth:`__getitem__`).
        """
        sents: List[List[str]] = doc["sents"]
        vertex_set: List[List[Dict[str, Any]]] = doc.get("vertexSet", [])

        # ------------------------------------------------------------------
        # Step 1: Build flat word list with per-word sentence ids and
        #         pre-compute word-level mention positions
        # ------------------------------------------------------------------
        words: List[str] = []
        sent_word_start: List[int] = []  # first word index of each sentence

        for sent in sents:
            sent_word_start.append(len(words))
            words.extend(sent)

        # ------------------------------------------------------------------
        # Step 2: Tokenize with entity markers (optional)
        #         We tokenize word-by-word to maintain offset alignment.
        # ------------------------------------------------------------------
        # token_ids: raw token ids (without CLS/SEP)
        # word_to_tok_start[i]: index of first subword token for word i
        # word_to_tok_end[i]:   index *after* last subword token for word i

        token_ids: List[int] = []
        word_to_tok_start: List[int] = []
        word_to_tok_end: List[int] = []

        # Collect sets of word indices that are entity mention starts/ends
        # so we can insert markers at the right positions.
        mention_start_words: Dict[int, List[Tuple[int, int]]] = {}  # word_idx → [(ent, men)]
        mention_end_words: Dict[int, List[Tuple[int, int]]] = {}

        if self.use_entity_markers:
            for ent_idx, entity in enumerate(vertex_set):
                for men_idx, mention in enumerate(entity):
                    sent_id: int = mention["sent_id"]
                    pos: List[int] = mention["pos"]  # [start, end) in sentence words
                    global_start = sent_word_start[sent_id] + pos[0]
                    global_end = sent_word_start[sent_id] + pos[1]  # exclusive

                    mention_start_words.setdefault(global_start, []).append(
                        (ent_idx, men_idx)
                    )
                    mention_end_words.setdefault(global_end, []).append(
                        (ent_idx, men_idx)
                    )

        start_marker_id = self.tokenizer.convert_tokens_to_ids(ENTITY_START_MARKER)
        end_marker_id = self.tokenizer.convert_tokens_to_ids(ENTITY_END_MARKER)

        # Token-level positions of markers for each mention (ent, men) → tok_idx
        mention_marker_start: Dict[Tuple[int, int], int] = {}
        mention_marker_end: Dict[Tuple[int, int], int] = {}

        for w_idx, word in enumerate(words):
            # Insert start markers for any mentions beginning at this word
            if self.use_entity_markers and w_idx in mention_start_words:
                for key in mention_start_words[w_idx]:
                    mention_marker_start[key] = len(token_ids)
                    token_ids.append(start_marker_id)

            word_tok_start = len(token_ids)
            word_toks = self.tokenizer.encode(word, add_special_tokens=False)
            token_ids.extend(word_toks)
            word_tok_end = len(token_ids)

            word_to_tok_start.append(word_tok_start)
            word_to_tok_end.append(word_tok_end)

            # Insert end markers for any mentions ending after this word
            if self.use_entity_markers and (w_idx + 1) in mention_end_words:
                for key in mention_end_words[w_idx + 1]:
                    mention_marker_end[key] = len(token_ids)
                    token_ids.append(end_marker_id)

        # ------------------------------------------------------------------
        # Step 3: Add CLS / SEP and truncate
        # ------------------------------------------------------------------
        cls_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id or 0
        sep_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id or 2

        if self.add_special_tokens:
            full_ids = [cls_id] + token_ids + [sep_id]
            offset = 1  # shift all token indices by 1 due to [CLS]
        else:
            full_ids = token_ids
            offset = 0

        # Truncate to max_length (preserve at least [CLS] and [SEP])
        if len(full_ids) > self.max_length:
            if self.add_special_tokens:
                full_ids = full_ids[: self.max_length - 1] + [sep_id]
            else:
                full_ids = full_ids[: self.max_length]

        input_ids_tensor = torch.tensor(full_ids, dtype=torch.long)
        attention_mask_tensor = torch.ones_like(input_ids_tensor)

        max_valid_tok = self.max_length - 1 if self.add_special_tokens else self.max_length

        # ------------------------------------------------------------------
        # Step 4: Map entity mention spans to (token_start, token_end)
        # ------------------------------------------------------------------
        entity_spans: List[List[Tuple[int, int]]] = []
        mention_to_entity: List[int] = []
        mention_idx_global = 0

        for ent_idx, entity in enumerate(vertex_set):
            ent_spans: List[Tuple[int, int]] = []
            for men_idx, mention in enumerate(entity):
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                g_start = sent_word_start[sent_id] + pos[0]
                g_end = sent_word_start[sent_id] + pos[1]  # exclusive

                if self.use_entity_markers and (ent_idx, men_idx) in mention_marker_start:
                    tok_start = mention_marker_start[(ent_idx, men_idx)] + offset
                    tok_end = mention_marker_end.get(
                        (ent_idx, men_idx),
                        word_to_tok_end[min(g_end - 1, len(word_to_tok_end) - 1)],
                    ) + offset
                else:
                    # Use the subword span of the mention words
                    tok_start = word_to_tok_start[g_start] + offset if g_start < len(word_to_tok_start) else offset
                    last_word = min(g_end - 1, len(word_to_tok_end) - 1)
                    tok_end = word_to_tok_end[last_word] + offset if last_word >= 0 else offset + 1

                # Clamp to truncated sequence length
                tok_start = min(tok_start, max_valid_tok)
                tok_end = min(tok_end, max_valid_tok)
                if tok_start >= tok_end:
                    tok_end = tok_start + 1

                ent_spans.append((tok_start, tok_end))
                mention_to_entity.append(ent_idx)
                mention_idx_global += 1

            entity_spans.append(ent_spans)

        # ------------------------------------------------------------------
        # Step 5: Sentence boundaries in token space
        # ------------------------------------------------------------------
        sentence_boundaries: List[Tuple[int, int]] = []
        for s_idx, sent in enumerate(sents):
            w_start = sent_word_start[s_idx]
            w_end = w_start + len(sent)
            t_start = word_to_tok_start[w_start] + offset if w_start < len(word_to_tok_start) else offset
            last_w = min(w_end - 1, len(word_to_tok_end) - 1)
            t_end = word_to_tok_end[last_w] + offset if last_w >= 0 else offset
            t_start = min(t_start, max_valid_tok)
            t_end = min(t_end, max_valid_tok)
            sentence_boundaries.append((t_start, t_end))

        # ------------------------------------------------------------------
        # Step 6: Build labels and evidence
        # ------------------------------------------------------------------
        num_entities = len(vertex_set)
        labels_tensor, evidence_dict = self._build_labels(doc, num_entities)

        # ------------------------------------------------------------------
        # Step 7: Entity pairs to classify
        # ------------------------------------------------------------------
        hts = self._generate_entity_pairs(num_entities)

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "entity_spans": entity_spans,
            "mention_to_entity": mention_to_entity,
            "labels": labels_tensor,
            "evidence": evidence_dict,
            "hts": hts,
            "title": doc.get("title", ""),
            "sentence_boundaries": sentence_boundaries,
        }

    def _build_labels(
        self,
        doc: Dict[str, Any],
        num_entities: int,
    ) -> Tuple[torch.Tensor, Dict[Tuple[int, int, int], List[int]]]:
        """Build multi-hot label matrix and evidence dict for a document.

        Args:
            doc: Raw DocRED document dict.
            num_entities: Number of unique entities in this document.

        Returns:
            Tuple of:
                labels – FloatTensor [num_entities, num_entities, num_relations]
                    ``labels[h, t, r] = 1.0`` for each annotated triple.
                evidence – Dict mapping ``(h_idx, t_idx, r_id)`` to sorted
                    list of supporting sentence indices.
        """
        labels = torch.zeros(
            (num_entities, num_entities, self.num_relations), dtype=torch.float32
        )
        evidence: Dict[Tuple[int, int, int], List[int]] = {}

        for label_dict in doc.get("labels", []):
            h: int = label_dict["h"]
            t: int = label_dict["t"]
            r_str: str = label_dict["r"]
            r_id = self.relation_info.get_id(r_str)

            if h >= num_entities or t >= num_entities:
                logger.warning(
                    "Label h=%d or t=%d out of bounds (num_entities=%d); skipping.",
                    h, t, num_entities,
                )
                continue

            labels[h, t, r_id] = 1.0

            evid_sents: List[int] = sorted(set(label_dict.get("evidence", [])))
            evidence[(h, t, r_id)] = evid_sents

        return labels, evidence

    def _generate_entity_pairs(self, num_entities: int) -> List[Tuple[int, int]]:
        """Generate all ordered entity pairs (h, t) where h != t.

        Args:
            num_entities: Total number of entities in the document.

        Returns:
            List of ``(h_idx, t_idx)`` tuples.
        """
        return [
            (h, t)
            for h in range(num_entities)
            for t in range(num_entities)
            if h != t
        ]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def docred_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of DocRED samples into a batch.

    Pads ``input_ids`` and ``attention_mask`` to the maximum sequence length
    in the batch.  All other fields are collected into lists (one element per
    document) since they have variable structure.

    Args:
        batch: List of dicts as returned by :meth:`DocREDDataset.__getitem__`.

    Returns:
        Batched dict with keys:

        ``input_ids`` – LongTensor [B, max_seq_len]
        ``attention_mask`` – LongTensor [B, max_seq_len]
        ``entity_spans`` – List[B] of entity span lists
        ``mention_to_entity`` – List[B] of mention→entity mappings
        ``labels`` – List[B] of FloatTensor [E_i, E_i, num_relations]
        ``evidence`` – List[B] of evidence dicts
        ``hts`` – List[B] of (h,t) pair lists
        ``titles`` – List[B] of document titles
        ``sentence_boundaries`` – List[B] of sentence boundary lists
    """
    max_seq_len = max(sample["input_ids"].size(0) for sample in batch)

    input_ids_list: List[torch.Tensor] = []
    attention_mask_list: List[torch.Tensor] = []

    entity_spans_list: List[Any] = []
    mention_to_entity_list: List[Any] = []
    labels_list: List[torch.Tensor] = []
    evidence_list: List[Any] = []
    hts_list: List[Any] = []
    titles_list: List[str] = []
    sentence_boundaries_list: List[Any] = []

    for sample in batch:
        seq_len = sample["input_ids"].size(0)
        pad_len = max_seq_len - seq_len

        # Pad input_ids with tokenizer pad_id (0 if unavailable)
        padded_ids = torch.cat(
            [
                sample["input_ids"],
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )
        padded_mask = torch.cat(
            [
                sample["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )

        input_ids_list.append(padded_ids)
        attention_mask_list.append(padded_mask)

        entity_spans_list.append(sample["entity_spans"])
        mention_to_entity_list.append(sample["mention_to_entity"])
        labels_list.append(sample["labels"])
        evidence_list.append(sample["evidence"])
        hts_list.append(sample["hts"])
        titles_list.append(sample["title"])
        sentence_boundaries_list.append(sample["sentence_boundaries"])

    return {
        "input_ids": torch.stack(input_ids_list),           # [B, max_seq_len]
        "attention_mask": torch.stack(attention_mask_list), # [B, max_seq_len]
        "entity_spans": entity_spans_list,
        "mention_to_entity": mention_to_entity_list,
        "labels": labels_list,
        "evidence": evidence_list,
        "hts": hts_list,
        "titles": titles_list,
        "sentence_boundaries": sentence_boundaries_list,
    }
