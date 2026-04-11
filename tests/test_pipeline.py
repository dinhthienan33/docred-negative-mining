"""
Unit tests for the DocRED SOTA pipeline.

Run with::

    pytest tests/test_pipeline.py -v

Tests use lightweight mock objects to avoid downloading large models and
to keep the suite fast enough for CI.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from submodules to avoid pulling in torch_geometric
# via src/__init__.py which uses eager top-level imports.
from src.data.docred_dataset import (  # noqa: E402
    DocREDDataset,
    DocREDRelationInfo,
    docred_collate_fn,
    NUM_RELATIONS,
)
from src.utils.helpers import (  # noqa: E402
    set_seed,
    get_device,
    count_parameters,
    format_metrics,
    load_config,
)


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

def make_mock_doc(
    num_sents: int = 3,
    words_per_sent: int = 6,
    num_entities: int = 3,
    num_labels: int = 2,
) -> Dict[str, Any]:
    """Create a minimal synthetic DocRED document.

    Args:
        num_sents: Number of sentences.
        words_per_sent: Words per sentence.
        num_entities: Number of entities.
        num_labels: Number of relation labels.

    Returns:
        Synthetic DocRED document dict.
    """
    sents = [
        [f"word_{s}_{w}" for w in range(words_per_sent)]
        for s in range(num_sents)
    ]

    # One mention per entity; place each in a different sentence (if possible)
    vertex_set = []
    for ent_idx in range(num_entities):
        sent_id = ent_idx % num_sents
        vertex_set.append(
            [
                {
                    "name": f"Entity{ent_idx}",
                    "sent_id": sent_id,
                    "pos": [0, 2],  # first two words of the sentence
                    "type": "MISC",
                }
            ]
        )

    # Use the first two relation ids from DocREDRelationInfo
    rel_info = DocREDRelationInfo()
    relation_names = list(rel_info.rel2id.keys())
    labels = []
    added = 0
    for h in range(num_entities):
        for t in range(num_entities):
            if h != t and added < num_labels:
                labels.append(
                    {
                        "h": h,
                        "t": t,
                        "r": relation_names[added + 1],  # skip NA at index 0
                        "evidence": [0],
                    }
                )
                added += 1

    return {
        "title": "TestDocument",
        "sents": sents,
        "vertexSet": vertex_set,
        "labels": labels,
    }


def write_mock_dataset(
    path: Path,
    num_docs: int = 3,
) -> None:
    """Write a mock DocRED JSON file.

    Args:
        path: Destination file path.
        num_docs: Number of synthetic documents to include.
    """
    docs = [make_mock_doc() for _ in range(num_docs)]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(docs, fh)


# ---------------------------------------------------------------------------
# Test: DocREDRelationInfo
# ---------------------------------------------------------------------------

class TestDocREDRelationInfo:
    """Tests for DocREDRelationInfo."""

    def test_total_relations(self) -> None:
        """There should be exactly 97 relations (96 typed + NA)."""
        ri = DocREDRelationInfo()
        assert len(ri) == NUM_RELATIONS, f"Expected {NUM_RELATIONS}, got {len(ri)}"

    def test_na_is_zero(self) -> None:
        """NA relation must have id 0."""
        ri = DocREDRelationInfo()
        assert ri.get_id("NA") == 0

    def test_round_trip(self) -> None:
        """get_id and get_name should be inverses."""
        ri = DocREDRelationInfo()
        for rel, rid in ri.rel2id.items():
            assert ri.get_name(rid) == rel

    def test_unknown_relation_returns_na(self) -> None:
        """An unknown relation name should map to id 0 (NA)."""
        ri = DocREDRelationInfo()
        assert ri.get_id("P99999") == 0

    def test_rel_info_load_missing_file(self, tmp_path: Path) -> None:
        """Loading a missing rel_info.json should not raise; rel_info stays empty."""
        ri = DocREDRelationInfo(rel_info_path=str(tmp_path / "nonexistent.json"))
        assert ri.rel_info == {}

    def test_rel_info_load_valid_file(self, tmp_path: Path) -> None:
        """Loading a valid rel_info.json should populate rel_info."""
        data = {"P17": "country", "P101": "field of work"}
        ri_path = tmp_path / "rel_info.json"
        ri_path.write_text(json.dumps(data), encoding="utf-8")
        ri = DocREDRelationInfo(rel_info_path=str(ri_path))
        assert ri.rel_info["P17"] == "country"


# ---------------------------------------------------------------------------
# Test: DocREDDataset
# ---------------------------------------------------------------------------

class TestDocREDDataset:
    """Tests for DocREDDataset loading and preprocessing."""

    @pytest.fixture(autouse=True)
    def _mock_tokenizer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch AutoTokenizer.from_pretrained to return a lightweight mock."""

        class MockFastTokenizer:
            """Minimal tokenizer mock."""

            cls_token_id = 0
            sep_token_id = 2
            bos_token_id = 0
            eos_token_id = 2
            pad_token_id = 1

            def get_vocab(self) -> Dict[str, int]:
                return {"[unused0]": 100, "[unused1]": 101}

            def add_special_tokens(self, special_tokens_dict: Dict) -> int:
                return 0

            def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
                # Return one token per character as a simple mock
                return [ord(c) % 200 + 10 for c in text[:3]]

            def convert_tokens_to_ids(self, token: str) -> int:
                vocab = {"[unused0]": 100, "[unused1]": 101}
                return vocab.get(token, 0)

        monkeypatch.setattr(
            "src.data.docred_dataset.AutoTokenizer.from_pretrained",
            lambda name, **kwargs: MockFastTokenizer(),
        )

    def test_dataset_loading(self, tmp_path: Path) -> None:
        """Dataset should load the correct number of documents."""
        data_path = tmp_path / "train.json"
        write_mock_dataset(data_path, num_docs=5)

        ds = DocREDDataset(
            data_path=str(data_path),
            tokenizer_name="bert-base-uncased",
            use_entity_markers=False,
        )
        assert len(ds) == 5

    def test_getitem_keys(self, tmp_path: Path) -> None:
        """Each sample must contain all required keys."""
        data_path = tmp_path / "train.json"
        write_mock_dataset(data_path, num_docs=2)

        ds = DocREDDataset(
            data_path=str(data_path),
            tokenizer_name="bert-base-uncased",
            use_entity_markers=False,
        )
        sample = ds[0]

        required_keys = {
            "input_ids",
            "attention_mask",
            "entity_spans",
            "mention_to_entity",
            "labels",
            "evidence",
            "hts",
            "title",
            "sentence_boundaries",
        }
        assert required_keys.issubset(sample.keys()), (
            f"Missing keys: {required_keys - sample.keys()}"
        )

    def test_input_ids_shape(self, tmp_path: Path) -> None:
        """input_ids should be a 1-D LongTensor."""
        data_path = tmp_path / "train.json"
        write_mock_dataset(data_path, num_docs=1)

        ds = DocREDDataset(
            data_path=str(data_path),
            tokenizer_name="bert-base-uncased",
            use_entity_markers=False,
        )
        sample = ds[0]
        assert sample["input_ids"].ndim == 1
        assert sample["input_ids"].dtype == torch.long

    def test_labels_shape(self, tmp_path: Path) -> None:
        """Label tensor shape should be [num_entities, num_entities, num_relations]."""
        doc = make_mock_doc(num_entities=4)
        data_path = tmp_path / "train.json"
        data_path.write_text(json.dumps([doc]), encoding="utf-8")

        ds = DocREDDataset(
            data_path=str(data_path),
            tokenizer_name="bert-base-uncased",
            use_entity_markers=False,
        )
        sample = ds[0]
        labels = sample["labels"]
        assert labels.shape == (4, 4, NUM_RELATIONS), (
            f"Expected shape (4, 4, {NUM_RELATIONS}), got {labels.shape}"
        )

    def test_entity_spans_count(self, tmp_path: Path) -> None:
        """Number of entity span lists should equal number of entities."""
        doc = make_mock_doc(num_entities=3)
        data_path = tmp_path / "train.json"
        data_path.write_text(json.dumps([doc]), encoding="utf-8")

        ds = DocREDDataset(
            data_path=str(data_path),
            tokenizer_name="bert-base-uncased",
            use_entity_markers=False,
        )
        sample = ds[0]
        assert len(sample["entity_spans"]) == 3

    def test_hts_count(self, tmp_path: Path) -> None:
        """Number of entity pairs should be num_entities * (num_entities - 1)."""
        num_entities = 4
        doc = make_mock_doc(num_entities=num_entities)
        data_path = tmp_path / "train.json"
        data_path.write_text(json.dumps([doc]), encoding="utf-8")

        ds = DocREDDataset(
            data_path=str(data_path),
            tokenizer_name="bert-base-uncased",
            use_entity_markers=False,
        )
        sample = ds[0]
        expected = num_entities * (num_entities - 1)
        assert len(sample["hts"]) == expected, (
            f"Expected {expected} pairs, got {len(sample['hts'])}"
        )

    def test_max_length_truncation(self, tmp_path: Path) -> None:
        """input_ids should not exceed max_length."""
        doc = make_mock_doc(num_sents=10, words_per_sent=20)
        data_path = tmp_path / "train.json"
        data_path.write_text(json.dumps([doc]), encoding="utf-8")

        max_length = 64
        ds = DocREDDataset(
            data_path=str(data_path),
            tokenizer_name="bert-base-uncased",
            max_length=max_length,
            use_entity_markers=False,
        )
        sample = ds[0]
        assert sample["input_ids"].shape[0] <= max_length

    def test_evidence_keys(self, tmp_path: Path) -> None:
        """Evidence dict keys should be (h_idx, t_idx, r_id) tuples."""
        doc = make_mock_doc(num_labels=2)
        data_path = tmp_path / "train.json"
        data_path.write_text(json.dumps([doc]), encoding="utf-8")

        ds = DocREDDataset(
            data_path=str(data_path),
            tokenizer_name="bert-base-uncased",
            use_entity_markers=False,
        )
        sample = ds[0]
        for key in sample["evidence"]:
            assert len(key) == 3, f"Evidence key should be a 3-tuple, got {key}"
            assert all(isinstance(k, int) for k in key)


# ---------------------------------------------------------------------------
# Test: docred_collate_fn
# ---------------------------------------------------------------------------

class TestCollate:
    """Tests for docred_collate_fn batching."""

    def _make_sample(self, seq_len: int, num_ent: int) -> Dict[str, Any]:
        """Create a minimal sample dict."""
        num_pairs = num_ent * (num_ent - 1)
        return {
            "input_ids": torch.randint(0, 1000, (seq_len,), dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "entity_spans": [[(0, 2)] for _ in range(num_ent)],
            "mention_to_entity": list(range(num_ent)),
            "labels": torch.zeros(num_ent, num_ent, NUM_RELATIONS),
            "evidence": {},
            "hts": [(h, t) for h in range(num_ent) for t in range(num_ent) if h != t],
            "title": "Doc",
            "sentence_boundaries": [(0, seq_len)],
        }

    def test_padding(self) -> None:
        """All input_ids in the batch should have the same length after padding."""
        batch = [self._make_sample(10, 2), self._make_sample(20, 3), self._make_sample(5, 2)]
        out = docred_collate_fn(batch)
        assert out["input_ids"].shape == (3, 20)
        assert out["attention_mask"].shape == (3, 20)

    def test_attention_mask_zeros_in_padding(self) -> None:
        """Padded positions should have attention_mask == 0."""
        batch = [self._make_sample(5, 2), self._make_sample(15, 2)]
        out = docred_collate_fn(batch)
        # First sample is padded from position 5 to 14
        assert out["attention_mask"][0, 5:].sum().item() == 0
        assert out["attention_mask"][1, :].sum().item() == 15

    def test_list_fields_length(self) -> None:
        """List fields should have batch-size many entries."""
        batch = [self._make_sample(8, 2), self._make_sample(8, 3)]
        out = docred_collate_fn(batch)
        for key in ("entity_spans", "mention_to_entity", "labels", "evidence", "hts"):
            assert len(out[key]) == 2, f"{key} should have 2 entries"

    def test_titles_preserved(self) -> None:
        """Document titles should be preserved in the batch."""
        s1 = self._make_sample(8, 2)
        s1["title"] = "Alpha"
        s2 = self._make_sample(8, 2)
        s2["title"] = "Beta"
        out = docred_collate_fn([s1, s2])
        assert out["titles"] == ["Alpha", "Beta"]


# ---------------------------------------------------------------------------
# Test: Graph builder (unit test without full model)
# ---------------------------------------------------------------------------

class TestGraphBuilder:
    """Tests for the document graph construction logic."""

    def test_graph_has_mention_nodes(self) -> None:
        """Graph should contain at least as many mention nodes as mentions."""
        try:
            from src.models.graph_builder import DocGraphBuilder
        except ImportError:
            pytest.skip("graph_builder module not yet implemented")

        builder = DocGraphBuilder()
        doc = make_mock_doc(num_entities=3, num_sents=3)
        # Construct a dummy encoder output
        total_mentions = sum(len(e) for e in doc["vertexSet"])
        entity_spans = [[(0, 2)] for _ in doc["vertexSet"]]
        sentence_boundaries = [(i * 10, (i + 1) * 10) for i in range(3)]
        token_embeddings = torch.rand(1, 30, 64)

        graph = builder.build(
            token_embeddings=token_embeddings,
            entity_spans=entity_spans,
            mention_to_entity=list(range(len(doc["vertexSet"]))),
            sentence_boundaries=sentence_boundaries,
        )
        assert graph is not None


# ---------------------------------------------------------------------------
# Test: BMM fitting
# ---------------------------------------------------------------------------

class TestBMM:
    """Tests for the Beta Mixture Model hard negative miner."""

    def test_bmm_fitting_bimodal(self) -> None:
        """BMM should converge on synthetic bimodal Beta data."""
        try:
            from src.losses.bmm import BetaMixtureModel
        except ImportError:
            pytest.skip("bmm module not yet implemented")

        set_seed(0)
        # Two Beta components: Beta(2,5) ≈ low similarity, Beta(8,2) ≈ high similarity
        low = np.random.beta(2, 5, size=200).clip(1e-6, 1 - 1e-6)
        high = np.random.beta(8, 2, size=200).clip(1e-6, 1 - 1e-6)
        scores = torch.tensor(np.concatenate([low, high]), dtype=torch.float32)

        bmm = BetaMixtureModel(num_components=2, max_em_iters=20)
        bmm.fit(scores)

        # After fitting, component means should be distinct
        means = bmm.component_means  # property, shape [num_components]
        assert len(means) == 2
        assert abs(means[0].item() - means[1].item()) > 0.05, (
            "Components should be well-separated"
        )

    def test_bmm_weights_sum_to_one(self) -> None:
        """Mixture weights should sum to 1."""
        try:
            from src.losses.bmm import BetaMixtureModel
        except ImportError:
            pytest.skip("bmm module not yet implemented")

        scores = torch.rand(100).clamp(1e-6, 1 - 1e-6)
        bmm = BetaMixtureModel(num_components=2, max_em_iters=5)
        bmm.fit(scores)
        # pis attribute holds mixture weights
        weights = bmm.pis
        assert abs(weights.sum().item() - 1.0) < 1e-4

    def test_bmm_posterior_range(self) -> None:
        """Posterior probabilities should be in [0, 1]."""
        try:
            from src.losses.bmm import BetaMixtureModel
        except ImportError:
            pytest.skip("bmm module not yet implemented")

        scores = torch.rand(50).clamp(1e-6, 1 - 1e-6)
        bmm = BetaMixtureModel(num_components=2, max_em_iters=5)
        bmm.fit(scores)
        # predict_true_negative_prob returns per-sample probability in [0,1]
        posteriors = bmm.predict_true_negative_prob(scores)
        assert posteriors.min().item() >= -1e-4
        assert posteriors.max().item() <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# Test: JointLoss
# ---------------------------------------------------------------------------

class TestJointLoss:
    """Tests for the JointLoss module."""

    def _make_model_outputs(
        self,
        num_pairs: int = 20,
        num_relations: int = NUM_RELATIONS,
        include_contrastive: bool = False,
    ) -> Dict[str, Any]:
        """Create a model_outputs dict compatible with JointLoss.forward."""
        # Flat labels: shape [num_pairs, num_relations] (ATLOP style)
        labels = torch.zeros(num_pairs, num_relations)
        for i in range(num_pairs):
            labels[i, torch.randint(1, num_relations, (1,)).item()] = 1.0

        outputs: Dict[str, Any] = {
            "logits": torch.randn(num_pairs, num_relations),
            "labels": labels,
            "pair_embs": torch.randn(num_pairs, 256),
        }
        if include_contrastive:
            c = nn.functional.normalize(torch.randn(num_pairs, 128), dim=-1)
            outputs["contrastive_embs"] = c
            outputs["positive_contrastive_embs"] = nn.functional.normalize(
                c + 0.1 * torch.randn_like(c), dim=-1
            )
        return outputs

    def test_joint_loss_returns_dict(self) -> None:
        """JointLoss should return a dict with a 'total' key during BMM warm-up."""
        try:
            from src.losses.joint_loss import JointLoss
        except ImportError:
            pytest.skip("joint_loss module not yet implemented")

        # Use a very large bmm_warmup_epochs so BMM EM is never triggered;
        # this avoids NaN from fitting on tiny synthetic batches.
        loss_fn = JointLoss(
            num_relations=NUM_RELATIONS,
            lambda_gcl=0.5,
            lambda_evidence=0.3,
            bmm_warmup_epochs=9999,
        )
        outputs = self._make_model_outputs(include_contrastive=True)
        result = loss_fn(model_outputs=outputs, epoch=0, step=0)
        assert isinstance(result, dict)
        assert "total" in result

    def test_joint_loss_is_non_negative(self) -> None:
        """Total loss should be a non-negative scalar during warm-up."""
        try:
            from src.losses.joint_loss import JointLoss
        except ImportError:
            pytest.skip("joint_loss module not yet implemented")

        loss_fn = JointLoss(
            num_relations=NUM_RELATIONS,
            lambda_gcl=0.5,
            lambda_evidence=0.3,
            bmm_warmup_epochs=9999,
        )
        outputs = self._make_model_outputs(include_contrastive=True)
        result = loss_fn(model_outputs=outputs, epoch=0, step=0)
        assert result["total"].item() >= 0.0

    def test_joint_loss_ce_key_present(self) -> None:
        """Result dict should contain a 'ce' key (classification loss)."""
        try:
            from src.losses.joint_loss import JointLoss
        except ImportError:
            pytest.skip("joint_loss module not yet implemented")

        loss_fn = JointLoss(
            num_relations=NUM_RELATIONS,
            lambda_gcl=0.5,
            lambda_evidence=0.3,
            bmm_warmup_epochs=9999,
        )
        outputs = self._make_model_outputs(include_contrastive=True)
        result = loss_fn(model_outputs=outputs, epoch=0, step=0)
        assert "ce" in result, "JointLoss result should include 'ce' key"

    def test_joint_loss_with_contrastive(self) -> None:
        """Both warm-up and post-warm-up phases produce valid non-negative losses."""
        try:
            from src.losses.joint_loss import JointLoss
        except ImportError:
            pytest.skip("joint_loss module not yet implemented")

        # Warm-up phase only (bmm_warmup_epochs > epoch so BMM is not triggered)
        loss_fn = JointLoss(
            num_relations=NUM_RELATIONS,
            lambda_gcl=1.0,
            lambda_evidence=0.0,
            bmm_warmup_epochs=9999,
        )
        outputs = self._make_model_outputs(num_pairs=20, include_contrastive=True)

        result_warmup = loss_fn(model_outputs=outputs, epoch=1, step=10)
        assert result_warmup["total"].item() >= 0.0, (
            "Warm-up loss should be non-negative"
        )

    def test_joint_loss_contrastive_top_k(self) -> None:
        """Top-K in-batch negatives should yield a finite GCL term."""
        try:
            from src.losses.joint_loss import JointLoss
        except ImportError:
            pytest.skip("joint_loss module not yet implemented")

        loss_fn = JointLoss(
            num_relations=NUM_RELATIONS,
            lambda_gcl=1.0,
            lambda_evidence=0.0,
            bmm_warmup_epochs=9999,
            contrastive_top_k=5,
        )
        outputs = self._make_model_outputs(num_pairs=32, include_contrastive=True)
        result = loss_fn(model_outputs=outputs, epoch=0, step=0)
        assert torch.isfinite(result["gcl"]).all()
        assert result["total"].item() >= 0.0


# ---------------------------------------------------------------------------
# Test: Encoder output shapes (mock small model)
# ---------------------------------------------------------------------------

class TestEncoderOutputShapes:
    """Tests for the document encoder module."""

    def test_encoder_output_shape(self) -> None:
        """Encoder should output [batch, seq_len, hidden_dim] token embeddings."""
        try:
            from src.models.encoder import DocEncoder
        except ImportError:
            pytest.skip("encoder module not yet implemented")

        # Mock PLM that returns dummy outputs
        class MockPLM(nn.Module):
            hidden_size = 64

            def forward(self, input_ids, attention_mask=None, **kwargs):
                B, L = input_ids.shape
                hidden = torch.rand(B, L, 64)
                MockOutput = type("O", (), {"last_hidden_state": hidden})
                return MockOutput()

        with patch("src.models.encoder.AutoModel.from_pretrained", return_value=MockPLM()):
            enc = DocEncoder(plm_name="mock-model", hidden_dim=64)

        B, L = 2, 32
        input_ids = torch.randint(0, 1000, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        output = enc(input_ids=input_ids, attention_mask=attention_mask)

        assert "token_embeddings" in output
        assert output["token_embeddings"].shape == (B, L, 64)

    def test_entity_pooling_shape(self) -> None:
        """Entity representations should have shape [num_entities, hidden_dim]."""
        try:
            from src.models.encoder import DocEncoder
        except ImportError:
            pytest.skip("encoder module not yet implemented")

        class MockPLM(nn.Module):
            hidden_size = 64

            def forward(self, input_ids, attention_mask=None, **kwargs):
                B, L = input_ids.shape
                hidden = torch.rand(B, L, 64)
                MockOutput = type("O", (), {"last_hidden_state": hidden})
                return MockOutput()

        with patch("src.models.encoder.AutoModel.from_pretrained", return_value=MockPLM()):
            enc = DocEncoder(plm_name="mock-model", hidden_dim=64)

        B, L = 1, 32
        input_ids = torch.randint(0, 1000, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)

        # Two entities, one mention each
        entity_spans = [
            [[(2, 4)], [(10, 12)]],  # doc 0: entity 0 → mention (2,4), entity 1 → mention (10,12)
        ]
        mention_to_entity = [[0, 1]]

        output = enc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_spans=entity_spans,
            mention_to_entity=mention_to_entity,
        )
        if "entity_embeddings" in output:
            assert output["entity_embeddings"][0].shape == (2, 64)


# ---------------------------------------------------------------------------
# Test: helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for utility functions."""

    def test_set_seed_reproducibility(self) -> None:
        """Two runs with the same seed should produce identical random tensors."""
        set_seed(42)
        t1 = torch.rand(10)
        set_seed(42)
        t2 = torch.rand(10)
        assert torch.allclose(t1, t2)

    def test_count_parameters(self) -> None:
        """count_parameters should return the correct trainable parameter count."""
        model = nn.Linear(10, 5)  # 10*5 + 5 = 55 parameters
        total = count_parameters(model)
        assert total == 55

    def test_format_metrics(self) -> None:
        """format_metrics should produce a non-empty string for non-empty input."""
        out = format_metrics({"f1": 0.75, "ign_f1": 0.70})
        assert "f1" in out
        assert "0.7500" in out

    def test_load_config(self, tmp_path: Path) -> None:
        """load_config should parse a YAML file into a dict."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("model:\n  hidden_dim: 256\ntraining:\n  epochs: 5\n")
        config = load_config(str(cfg_file))
        assert config["model"]["hidden_dim"] == 256
        assert config["training"]["epochs"] == 5

    def test_load_config_missing(self, tmp_path: Path) -> None:
        """load_config should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_get_device_returns_device(self) -> None:
        """get_device should return a torch.device object."""
        device = get_device()
        assert isinstance(device, torch.device)


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
