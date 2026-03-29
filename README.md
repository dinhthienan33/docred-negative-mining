# DocRED SOTA Pipeline: LLM + GNN + Debiased Graph Contrastive Learning

A state-of-the-art document-level relation extraction pipeline for the [DocRED benchmark](https://github.com/thunlp/DocRED), combining a pre-trained language model encoder (DeBERTa-v3-large), a relational graph neural network, and a debiased graph contrastive learning objective with evidence-aware hard negative mining.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Document                           │
│   "Alice was born in Paris , France ..."                        │
└────────────────────────┬────────────────────────────────────────┘
                         │  Tokenization + entity markers
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          Module 1: Document Encoder (DeBERTa-v3-large)          │
│                                                                 │
│  Token embeddings [B, seq_len, 1024]                            │
│  → Mention pooling (attention-weighted)                         │
│  → Entity pooling (logsumexp across mentions)                   │
│  → Entity embeddings [num_entities, 1024]                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│      Module 2: Document Graph Builder (PyG HeteroData)          │
│                                                                 │
│  Node types:  mention · entity · sentence                       │
│  Edge types:  intra-sentence · inter-sentence · coreference     │
│               entity↔mention · adjacent-sentence · evidence     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          Module 3: GNN Reasoning (R-GCN, 3 layers)              │
│                                                                 │
│  Basis decomposition (4 bases) · multi-head attention (4 heads) │
│  Residual connections · LayerNorm                               │
│  → Refined entity representations [num_entities, 256]           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          Module 4: Triple Head                                  │
│                                                                 │
│  For each (h, t) pair:                                          │
│    pair_emb = MLP([h; t; h*t; context])                         │
│    z(h,t,r) = MLP([pair_emb; rel_emb_r])   (triple repr)        │
│  Classification: bilinear scorer → logits [num_pairs, 97]       │
│  Contrastive head: project z to unit sphere [num_pairs, 256]    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          Module 7: Joint Loss                                   │
│                                                                 │
│  L_total = L_CE + λ_gcl·L_contrastive + λ_evid·L_evidence_cl   │
│                                                                 │
│  L_CE:          ATLOP-style adaptive threshold cross-entropy    │
│  L_contrastive: BMM-reweighted InfoNCE (τ=0.07)                 │
│  L_evidence_cl: Evidence-aware InfoNCE                          │
│                                                                 │
│  ┌─────────────────┐    ┌────────────────────────┐             │
│  │ Module 5: BMM   │    │ Module 6: Evidence Neg. │             │
│  │ Beta Mixture    │    │ Co-occur + sent overlap │             │
│  │ hard neg mining │    │ hard/medium/easy split  │             │
│  └─────────────────┘    └────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
docred_pipeline/
├── configs/
│   └── default.yaml          # All hyperparameters
├── src/
│   ├── models/
│   │   ├── encoder.py         # PLM/LLM document encoder
│   │   ├── graph_builder.py   # Doc-level heterogeneous graph construction
│   │   ├── gnn.py             # R-GCN with multi-head attention
│   │   ├── triple_head.py     # Triple representation + relation classifier
│   │   └── pipeline.py        # Full model combining all modules
│   ├── losses/
│   │   ├── bmm.py             # Beta Mixture Model hard negative miner
│   │   ├── evidence_negatives.py  # Evidence-aware negative mining
│   │   └── joint_loss.py      # Joint CE + contrastive + BMM loss
│   ├── graph/
│   │   └── triple_graph.py    # Triple-level graph for GCL
│   ├── data/
│   │   └── docred_dataset.py  # DocRED data loading & preprocessing
│   └── utils/
│       └── helpers.py         # Utility functions
├── scripts/
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation & submission generation
├── tests/
│   └── test_pipeline.py       # Unit tests
├── data/                      # DocRED data files (not included)
├── outputs/                   # Checkpoints & logs (auto-created)
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/docred-gcl.git
cd docred-gcl
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on torch-geometric:** Install the version matching your CUDA toolkit.
> See the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
# Example for CUDA 12.1 + PyTorch 2.2:
pip install torch-geometric
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

---

## Data Preparation

### Download DocRED

```bash
mkdir -p data
# Official DocRED dataset (requires LDC agreement for full version)
# Distantly supervised + human-annotated splits available at:
# https://github.com/thunlp/DocRED/tree/master/data

# Place the following files in ./data/:
#   train_annotated.json
#   dev.json
#   test.json
#   rel_info.json
```

### File Format

Each JSON file is a list of document dicts:

```json
{
  "title": "Lark Force",
  "sents": [["Lark", "Force", "was", ...], ["It", "was", ...]],
  "vertexSet": [
    [{"name": "Lark Force", "sent_id": 0, "pos": [0, 2], "type": "ORG"}],
    [{"name": "Australia",  "sent_id": 0, "pos": [5, 6], "type": "LOC"}]
  ],
  "labels": [
    {"h": 0, "t": 1, "r": "P17", "evidence": [0, 1]}
  ]
}
```

---

## Training

### Basic training run

```bash
python scripts/train.py --config configs/default.yaml
```

### With CLI overrides

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --overrides training.epochs=20 model.use_lora=true training.fp16=true
```

### Resume from checkpoint

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --resume outputs/best_model.pt
```

### Mixed-precision + gradient accumulation (default)

The default config uses `fp16=true` and `grad_accumulation_steps=8` for an
effective batch size of 32 with 4 documents per GPU step.

---

## Evaluation

### Evaluate on dev set with threshold search

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint outputs/best_model.pt \
    --split dev \
    --threshold_search
```

### Evaluate on test set with a fixed threshold

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint outputs/best_model.pt \
    --split test \
    --threshold 0.35 \
    --output outputs/test_predictions.json
```

### Output format (DocRED official submission)

```json
[
  {"title": "Lark Force", "h_idx": 0, "t_idx": 1, "r": "P17", "evidence": [0, 3]},
  ...
]
```

---

## Configuration Guide

All hyperparameters live in `configs/default.yaml`.

| Section | Key | Description | Default |
|---------|-----|-------------|---------|
| `model` | `plm_name` | HuggingFace model name | `microsoft/deberta-v3-large` |
| `model` | `use_lora` | Enable LoRA PEFT | `false` |
| `model` | `gnn_layers` | R-GCN depth | `3` |
| `model` | `num_relations` | Total relation classes (incl. NA) | `97` |
| `data` | `max_length` | Max subword tokens per document | `1024` |
| `data` | `use_entity_markers` | Insert `[unused0]`/`[unused1]` markers | `true` |
| `training` | `epochs` | Training epochs | `30` |
| `training` | `batch_size` | Documents per GPU step | `4` |
| `training` | `grad_accumulation_steps` | Gradient accumulation | `8` |
| `training` | `learning_rate_plm` | PLM backbone LR | `2e-5` |
| `training` | `learning_rate_other` | GNN/head LR | `1e-3` |
| `training` | `fp16` | Mixed-precision | `true` |
| `loss` | `lambda_gcl` | GCL loss weight | `0.5` |
| `loss` | `lambda_evidence` | Evidence-CL loss weight | `0.3` |
| `loss` | `contrastive_temperature` | InfoNCE temperature τ | `0.07` |
| `loss` | `bmm_warmup_epochs` | Epochs before BMM activates | `3` |
| `evaluation` | `threshold_search` | Grid-search threshold on dev | `true` |

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

Tests that depend on unimplemented modules are automatically skipped with an
informative message.  The following tests run without any external dependencies:

- `TestDocREDRelationInfo` — relation mapping tests
- `TestDocREDDataset` — tokenization & preprocessing (mocked tokenizer)
- `TestCollate` — batch collation and padding
- `TestHelpers` — seed setting, parameter counting, config loading, metrics
- `TestBMM` — Beta Mixture Model (skipped if `bmm.py` not yet present)
- `TestJointLoss` — loss computation (skipped if `joint_loss.py` not yet present)
- `TestEncoderOutputShapes` — encoder shapes (skipped if `encoder.py` not yet present)

---

## Architecture Explanation

### Why DeBERTa-v3-large?

DeBERTa-v3-large achieves best-in-class performance on NLU benchmarks among
encoder-only models.  Its disentangled attention mechanism separates content
and positional information, giving sharper entity representations.

### Why R-GCN + basis decomposition?

Document graphs have many edge types (6+).  Basis decomposition avoids
parameter explosion by sharing a small number of basis matrices across all
edge types, controlled by `gnn_bases=4`.

### Why Beta Mixture Model for hard negatives?

In contrastive learning, false negatives (semantically similar pairs labelled
negative) harm training.  The BMM (inspired by ProGCL) distinguishes true
negatives from false negatives by fitting a 2-component Beta distribution over
similarity scores and up-weighting true negatives in the InfoNCE denominator.

### Why evidence-aware negatives?

DocRED relation triples are annotated with supporting sentence indices.
Relations that share evidence sentences are much harder to distinguish.  We
explicitly construct such hard negatives (high sentence overlap × high
co-occurrence frequency) to focus the contrastive objective where it matters.

---

## Citation

If you use this pipeline, please cite the following works:

```bibtex
@inproceedings{yao2019docred,
  title     = {DocRED: A Large-Scale Document-Level Relation Extraction Dataset},
  author    = {Yao, Yuan and Ye, Deming and Li, Peng and Han, Xu and Lin, Yankai
               and Liu, Zhiyuan and Liu, Zhenghao and Huang, Linfeng and Zhou, Jie
               and Sun, Maosong},
  booktitle = {ACL},
  year      = {2019}
}

@inproceedings{zhou2022atlop,
  title     = {Adaptive Thresholding and Localized Context Pooling for Document-Level
               Relation Extraction},
  author    = {Zhou, Wenxuan and Huang, Kevin and Ma, Tengyu and Huang, Jing},
  booktitle = {AAAI},
  year      = {2022}
}

@inproceedings{han2022progcl,
  title     = {ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning},
  author    = {Xia, Jun and Wu, Lirong and Chen, Jintao and Hu, Bozhen
               and Li, Stan Z.},
  booktitle = {ICML},
  year      = {2022}
}

@inproceedings{he2021deberta,
  title     = {DeBERTa: Decoding-Enhanced BERT with Disentangled Attention},
  author    = {He, Pengcheng and Liu, Xiaodong and Gao, Jianfeng and Chen, Weizhu},
  booktitle = {ICLR},
  year      = {2021}
}
```

---

## License

MIT License. See `LICENSE` for details.
