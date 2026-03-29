# DocRED SOTA Pipeline: LLM + GNN + Debiased GCL

## Full Architecture Specification

### Project Structure
```
docred_pipeline/
├── configs/
│   └── default.yaml          # All hyperparameters
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py         # Module 1: PLM/LLM document encoder
│   │   ├── graph_builder.py   # Module 2: Doc-level graph construction
│   │   ├── gnn.py             # Module 3: R-GCN / Graph Transformer
│   │   ├── triple_head.py     # Module 4: Triple representation + relation classifier
│   │   └── pipeline.py        # Full model combining all modules
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── bmm.py             # Module 5: Beta Mixture Model for hard negative mining
│   │   ├── evidence_negatives.py  # Module 6: Evidence-aware negative mining
│   │   └── joint_loss.py      # Module 7: Joint CE + contrastive + BMM loss
│   ├── graph/
│   │   ├── __init__.py
│   │   └── triple_graph.py    # Triple-level graph for GCL
│   ├── data/
│   │   ├── __init__.py
│   │   └── docred_dataset.py  # Module 8: DocRED data loading
│   └── utils/
│       ├── __init__.py
│       └── helpers.py         # Utility functions
├── scripts/
│   ├── train.py               # Module 9: Training script
│   └── evaluate.py            # Module 10: Evaluation script
├── tests/
│   └── test_pipeline.py       # Unit tests
├── requirements.txt
└── README.md
```

### Module Specifications

#### Module 1: Document Encoder (`src/models/encoder.py`)
- Backbone: DeBERTa-v3-large or RoBERTa-large (configurable)
- Optional: LoRA/QLoRA PEFT adapter for LLM-style fine-tuning
- Input: tokenized document (subword tokens)
- Output: contextual token embeddings [batch, seq_len, hidden_dim]
- Entity mention pooling: attention-weighted average of mention tokens
- Entity-level representation: logsumexp pooling across mentions of same entity

#### Module 2: Document Graph Builder (`src/models/graph_builder.py`)
- Node types:
  - mention nodes (one per entity mention span)
  - entity nodes (one per unique entity, aggregated from mentions)
  - sentence nodes (one per sentence)
- Edge types:
  - intra-sentence: mention↔mention in same sentence
  - inter-sentence: mention↔sentence containment
  - coreference: mention↔mention of same entity
  - entity↔mention: entity to its mentions
  - adjacent-sentence: sentence_i ↔ sentence_{i+1}
  - evidence: sentence↔entity-pair (when evidence annotations available)
- Output: heterogeneous graph (PyG HeteroData) with node features from encoder

#### Module 3: GNN Reasoning (`src/models/gnn.py`)
- Architecture: Relational Graph Convolutional Network (R-GCN) with multi-head attention
- 2-3 layers of message passing
- Separate weight matrices per edge type (or basis decomposition for efficiency)
- Residual connections + LayerNorm
- Output: refined entity representations [num_entities, hidden_dim]

#### Module 4: Triple Head (`src/models/triple_head.py`)
- For each entity pair (h, t):
  - Concatenate [h_emb; t_emb; h_emb * t_emb; context_emb]
  - Project through MLP
- Relation embeddings: learnable embedding table [num_relations, rel_dim]
- Triple representation z(h,t,r) = MLP([pair_emb; rel_emb])
- Classification head: bilinear scorer or MLP → logits over relations
- Contrastive head: project z(h,t,r) to unit sphere for contrastive loss

#### Module 5: BMM Hard Negative Mining (`src/losses/bmm.py`)
- ProGCL-style Beta Mixture Model:
  1. Collect similarity scores between anchor triple and all negatives in batch
  2. Fit 2-component Beta Mixture (EM algorithm, ~10 iterations)
  3. Component with higher mean = false negatives, lower mean = true negatives
  4. For each negative with similarity s: compute p(true_negative | s)
  5. Use p(true_negative | s) as weight in contrastive loss
- BMM parameters: alpha1, beta1, alpha2, beta2, mixing weight pi
- EM fitting: E-step (posterior responsibilities), M-step (MLE for Beta params)
- Warm-up: skip BMM for first N epochs, use uniform weights

#### Module 6: Evidence-Aware Negatives (`src/losses/evidence_negatives.py`)
- DocRED-specific hard negative construction:
  - For each (h, t, r) with evidence sentences E_r:
    - Find candidate negative relations r' that:
      a) Co-occur frequently with r across dataset
      b) Share evidence sentence overlap with E_r
      c) Are not labeled for this (h, t) pair
  - Score hardness: overlap(E_r, E_r') * freq_cooccur(r, r')
- Evidence-aware negative sampling:
  - Hard negatives: high evidence overlap + high co-occurrence
  - Medium negatives: some overlap or co-occurrence
  - Easy negatives: random other relations
- Integration with BMM: use evidence hardness as prior, BMM refines

#### Module 7: Joint Loss (`src/losses/joint_loss.py`)
- L_total = L_CE + λ_gcl * L_contrastive + λ_evidence * L_evidence_cl
- L_CE: adaptive thresholding cross-entropy (ATLOP-style) for multi-label
- L_contrastive: BMM-reweighted InfoNCE over triple graph
  - For anchor z_i, positives z_i+ (augmented views), negatives z_j:
  - L = -log[ exp(sim(z_i, z_i+)/τ) / (exp(sim(z_i, z_i+)/τ) + Σ_j w_j * exp(sim(z_i, z_j)/τ)) ]
  - w_j = p(true_negative | sim(z_i, z_j)) from BMM
- L_evidence_cl: same formulation but negatives sampled evidence-aware
- Augmentation for positive pairs:
  - Dropout-based: two forward passes with different dropout masks
  - Evidence masking: randomly mask 1 evidence sentence
  - Entity mention dropping: randomly drop 1 mention of a multi-mention entity

#### Module 8: DocRED Dataset (`src/data/docred_dataset.py`)
- Load DocRED JSON format (train_annotated.json, dev.json, test.json)
- Tokenization with PLM tokenizer
- Entity mention span mapping (char→token offsets)
- Relation label matrix construction [num_entities, num_entities, num_relations]
- Evidence sentence extraction per relation triple
- Collation: batching documents with padding

#### Module 9: Training (`scripts/train.py`)
- Training loop with:
  - Warm-up phase: CE-only for first few epochs
  - Joint phase: CE + contrastive with BMM gradually introduced
  - BMM update: re-fit every K steps
- Optimizer: AdamW with linear warm-up + cosine decay
- Gradient accumulation for large models
- Mixed precision (fp16/bf16)
- Logging: wandb/tensorboard
- Checkpointing: save best model by dev F1

#### Module 10: Evaluation (`scripts/evaluate.py`)
- Metrics: F1, Ign F1 (ignoring shared train/dev entity pairs)
- Inference with adaptive threshold (learned or searched)
- Output: relation predictions in DocRED submission format
- Analysis: per-relation F1 breakdown

### Key Hyperparameters (configs/default.yaml)
- PLM: microsoft/deberta-v3-large
- GNN layers: 3, heads: 4
- Hidden dim: 1024, GNN dim: 256
- Triple dim: 512, Relation emb dim: 128
- BMM warm-up epochs: 3, EM iterations: 10
- Contrastive temperature τ: 0.07
- Loss weights: λ_gcl=0.5, λ_evidence=0.3
- Batch size: 4, grad accumulation: 8
- Learning rate: 2e-5 (PLM), 1e-3 (GNN/heads)
- Epochs: 30, patience: 5

### DocRED Relation Set
- 96 relation types + NA (no relation)
- Multi-label: entity pair can have multiple relations
- Evidence: each relation triple annotated with supporting sentences
