# Architecture — Hybrid LSTM-XGBoost Pipeline

## Why a Two-Stage Hybrid?

Neither deep learning nor gradient boosting alone gives optimal results on NLP classification tasks:

- **LSTM alone** learns rich sequential representations but its sigmoid output is sensitive to class imbalance and threshold choice
- **XGBoost alone** on raw text (bag-of-words / TF-IDF) loses sequential context entirely
- **Hybrid:** LSTM learns the representation, XGBoost makes the final decision — getting the best of both

## Stage 1 — Bidirectional LSTM (Feature Learner)

```
Input: tokenized, padded sequence  (shape: batch × 300)
  │
  ▼
Embedding Layer  (vocab × 128)
  │
  ▼
BiLSTM Layer 1  (256 units, return_sequences=True)
  │   Reads text left→right AND right→left simultaneously
  │   Captures long-range dependencies in news articles
  ▼
Dropout (0.3)
  │
  ▼
BiLSTM Layer 2  (128 units, return_sequences=False)
  │   Compresses the full sequence into a single vector
  ▼
Dropout (0.3)
  │
  ▼
Dense (64, ReLU)  ◄── THIS is the feature vector passed to XGBoost
  │
  ▼
Dense (1, Sigmoid)  ◄── LSTM standalone prediction (auxiliary output)
```

**Why Bidirectional?**
Standard LSTM reads left-to-right only. Fake news detection benefits from full sentence context — a word's meaning may depend on what comes after it. BiLSTM reads both directions, giving each timestep access to the full surrounding context.

## Stage 2 — XGBoost (Classifier)

```
Input: 64-dim feature vector from LSTM Dense layer
  │
  ▼
XGBoost Classifier
  │   n_estimators=200, max_depth=6, lr=0.1
  │   Trained on LSTM features, not raw text
  ▼
Output: P(fake) ∈ [0, 1]
```

**Why XGBoost on top of LSTM features?**
The Dense(64) layer activations encode compressed semantic information about the article. XGBoost's ensemble of decision trees can find non-linear boundaries in this learned feature space that the LSTM's single sigmoid layer may miss — particularly useful for borderline cases near the decision boundary.

## Data Flow Summary

```
train.csv
  │
  ├─► [80%] Training set
  │     │
  │     ├─► Tokenizer.fit()
  │     ├─► LSTM.fit()       (5 epochs, batch=64)
  │     └─► XGBoost.fit()    (on LSTM features)
  │
  └─► [20%] Validation set
        │
        └─► Evaluate both models → confusion matrix, ROC curve
```

## Key Design Decisions

**MAX_SEQ_LEN = 300:** News articles vary widely in length. 300 tokens covers the majority of article bodies while keeping memory usage manageable. Longer sequences are truncated; shorter ones are zero-padded.

**EMBED_DIM = 128:** Learned from scratch on the training corpus. No pre-trained embeddings used (paper methodology) — keeps the pipeline self-contained and domain-adaptable.

**LSTM feature extraction from Dense(64), not Dense(1):** The 64-dim layer is a richer, higher-dimensional representation than the final scalar sigmoid output. XGBoost gets a proper feature space to work in, not just a single probability score.
