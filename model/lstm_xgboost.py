"""
Hybrid LSTM-XGBoost Fake News Detector
=======================================
Implementation of the model described in:

  MV Sujan Kumar, Ganesh Khekare, Anurup Sankriti,
  "Enhancing Knowledge Management Integrity through Fake News Detection:
   A Hybrid LSTM-XGBoost Approach for Cybersecurity"
  In: Handbook of Research on Cybersecurity Issues and Challenges
      for Business and FinTech Applications, IGI Global / CRC Press, 2024.
  DOI: https://doi.org/10.1201/9781003498094-9

Pipeline
--------
Raw text
  └─► Tokenize & Pad (maxlen=300)
        └─► Embedding Layer
              └─► BiLSTM (256 units) ──► Dropout
                    └─► BiLSTM (128 units) ──► Dropout
                          └─► Dense (64, ReLU)
                                ├─► Dense (1, Sigmoid)  [LSTM standalone]
                                └─► Feature vector ──► XGBoost ──► Final prediction
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

import xgboost as xgb

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyperparameters (from paper) ───────────────────────────────────────────────
MAX_VOCAB    = 50_000   # tokenizer vocabulary cap
MAX_SEQ_LEN  = 300      # padding / truncation length  (paper §3.2)
EMBED_DIM    = 128      # word embedding dimension
LSTM_UNITS_1 = 256      # first  BiLSTM layer          (paper §3.3)
LSTM_UNITS_2 = 128      # second BiLSTM layer          (paper §3.3)
DROPOUT_RATE = 0.3
DENSE_UNITS  = 64
BATCH_SIZE   = 64       # paper §3.4
EPOCHS       = 5        # paper §3.4
TEST_SIZE    = 0.20     # 80/20 train/val split        (paper §3.1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(train_path: str) -> pd.DataFrame:
    """
    Load and clean the Kaggle fake news dataset.

    Columns: id, title, author, text, label
      label: 0 = real news, 1 = fake news   (paper §5.1)

    Dataset: https://www.kaggle.com/competitions/fake-news/data
    """
    df = pd.read_csv(train_path)

    # Combine title + author + text for richer input signal
    df["title"]   = df["title"].fillna("")
    df["author"]  = df["author"].fillna("")
    df["text"]    = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["author"] + " " + df["text"]

    df = df[["content", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. TOKENIZATION & PADDING  (paper §3.2)
# ══════════════════════════════════════════════════════════════════════════════

def build_tokenizer(texts: list) -> Tokenizer:
    tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    return tok


def encode(tokenizer: Tokenizer, texts: list) -> np.ndarray:
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(
        seqs, maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. LSTM MODEL  (paper §3.3)
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm(vocab_size: int) -> tf.keras.Model:
    """
    Architecture (paper Figure 1):
      Embedding
        → BiLSTM(256, return_sequences=True)  → Dropout
        → BiLSTM(128, return_sequences=False) → Dropout
        → Dense(64, ReLU)
        → Dense(1, Sigmoid)
    """
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBED_DIM,
            input_length=MAX_SEQ_LEN
        ),
        Bidirectional(LSTM(
            LSTM_UNITS_1,
            return_sequences=True,
            dropout=DROPOUT_RATE,
            recurrent_dropout=0.1
        )),
        Dropout(DROPOUT_RATE),
        Bidirectional(LSTM(
            LSTM_UNITS_2,
            return_sequences=False,
            dropout=DROPOUT_RATE,
            recurrent_dropout=0.1
        )),
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE EXTRACTION  (paper §3.5)
#    Tap the penultimate Dense(64) layer — its activations become XGBoost input
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_extractor(lstm_model: tf.keras.Model) -> tf.keras.Model:
    """
    Returns a sub-model that outputs the 64-dim Dense layer activations
    instead of the final sigmoid prediction.
    """
    feature_layer = lstm_model.layers[-2]   # Dense(64, relu)
    return Model(inputs=lstm_model.input, outputs=feature_layer.output)


def extract_features(extractor: tf.keras.Model,
                     X: np.ndarray) -> np.ndarray:
    return extractor.predict(X, batch_size=BATCH_SIZE, verbose=0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. XGBOOST  (paper §3.6)
# ══════════════════════════════════════════════════════════════════════════════

def build_xgboost() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION UTILITIES  (paper §3.7 + §6)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, y_prob=None, model_name="Model") -> dict:
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred)  * 100,
        "precision": precision_score(y_true, y_pred) * 100,
        "recall":    recall_score(y_true, y_pred)    * 100,
        "f1":        f1_score(y_true, y_pred)        * 100,
    }
    print(f"\n{'─'*48}")
    print(f"  {model_name}")
    print(f"{'─'*48}")
    for k, v in metrics.items():
        print(f"  {k.capitalize():<12}: {v:.2f}%")
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["auc"] = auc(fpr, tpr)
        print(f"  AUC          : {metrics['auc']:.4f}")
    print(f"{'─'*48}")
    return metrics


def plot_confusion_matrix(y_true, y_pred,
                          title="Confusion Matrix",
                          save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Real (0)", "Fake (1)"],
        yticklabels=["Real (0)", "Fake (1)"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def plot_roc(y_true, y_prob, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def print_comparison_table(hybrid_metrics: dict):
    """Reproduce paper Table 1.4."""
    baselines = [
        ("Naive Bayes",   71.47, 99.00, 33.90, 50.59),
        ("Random Forest", 90.81, 95.99, 82.09, 88.50),
        ("CNN",           91.60, 90.48, 89.96, 90.22),
    ]
    print("\n" + "═"*58)
    print("  Model Comparison  (paper Table 1.4)")
    print("═"*58)
    print(f"  {'Model':<24} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("─"*58)
    for name, acc, prec, rec, f1 in baselines:
        print(f"  {name:<24} {acc:>5.2f} {prec:>6.2f} {rec:>6.2f} {f1:>6.2f}")
    hm = hybrid_metrics
    print(f"  {'LSTM+XGBoost (ours)':<24} "
          f"{hm['accuracy']:>5.2f} "
          f"{hm['precision']:>6.2f} "
          f"{hm['recall']:>6.2f} "
          f"{hm['f1']:>6.2f}  ◄ best")
    print("═"*58)


# ══════════════════════════════════════════════════════════════════════════════
# 7. FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(train_path: str, save_figures: bool = True) -> dict:
    """
    End-to-end pipeline replicating the paper's methodology.

    Parameters
    ----------
    train_path   : path to train.csv  (Kaggle fake-news dataset)
    save_figures : save confusion matrix + ROC curve to figures/

    Returns
    -------
    dict with evaluation metrics for LSTM standalone and hybrid model
    """

    # 1. Load ──────────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df = load_data(train_path)
    print(f"  Samples: {len(df):,}  |  "
          f"Fake: {df['label'].sum():,}  |  "
          f"Real: {(df['label']==0).sum():,}")

    X_text = df["content"].tolist()
    y      = df["label"].values

    # 2. Split ─────────────────────────────────────────────────────────────────
    print("\n[2/7] Train / val split  (80 / 20)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"  Train: {len(X_tr):,}  |  Val: {len(X_val):,}")

    # 3. Tokenize & pad ────────────────────────────────────────────────────────
    print("\n[3/7] Tokenizing & padding  (maxlen={})...".format(MAX_SEQ_LEN))
    tokenizer  = build_tokenizer(X_tr)
    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    print(f"  Vocabulary size: {vocab_size:,}")

    X_tr_pad  = encode(tokenizer, X_tr)
    X_val_pad = encode(tokenizer, X_val)

    # 4. Train LSTM ────────────────────────────────────────────────────────────
    print(f"\n[4/7] Training Bidirectional LSTM  (epochs={EPOCHS})...")
    lstm_model = build_lstm(vocab_size)
    lstm_model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    lstm_model.fit(
        X_tr_pad, y_tr,
        validation_data=(X_val_pad, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    # LSTM standalone eval
    y_prob_lstm = lstm_model.predict(X_val_pad, batch_size=BATCH_SIZE).flatten()
    y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)
    lstm_metrics = evaluate(y_val, y_pred_lstm, y_prob_lstm, "BiLSTM (standalone)")

    # 5. Extract features ──────────────────────────────────────────────────────
    print("\n[5/7] Extracting LSTM feature vectors for XGBoost...")
    extractor = build_feature_extractor(lstm_model)
    feats_tr  = extract_features(extractor, X_tr_pad)
    feats_val = extract_features(extractor, X_val_pad)
    print(f"  Feature shape (train): {feats_tr.shape}")

    # 6. Train XGBoost ─────────────────────────────────────────────────────────
    print("\n[6/7] Training XGBoost on LSTM features...")
    xgb_model = build_xgboost()
    xgb_model.fit(
        feats_tr, y_tr,
        eval_set=[(feats_val, y_val)],
        verbose=False
    )

    y_prob_hybrid = xgb_model.predict_proba(feats_val)[:, 1]
    y_pred_hybrid = xgb_model.predict(feats_val)
    hybrid_metrics = evaluate(
        y_val, y_pred_hybrid, y_prob_hybrid, "LSTM + XGBoost (Hybrid)"
    )

    # 7. Plots & table ─────────────────────────────────────────────────────────
    print("\n[7/7] Generating plots...")
    cm_path  = "figures/confusion_matrix_hybrid.png" if save_figures else None
    roc_path = "figures/roc_curve_hybrid.png"        if save_figures else None

    plot_confusion_matrix(
        y_val, y_pred_hybrid,
        title="Confusion Matrix — LSTM + XGBoost (Hybrid)",
        save_path=cm_path
    )
    plot_roc(y_val, y_prob_hybrid, save_path=roc_path)
    print_comparison_table(hybrid_metrics)

    return {
        "lstm":   lstm_metrics,
        "hybrid": hybrid_metrics,
        "models": {"lstm": lstm_model, "xgb": xgb_model},
        "tokenizer": tokenizer
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. INFERENCE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class FakeNewsDetector:
    """
    Inference wrapper for the trained hybrid model.

    Example
    -------
    results  = run_pipeline("data/train.csv")
    detector = FakeNewsDetector(
        results["models"]["lstm"],
        results["models"]["xgb"],
        results["tokenizer"]
    )
    label, confidence = detector.predict("Breaking: Scientists discover cure...")
    print(label, f"{confidence:.1%}")   # REAL  97.3%
    """

    def __init__(self,
                 lstm_model: tf.keras.Model,
                 xgb_model:  xgb.XGBClassifier,
                 tokenizer:  Tokenizer):
        self.tokenizer = tokenizer
        self.xgb       = xgb_model
        self.extractor = build_feature_extractor(lstm_model)

    def predict(self, text: str) -> tuple:
        """
        Returns
        -------
        label      : "FAKE" or "REAL"
        confidence : probability of the predicted class  (0.0–1.0)
        """
        padded = encode(self.tokenizer, [text])
        feats  = self.extractor.predict(padded, verbose=0)
        prob   = float(self.xgb.predict_proba(feats)[0, 1])
        label  = "FAKE" if prob >= 0.5 else "REAL"
        conf   = prob if prob >= 0.5 else 1.0 - prob
        return label, conf


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    if not os.path.exists(path):
        print(f"\n[ERROR] Dataset not found at '{path}'")
        print("Download: https://www.kaggle.com/competitions/fake-news/data")
        print("Place train.csv in the data/ directory, then re-run.\n")
        sys.exit(1)
    run_pipeline(path)
