# FakeNewsDetector-LSTM-XGBoost

**Enhancing Knowledge Management Integrity through Fake News Detection: A Hybrid LSTM-XGBoost Approach for Cybersecurity**

> MV Sujan Kumar, Ganesh Khekare, Anurup Sankriti  
> School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India

[![DOI](https://img.shields.io/badge/DOI-10.1201%2F9781003498094--9-blue)](https://doi.org/10.1201/9781003498094-9)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-Academic%20Citation%20Required-red)](LICENSE)

> **Published in:** *Handbook of Research on Cybersecurity Issues and Challenges for Business and FinTech Applications*, IGI Global / CRC Press, 2024.  
> DOI: [10.1201/9781003498094-9](https://doi.org/10.1201/9781003498094-9)

---

## Overview

This repository contains the full implementation of the hybrid LSTM-XGBoost model for fake news detection described in the book chapter above. The approach fuses:

- **Bidirectional LSTM** — captures sequential and contextual patterns in news text from both directions
- **XGBoost** — uses the LSTM's learned feature representations as input, adding gradient-boosted decision tree classification on top

The result is a two-stage pipeline where the deep learning model acts as a feature extractor and XGBoost acts as the final classifier — combining the representational power of recurrent networks with the robustness of ensemble methods.

**Best model: LSTM + XGBoost — 94.04% accuracy, AUC = 0.98**

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|:---:|:---:|:---:|:---:|
| Naive Bayes | 71.47% | 99.00% | 33.90% | 50.59% |
| Random Forest | 90.81% | 95.99% | 82.09% | 88.50% |
| CNN | 91.60% | 90.48% | 89.96% | 90.22% |
| **LSTM + XGBoost (ours)** | **94.04%** | **93.74%** | **92.14%** | **92.93%** |

AUC = 0.98 on the ROC curve.

---

## Pipeline

```
Raw news text
  └─► Tokenize & Pad  (maxlen=300, vocab=50k)
        └─► Embedding Layer  (dim=128)
              └─► BiLSTM  (256 units, return_sequences=True)
                    └─► Dropout (0.3)
                          └─► BiLSTM  (128 units)
                                └─► Dropout (0.3)
                                      └─► Dense (64, ReLU)   ← feature vector
                                            └─► XGBoost ──► FAKE / REAL
```

---

## Repository Structure

```
FakeNewsDetector-LSTM-XGBoost/
│
├── model/
│   └── lstm_xgboost.py      # Full pipeline — load, train, evaluate, infer
│
├── notebooks/
│   └── demo.ipynb           # End-to-end walkthrough on Kaggle dataset
│
├── docs/
│   └── architecture.md      # Pipeline diagram and design notes
│
├── data/
│   └── .gitkeep             # Place train.csv here (not tracked)
│
├── figures/                 # Auto-generated: confusion matrix, ROC curve
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Dataset

This implementation uses the **Kaggle Fake News Dataset** — publicly available, no restrictions.

**Download:** [kaggle.com/competitions/fake-news/data](https://www.kaggle.com/competitions/fake-news/data)

Place `train.csv` in the `data/` directory before running.

Dataset split used in the paper: 80% train / 20% validation (stratified).

| Split | Rows | Columns |
|-------|------|---------|
| Train | 20,800 | 5 |
| Test | 5,200 | 4 |

---

## Installation

```bash
git clone https://github.com/KRYSTALM7/FakeNewsDetector-LSTM-XGBoost.git
cd FakeNewsDetector-LSTM-XGBoost
pip install -r requirements.txt
```

**Run the full pipeline:**
```bash
python model/lstm_xgboost.py data/train.csv
```

**Or use the notebook:**
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Quick Inference

```python
from model.lstm_xgboost import run_pipeline, FakeNewsDetector

# Train
results  = run_pipeline("data/train.csv")

# Infer on new text
detector = FakeNewsDetector(
    results["models"]["lstm"],
    results["models"]["xgb"],
    results["tokenizer"]
)

label, confidence = detector.predict(
    "Scientists announce major breakthrough in renewable energy storage..."
)
print(label, f"{confidence:.1%}")   # REAL  96.2%
```

---

## Citation

If you use this code, please cite:

```bibtex
@incollection{sujankumar2024fakenews,
  title     = {Enhancing Knowledge Management Integrity through Fake News Detection:
               A Hybrid LSTM-XGBoost Approach for Cybersecurity},
  author    = {MV Sujan Kumar and Ganesh Khekare and Anurup Sankriti},
  booktitle = {Handbook of Research on Cybersecurity Issues and Challenges
               for Business and FinTech Applications},
  publisher = {IGI Global / CRC Press},
  year      = {2024},
  doi       = {10.1201/9781003498094-9},
  url       = {https://doi.org/10.1201/9781003498094-9}
}
```
## Authors

**MV Sujan Kumar** — [sujankumar7702@gmail.com](mailto:sujankumar7702@gmail.com) | [GitHub @KRYSTALM7](https://github.com/KRYSTALM7)

**Ganesh Khekare** — [ganesh.khekare@vit.ac.in](mailto:ganesh.khekare@vit.ac.in)

**Anurup Sankriti** — [anurupsankriti101@gmail.com](mailto:anurupsankriti101@gmail.com) | [GitHub @Anurup-Sankriti](https://github.com/Anurup-Sankriti)

---

## License

Academic Use License — see [LICENSE](LICENSE) for full terms. Citation required for any use in academic work.
