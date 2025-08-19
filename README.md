# Amazon Reviews — TF-IDF + Logistic Regression Pipeline

## Overview

This repository contains a compact, production-minded pipeline for binary sentiment classification on Amazon reviews. It combines a small ETL step, TF‑IDF text vectorization and a `LogisticRegression` classifier with optional hyperparameter search, threshold analysis and artifact saving (model, vectorizer, metrics and plots). The README is written in a professional tone but in my style — concise, practical and focused on what actually matters.

---

![License](https://img.shields.io/badge/license-CC0%201.0-lightgrey)

---

## Key features

* Simple, reproducible CLI workflow (`--input`, `--outdir`, `--test-size`, `--no-grid`).
* Basic data cleaning and label binarization (rating → sentiment).
* TF‑IDF vectorizer (unigrams + bigrams) fit on train split only.
* `LogisticRegression` with optional `GridSearchCV` over `C` (fallback to safe behaviour if GridSearch fails).
* Threshold analysis: finds best F1 threshold and reports candidate thresholds that respect a negative-class recall constraint (>= 0.45).
* Saves useful artifacts: trained model, vectorizer, PR/ROC plot, JSON metrics, and CSV of misclassified examples.
* Robust logging and deterministic behaviour via `RANDOM_STATE`.

---

## Quickstart

### Requirements

```bash
python >= 3.8
pip install -r requirements.txt
# or
pip install pandas numpy scikit-learn matplotlib joblib
```

### Run (default)

```bash
python amazon_logreg_pipeline.py
```

### Example with custom input and without GridSearch

```bash
python amazon_logreg_pipeline.py --input data/amazon.csv --outdir artifacts --no-grid
```

---

## CLI arguments

| Flag          | Type        | Default                        | Description                                         |
| ------------- | ----------- | ------------------------------ | --------------------------------------------------- |
| `--input`     | str         | `Git_folder\Amazon\amazon.csv` | Path to input CSV with Amazon reviews               |
| `--outdir`    | str         | `Git_folder\Amazon\artifacts`  | Folder where artifacts will be written              |
| `--test-size` | float       | `0.2`                          | Fraction of data to use as test set                 |
| `--no-grid`   | bool (flag) | `False`                        | Disable `GridSearchCV` and use baseline hyperparams |

---

## Output artifacts

After a run, you will find the following files in the `outdir`:

| File                         | Description                                                        |
| ---------------------------- | ------------------------------------------------------------------ |
| `tfidf_vectorizer.pkl`       | `joblib` dump of the trained `TfidfVectorizer`                     |
| `logreg_model.pkl`           | `joblib` dump of the trained `LogisticRegression` model            |
| `pr_roc_plot.png`            | Combined Precision‑Recall and ROC plot                             |
| `metrics.json`               | JSON summary with baseline metrics and threshold analysis summary  |
| `misclassified_examples.csv` | Small CSV with test examples misclassified at the chosen threshold |

> **Note:** the `artifacts/` folder (models, plots, CSVs) is listed in `.gitignore` and is not committed.

---

## How it works (short)

1. **Load** — reads input CSV into a `pandas.DataFrame` and logs shape.
2. **Preprocess** — coerces `rating` to numeric, drops rows without rating, binarizes sentiment (`rating >= 4 → 1`), chooses `review_content` (fallback: `product_name`) as text and drops empty text rows.
3. **Train / Evaluate**

   * Train/test split (stratified if possible).
   * Fit `TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words='english', min_df=2)` on train only.
   * Fit `LogisticRegression(max_iter=2000, class_weight='balanced')` with optional `GridSearchCV` over `C`.
   * Compute baseline metrics (accuracy, precision, recall, f1, roc\_auc) at threshold 0.5 and save vectorizer + model.
4. **Threshold analysis** — compute precision/recall curve, F1 per threshold, find threshold of best F1, compute negative-class recall for each threshold and prefer candidates with `recall_neg >= 0.45` when available. Save PR/ROC plot and JSON summary.
5. **Artifacts** — save model, vectorizer, metrics.json and misclassified examples.

---

## Inference / Predict

A small `predict.py` script is included to quickly load artifacts and predict on one or more texts.

Example:

```bash
python predict.py "This product is great, works as advertised"
# -> {"proba": 0.92, "pred": 1}
```

By default `predict.py` loads models from `artifacts/logreg_model.pkl` and `artifacts/tfidf_vectorizer.pkl`. You can also pass custom paths with CLI arguments.

---

## Development

Install dev dependencies (linters, test runner):

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

Formatting / linting:

```bash
black .
flake8 .
```

---

## Testing & CI

Unit tests live in `tests/`. Example test run locally:

```bash
pytest -q
```

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs tests and linters on each push/PR (see badge at the top).

---

## Docker

Build image:

```bash
docker build -t amazon-logreg:latest .
```

Run (example — will execute the pipeline inside container):

```bash
docker run --rm -v $(pwd)/artifacts:/app/artifacts amazon-logreg:latest
```

Run inference (example):

```bash
docker run --rm amazon-logreg:latest python predict.py "nice product"
```

---

## What not to commit

Local artifacts and environment files are ignored via `.gitignore`. Typical entries: `artifacts/`, `*.pkl`, `*.joblib`, `venv/`, `__pycache__/`.

---

## License

This project is released under **CC0 1.0 Universal** — public domain. See `LICENSE` for full text.

---

## Author

**Andriy Vlonha** (nickname: *Western*)
GitHub: [github.com/western](https://github.com/western-1)