#!/usr/bin/env python3
"""
amazon_logreg_pipeline.py
Refactored ETL + TF-IDF + LogisticRegression pipeline with threshold analysis,
artifact saving (model + vectorizer + metrics + plots) and CLI.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split

# === File paths / config ===
INPUT_CSV = "Git_folder\\Amazon\\amazon.csv" # Delete "Git_folder\\" If the code does not run
DEFAULT_OUTDIR = "Git_folder\\Amazon\\artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# === 0. Config / Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# === 1. Load ===
def load_data(path: str) -> pd.DataFrame:
    """1. Load CSV into DataFrame (basic checks)."""
    df = pd.read_csv(path)
    logger.info("Loaded data: rows=%d cols=%d", df.shape[0], df.shape[1])
    return df


# === 2. Preprocess ===
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """2. Preprocess DataFrame: handle missing, binarize rating, pick text col."""
    df = df.copy()

    # 2.1 Handle missing values for rating (convert to numeric, drop rows without rating)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()

    # 2.2 Binarize sentiment: 1 if rating >= 4, else 0
    df.loc[:, "sentiment"] = (df["rating"] >= 4).astype(int)

    # 2.3 Select text column: prefer 'review_content' else fallback to 'product_name'
    TEXT_COL = "review_content" if "review_content" in df.columns else "product_name"
    df.loc[:, "text"] = df[TEXT_COL].astype(str).str.lower().str.strip()

    # 2.4 Remove rows with empty text
    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    logger.info("Dropped %d rows with empty text", before - len(df))

    # 2.5 Check class distribution
    vc = df["sentiment"].value_counts()
    logger.info("Class distribution after binarization: %s", vc.to_dict())

    return df


# === 3. Train / Evaluate ===
def train_and_evaluate(
    df: pd.DataFrame,
    output_dir: Path,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    do_gridsearch: bool = True,
):
    """3. Train TF-IDF + LogisticRegression and evaluate; save artifacts."""
    # 3.1 Prepare X, y
    texts = df["text"].values
    labels = df["sentiment"].values

    # 3.2 Split with stratify if possible
    vc = pd.Series(labels).value_counts()
    use_stratify = True
    if vc.min() < 2:
        logger.warning("Some classes have <2 samples; splitting without stratify")
        use_stratify = False

    if use_stratify:
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
    else:
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=None
        )

    # 3.3 TF-IDF vectorizer (fit on train only)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english", min_df=2)
    Xtr = vec.fit_transform(X_train_texts)
    Xte = vec.transform(X_test_texts)

    # 3.4 Baseline LogisticRegression with class_weight balanced
    base_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)

    # 3.5 GridSearch over C (regularization) - optional
    if do_gridsearch:
        param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
        # Try GridSearch with parallel jobs; if environment lacks posix subprocess
        # (e.g. some Windows MS-store Python builds), fallback to n_jobs=1 or no grid.
        try:
            grid = GridSearchCV(base_clf, param_grid, cv=3, scoring="f1", n_jobs=-1)
            grid.fit(Xtr, y_train)
            clf = grid.best_estimator_
            logger.info("GridSearch best params: %s", grid.best_params_)
        except ModuleNotFoundError as e:
            # Common on some restricted Windows Python builds: _posixsubprocess missing.
            logger.warning("GridSearchCV failed with ModuleNotFoundError: %s", e)
            logger.warning("Falling back to GridSearchCV with n_jobs=1 (no parallelism).")
            try:
                grid = GridSearchCV(base_clf, param_grid, cv=3, scoring="f1", n_jobs=1)
                grid.fit(Xtr, y_train)
                clf = grid.best_estimator_
                logger.info("GridSearch best params (n_jobs=1): %s", grid.best_params_)
            except Exception as e2:
                logger.warning("GridSearch (n_jobs=1) also failed: %s. Falling back to baseline fit.", e2)
                clf = base_clf
                clf.fit(Xtr, y_train)
    else:
        clf = base_clf
        clf.fit(Xtr, y_train)

    # 3.6 Predictions and probabilities
    y_pred = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]

    # 3.7 Standard metrics (threshold 0.5)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    logger.info("Baseline metrics (threshold=0.5): %s", metrics)
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    # 3.8 Save artifacts: vectorizer and model
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, output_dir / "tfidf_vectorizer.pkl")
    joblib.dump(clf, output_dir / "logreg_model.pkl")
    logger.info("Saved vectorizer and model to %s", output_dir)

    # 3.9 Compute threshold analysis and plots
    thresholds_results = find_thresholds_and_plot(y_test, y_proba, output_dir)

    # 3.10 Save metrics, test split sample and misclassified examples
    results = {
        "baseline_metrics": metrics,
        "thresholds_summary": thresholds_results["summary"],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf8") as f:
        json.dump(results, f, indent=2)

    mis_idx = np.where(y_test != (y_proba >= thresholds_results["best_threshold"]))[0]
    mis_df = pd.DataFrame(
        {
            "text": X_test_texts[mis_idx],
            "y_true": y_test[mis_idx],
            "y_proba": y_proba[mis_idx],
            "y_pred_threshold": (y_proba[mis_idx] >= thresholds_results["best_threshold"]).astype(int),
        }
    )
    mis_df.to_csv(output_dir / "misclassified_examples.csv", index=False)

    logger.info("Saved metrics.json and misclassified_examples.csv")
    return {
        "model": clf,
        "vectorizer": vec,
        "y_test": y_test,
        "y_proba": y_proba,
        "best_threshold": thresholds_results["best_threshold"],
        "metrics": metrics,
    }


# === 4. Threshold analysis & plotting ===
def find_thresholds_and_plot(y_true: np.ndarray, y_proba: np.ndarray, output_dir: Path) -> dict:
    """4. Compute metrics across thresholds, plot PR/ROC and return best threshold choices."""
    # 4.1 Compute precision/recall curve and F1 per threshold
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)

    # 4.2 Prepare threshold array aligned with pr_thresholds (precision_recall_curve gives n+1 precision/recalls)
    ths = np.concatenate(([0.0], pr_thresholds, [1.0]))
    best_idx = np.nanargmax(f1s)
    best_threshold = ths[best_idx]
    best_f1 = float(f1s[best_idx])
    best_prec = float(precisions[best_idx])
    best_rec = float(recalls[best_idx])

    # 4.3 Compute recall for negative (class 0) for each threshold
    neg_recalls = []
    for t in ths:
        preds = (y_proba >= t).astype(int)
        neg_recalls.append(recall_score(1 - y_true, 1 - preds))

    # 4.4 Find threshold with neg_recalls >= 0.45 and best f1 among them
    candidates = [
        (t,
         f1s[i] if i < len(f1s) else 0,
         precisions[i] if i < len(precisions) else 0,
         recalls[i] if i < len(recalls) else 0,
         neg_recalls[i])
        for i, t in enumerate(ths)
    ]
    candidates_meet = [c for c in candidates if c[4] >= 0.45]
    best_candidate = None
    if candidates_meet:
        best_candidate = sorted(candidates_meet, key=lambda x: -x[1])[0]

    # 4.5 ROC curve for plot
    fpr, tpr, roc_th = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # 4.6 Plot PR and ROC and save
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recalls, precisions, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.grid(True)
    plt.scatter(best_rec, best_prec, color="red", label=f"best F1={best_f1:.3f} @ {best_threshold:.2f}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.grid(True)
    plt.legend()

    out_plot = output_dir / "pr_roc_plot.png"
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()
    logger.info("Saved PR+ROC plot to %s", out_plot)

    summary = {
        "best_f1_threshold": float(best_threshold),
        "best_f1": best_f1,
        "best_precision": best_prec,
        "best_recall": best_rec,
        "roc_auc": float(roc_auc),
    }
    if best_candidate:
        summary["best_candidate_with_negrec>=0.45"] = {
            "threshold": float(best_candidate[0]),
            "f1": float(best_candidate[1]),
            "precision": float(best_candidate[2]),
            "recall_pos": float(best_candidate[3]),
            "recall_neg": float(best_candidate[4]),
        }

    return {"summary": summary, "best_threshold": float(best_threshold)}


# === 5. CLI ===
def parse_args():
    p = argparse.ArgumentParser(description="Amazon reviews TF-IDF + LogisticRegression pipeline")
    p.add_argument("--input", type=str, default=INPUT_CSV, help="Path to amazon csv")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output folder for artifacts")
    p.add_argument("--test-size", type=float, default=TEST_SIZE, help="Test set fraction")
    p.add_argument("--no-grid", action="store_true", help="Disable GridSearch and use baseline hyperparams")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    outdir = Path(args.outdir)

    df = load_data(input_path)
    df_clean = preprocess(df)
    res = train_and_evaluate(df_clean, outdir, test_size=args.test_size, do_gridsearch=not args.no_grid)
    logger.info("Best threshold chosen: %s", res["best_threshold"])


if __name__ == "__main__":
    main()
