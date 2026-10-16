"""Training script with MLflow tracking.

Run:
    python -m ml.train --data-version v1 --experiment baseline
    python -m ml.train --data-version v2 --experiment improved

Logs:
  - params: data_version, ngram_range, max_features, C, etc.
  - metrics: accuracy, f1, precision, recall
  - tags: git_commit, data_dvc_hash
  - artifact: trained sklearn pipeline (joblib)
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = ROOT / "ml" / "registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _data_hash(data_dir: Path) -> str:
    """Cheap fingerprint: sorted (filename, sha256-of-file) joined."""
    import hashlib

    out = []
    for p in sorted(data_dir.glob("*.csv")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        out.append(f"{p.name}:{h}")
    return ",".join(out)


def train(data_version: str, experiment: str, c: float, ngram_max: int, max_features: int) -> None:
    data_dir = ROOT / "data" / "raw" / data_version
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # Drop rows where cleaned text became empty (v2 may have some)
    train_df = train_df.dropna(subset=["text"])
    test_df = test_df.dropna(subset=["text"])
    train_df = train_df[train_df["text"].str.strip().astype(bool)]
    test_df = test_df[test_df["text"].str.strip().astype(bool)]

    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]

    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=f"{data_version}-{experiment}"):
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features, stop_words="english")),
                ("clf", LogisticRegression(C=c, max_iter=1000)),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
        }

        mlflow.log_params(
            {
                "data_version": data_version,
                "ngram_max": ngram_max,
                "max_features": max_features,
                "C": c,
                "model": "LogisticRegression",
                "vectorizer": "TF-IDF",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tag("git_commit", _git_sha())
        mlflow.set_tag("data_dvc_hash", _data_hash(data_dir))

        # Save the pipeline both as MLflow artifact and to the registry folder
        # (the registry copy is what the FastAPI app loads at runtime).
        model_path = REGISTRY_DIR / "model.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"[{experiment}/{data_version}] metrics={metrics}")
        print(f"model saved -> {model_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-version", choices=["v1", "v2"], required=True)
    p.add_argument("--experiment", default="baseline")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--ngram-max", type=int, default=1)
    p.add_argument("--max-features", type=int, default=20000)
    args = p.parse_args()
    train(args.data_version, args.experiment, args.C, args.ngram_max, args.max_features)


if __name__ == "__main__":
    main()
