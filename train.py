#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/model.joblib")
    p.add_argument("--example_path", type=str, default="models/example_request.json")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=list(data.feature_names))
    y = pd.Series(data.target, name="target").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=args.C, max_iter=2000, n_jobs=1)),
        ]
    )

    pipeline.fit(X_train.values, y_train.values)

    proba = pipeline.predict_proba(X_test.values)[:, 1]
    auc = roc_auc_score(y_test.values, proba)

    model_payload = {
        "model": pipeline,
        "feature_names": list(X.columns),
        "meta": {
            "dataset": "sklearn.datasets.load_breast_cancer (offline)",
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "C": float(args.C),
        },
    }

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, model_path)

    example_row = X_test.iloc[0].to_dict()
    example_req = {"features": {k: float(v) for k, v in example_row.items()}}
    example_path = Path(args.example_path)
    example_path.parent.mkdir(parents=True, exist_ok=True)
    example_path.write_text(json.dumps(example_req, indent=2), encoding="utf-8")

    print(f"AUC={auc:.6f}")
    print(f"Saved model -> {model_path}")
    print(f"Saved example request -> {example_path}")


if __name__ == "__main__":
    main()
