#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/model.joblib")
    p.add_argument("--request_json", type=str, default="models/example_request.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = joblib.load(args.model_path)
    model = payload["model"]
    feature_names = payload["feature_names"]

    req = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    feats = req["features"]

    x = np.array([[float(feats[name]) for name in feature_names]], dtype=float)
    proba = float(model.predict_proba(x)[0, 1])
    print(json.dumps({"probability_class1": proba}, indent=2))


if __name__ == "__main__":
    main()
