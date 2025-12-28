#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Dict

import joblib
import numpy as np
from flask import Flask, request, jsonify


MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

app = Flask(__name__)

_MODEL = None
_FEATURES = None


def _load() -> None:
    global _MODEL, _FEATURES
    payload = joblib.load(MODEL_PATH)
    _MODEL = payload["model"]
    _FEATURES = payload["feature_names"]


def _vectorize(features: Dict[str, Any]) -> np.ndarray:
    x = np.zeros((1, len(_FEATURES)), dtype=float)
    for i, name in enumerate(_FEATURES):
        if name not in features:
            raise KeyError(f"Missing feature: {name}")
        x[0, i] = float(features[name])
    return x


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "model_path": MODEL_PATH})


@app.post("/predict")
def predict() -> Any:
    if _MODEL is None:
        _load()

    data = request.get_json(force=True, silent=False)
    if not isinstance(data, dict) or "features" not in data:
        return jsonify({"error": "Expected JSON body with key 'features' (dict)"}), 400

    features = data["features"]
    if not isinstance(features, dict):
        return jsonify({"error": "'features' must be a JSON object/dict"}), 400

    try:
        x = _vectorize(features)
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Bad feature values: {e}"}), 400

    proba = float(_MODEL.predict_proba(x)[0, 1])
    return jsonify({"probability_class1": proba})


if __name__ == "__main__":
    _load()
    app.run(host="0.0.0.0", port=9696, debug=False)
