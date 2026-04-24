"""Aggregate compression result files into per-family statistics."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(mean(arr)),
        "std": float(stdev(arr)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "q3": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def aggregate_results(dataset_root: str) -> dict:
    root = Path(dataset_root)
    records = []
    for model_dir in sorted(root.glob("model_*")):
        result_path = model_dir / "compression_result.json"
        if not result_path.exists():
            continue
        with result_path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))

    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for rec in records:
        family = rec["family"]
        for method, payload in rec.get("methods", {}).items():
            key = (family, method)
            grouped.setdefault(key, {"acc": [], "params": []})
            grouped[key]["acc"].append(float(payload["test_acc"]))
            grouped[key]["params"].append(float(rec["params_reduction_pct"]))

    out = {"num_models": len(records), "families": {}}
    for (family, method), vals in grouped.items():
        out["families"].setdefault(family, {})
        out["families"][family][method] = {
            "test_acc": _stats(vals["acc"]),
            "params_reduction_pct": _stats(vals["params"]),
        }
    return out
