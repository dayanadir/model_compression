#!/usr/bin/env python3
"""Compute dataset-wide statistics and actionable insights for trained models."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _iter_tensors(obj: Any):
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        yield obj
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors(value)


def _compute_weight_l2_norm(weights_path: Path) -> float | None:
    try:
        import torch  # local import to keep default mode lightweight
    except Exception:
        return None

    try:
        try:
            payload = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(weights_path, map_location="cpu")
    except Exception:
        return None

    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]

    sq_sum = 0.0
    has_tensor = False
    for tensor in _iter_tensors(payload):
        try:
            sq_sum += float(tensor.detach().float().pow(2).sum().item())
            has_tensor = True
        except Exception:
            continue
    if not has_tensor:
        return None
    return math.sqrt(sq_sum)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": float(len(ordered)),
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p25": _percentile(ordered, 0.25),
        "median": _percentile(ordered, 0.50),
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.90),
        "max": ordered[-1],
    }


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = math.sqrt(den_x * den_y)
    if den == 0.0:
        return None
    return num / den


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _bucketize(value: float | None, edges: list[float], labels: list[str]) -> str | None:
    if value is None:
        return None
    for idx, edge in enumerate(edges):
        if value < edge:
            return labels[idx]
    return labels[-1]


def _bucketize_quantiles(
    value: float | None,
    *,
    q25: float,
    q50: float,
    q75: float,
    labels: tuple[str, str, str, str],
) -> str | None:
    if value is None:
        return None
    if value < q25:
        return labels[0]
    if value < q50:
        return labels[1]
    if value < q75:
        return labels[2]
    return labels[3]


def _group_metric_rows(
    records: list[dict[str, Any]],
    group_key: str,
    metric_key: str = "test_acc",
    min_count: int = 20,
) -> list[tuple[str, int, float, float, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        group = rec.get(group_key)
        metric = rec.get(metric_key)
        if group is None or metric is None:
            continue
        grouped[str(group)].append(float(metric))

    rows: list[tuple[str, int, float, float, float]] = []
    for group, values in grouped.items():
        if len(values) < min_count:
            continue
        ordered = sorted(values)
        rows.append(
            (
                group,
                len(ordered),
                statistics.fmean(ordered),
                _percentile(ordered, 0.90),
                _percentile(ordered, 0.10),
            )
        )
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows


def _print_group_metric_table(
    title: str,
    rows: list[tuple[str, int, float, float, float]],
    label_width: int = 22,
    top_n: int = 12,
) -> None:
    print(f"\n{title}")
    if not rows:
        print("  no data")
        return
    print(
        f"  {'group':{label_width}s} {'n':>6s} {'mean_acc':>10s} {'p90_acc':>10s} {'p10_acc':>10s}"
    )
    for group, n, mean_v, p90, p10 in rows[:top_n]:
        print(f"  {group:{label_width}s} {n:6d} {mean_v:10.4f} {p90:10.4f} {p10:10.4f}")


def _print_summary_block(title: str, summary: dict[str, float], value_digits: int = 4) -> None:
    print(f"\n{title}")
    if not summary:
        print("  no data")
        return
    print(
        "  n={count:.0f}, mean={mean}, std={std}, min={min}, p25={p25}, "
        "median={median}, p75={p75}, p90={p90}, max={max}".format(
            count=summary["count"],
            mean=_fmt_float(summary["mean"], value_digits),
            std=_fmt_float(summary["std"], value_digits),
            min=_fmt_float(summary["min"], value_digits),
            p25=_fmt_float(summary["p25"], value_digits),
            median=_fmt_float(summary["median"], value_digits),
            p75=_fmt_float(summary["p75"], value_digits),
            p90=_fmt_float(summary["p90"], value_digits),
            max=_fmt_float(summary["max"], value_digits),
        )
    )


def load_records(
    dataset_dir: Path,
    *,
    compute_weight_l2: bool = False,
    weight_l2_max_models: int | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    records: list[dict[str, Any]] = []
    parse_failures = 0
    weight_l2_computed = 0

    for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("model_")):
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        try:
            metadata = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            parse_failures += 1
            continue

        results = metadata.get("results", {})
        summary = metadata.get("summary", {})
        model_info = metadata.get("model_info", {})
        training = metadata.get("training", {})
        arch = metadata.get("architecture", {})

        rec = {
            "model_id": metadata.get("model_id", model_dir.name),
            "family": metadata.get("family"),
            "completed": bool(results.get("completed", False)),
            "error": results.get("error"),
            "test_acc": _to_float(results.get("test_acc")),
            "val_acc": _to_float(results.get("val_acc")),
            "train_acc": _to_float(results.get("final_train_acc")),
            "test_loss": _to_float(results.get("test_loss")),
            "wall_time_seconds": _to_float(results.get("wall_time_seconds")),
            "num_params": _to_float(summary.get("num_params") or model_info.get("num_params")),
            "optimizer": training.get("optimizer"),
            "lr": _to_float(training.get("lr")),
            "weight_decay": _to_float(training.get("weight_decay")),
            "label_smoothing": _to_float(training.get("label_smoothing")),
            "hidden_dim": _to_float(arch.get("hidden_dim")),
            "conv_layers": _to_float(arch.get("conv_layers")),
            "fc_layers": _to_float(arch.get("fc_layers")),
            "depth": _to_float(summary.get("depth")),
            "weight_l2_norm": None,
        }
        rec["lr_log10"] = math.log10(rec["lr"]) if rec.get("lr") and rec["lr"] > 0 else None
        rec["weight_decay_log10"] = (
            math.log10(rec["weight_decay"])
            if rec.get("weight_decay") and rec["weight_decay"] > 0
            else None
        )
        rec["acc_per_mparam"] = (
            rec["test_acc"] / (rec["num_params"] / 1e6)
            if rec.get("test_acc") is not None
            and rec.get("num_params") is not None
            and rec["num_params"] > 0
            else None
        )
        rec["acc_per_second"] = (
            rec["test_acc"] / rec["wall_time_seconds"]
            if rec.get("test_acc") is not None
            and rec.get("wall_time_seconds") is not None
            and rec["wall_time_seconds"] > 0
            else None
        )
        if compute_weight_l2:
            can_compute = weight_l2_max_models is None or weight_l2_computed < weight_l2_max_models
            if can_compute:
                weight_l2 = _compute_weight_l2_norm(model_dir / "weights.pt")
                rec["weight_l2_norm"] = weight_l2
                if weight_l2 is not None:
                    weight_l2_computed += 1
        rec["weight_l2_per_sqrt_param"] = (
            rec["weight_l2_norm"] / math.sqrt(rec["num_params"])
            if rec.get("weight_l2_norm") is not None
            and rec.get("num_params") is not None
            and rec["num_params"] > 0
            else None
        )
        records.append(rec)

    return records, parse_failures, weight_l2_computed


def _collect_pairs(records: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for rec in records:
        x = rec.get(x_key)
        y = rec.get(y_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def report(
    records: list[dict[str, Any]],
    parse_failures: int,
    dataset_dir: Path,
    top_k: int,
    weight_l2_computed: int,
) -> None:
    families = Counter(rec["family"] for rec in records if rec.get("family"))
    optimizers = Counter(rec["optimizer"] for rec in records if rec.get("optimizer"))

    completed_count = sum(1 for rec in records if rec.get("completed"))
    failed_count = sum(1 for rec in records if rec.get("error"))
    incomplete_count = len(records) - completed_count

    test_accs = [rec["test_acc"] for rec in records if rec.get("test_acc") is not None]
    val_accs = [rec["val_acc"] for rec in records if rec.get("val_acc") is not None]
    train_accs = [rec["train_acc"] for rec in records if rec.get("train_acc") is not None]
    params = [rec["num_params"] for rec in records if rec.get("num_params") is not None]
    wall_times = [rec["wall_time_seconds"] for rec in records if rec.get("wall_time_seconds") is not None]

    val_test_gaps = [
        rec["val_acc"] - rec["test_acc"]
        for rec in records
        if rec.get("val_acc") is not None and rec.get("test_acc") is not None
    ]
    for rec in records:
        rec["lr_bin"] = _bucketize(
            rec.get("lr"),
            edges=[3e-3, 1e-2, 3e-2],
            labels=["<3e-3", "3e-3..1e-2", "1e-2..3e-2", ">=3e-2"],
        )
        rec["weight_decay_bin"] = _bucketize(
            rec.get("weight_decay"),
            edges=[3e-5, 1e-4, 1e-3],
            labels=["<3e-5", "3e-5..1e-4", "1e-4..1e-3", ">=1e-3"],
        )
        rec["label_smoothing_bin"] = _bucketize(
            rec.get("label_smoothing"),
            edges=[0.05, 0.1, 0.15],
            labels=["<0.05", "0.05..0.1", "0.1..0.15", ">=0.15"],
        )
        if rec.get("optimizer") and rec.get("lr_bin"):
            rec["optimizer_x_lr"] = f"{rec['optimizer']} | {rec['lr_bin']}"
        else:
            rec["optimizer_x_lr"] = None
    weight_l2_vals = sorted(
        [float(rec["weight_l2_norm"]) for rec in records if rec.get("weight_l2_norm") is not None]
    )
    weight_l2_norm_vals = sorted(
        [
            float(rec["weight_l2_per_sqrt_param"])
            for rec in records
            if rec.get("weight_l2_per_sqrt_param") is not None
        ]
    )
    if len(weight_l2_vals) >= 4:
        q25, q50, q75 = (
            _percentile(weight_l2_vals, 0.25),
            _percentile(weight_l2_vals, 0.50),
            _percentile(weight_l2_vals, 0.75),
        )
        for rec in records:
            rec["weight_l2_bin"] = _bucketize_quantiles(
                rec.get("weight_l2_norm"),
                q25=q25,
                q50=q50,
                q75=q75,
                labels=("Q1(low)", "Q2", "Q3", "Q4(high)"),
            )
    else:
        for rec in records:
            rec["weight_l2_bin"] = None
    if len(weight_l2_norm_vals) >= 4:
        q25, q50, q75 = (
            _percentile(weight_l2_norm_vals, 0.25),
            _percentile(weight_l2_norm_vals, 0.50),
            _percentile(weight_l2_norm_vals, 0.75),
        )
        for rec in records:
            rec["weight_l2_per_sqrt_param_bin"] = _bucketize_quantiles(
                rec.get("weight_l2_per_sqrt_param"),
                q25=q25,
                q50=q50,
                q75=q75,
                labels=("Q1(low)", "Q2", "Q3", "Q4(high)"),
            )
    else:
        for rec in records:
            rec["weight_l2_per_sqrt_param_bin"] = None

    print(f"Dataset: {dataset_dir}")
    print(f"Loaded metadata records: {len(records)}")
    print(f"Completed models: {completed_count}")
    print(f"Incomplete models: {incomplete_count}")
    print(f"Failed models (error field set): {failed_count}")
    print(f"Metadata parse failures: {parse_failures}")
    print(f"Weight L2 norms computed: {weight_l2_computed}")

    print("\nFamily distribution")
    for family, count in families.most_common():
        pct = 100.0 * count / max(1, len(records))
        print(f"  {family:10s} {count:6d} ({pct:5.2f}%)")

    print("\nOptimizer usage")
    for optimizer, count in optimizers.most_common():
        pct = 100.0 * count / max(1, len(records))
        print(f"  {optimizer:10s} {count:6d} ({pct:5.2f}%)")

    _print_summary_block("Test accuracy", _numeric_summary(test_accs), value_digits=4)
    _print_summary_block("Validation accuracy", _numeric_summary(val_accs), value_digits=4)
    _print_summary_block("Train accuracy", _numeric_summary(train_accs), value_digits=4)
    _print_summary_block("Validation - test gap", _numeric_summary(val_test_gaps), value_digits=4)
    _print_summary_block("Num params", _numeric_summary(params), value_digits=0)
    _print_summary_block("Wall time (seconds)", _numeric_summary(wall_times), value_digits=2)
    _print_summary_block(
        "Accuracy per million params (test_acc / M params)",
        _numeric_summary([rec["acc_per_mparam"] for rec in records if rec.get("acc_per_mparam") is not None]),
        value_digits=4,
    )
    _print_summary_block(
        "Accuracy per second (test_acc / sec)",
        _numeric_summary([rec["acc_per_second"] for rec in records if rec.get("acc_per_second") is not None]),
        value_digits=6,
    )
    _print_summary_block(
        "Weight L2 norm",
        _numeric_summary([rec["weight_l2_norm"] for rec in records if rec.get("weight_l2_norm") is not None]),
        value_digits=4,
    )
    _print_summary_block(
        "Weight L2 normalized (L2 / sqrt(num_params))",
        _numeric_summary(
            [
                rec["weight_l2_per_sqrt_param"]
                for rec in records
                if rec.get("weight_l2_per_sqrt_param") is not None
            ]
        ),
        value_digits=4,
    )

    print("\nPer-family performance")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        if rec.get("family"):
            by_family[str(rec["family"])].append(rec)

    family_rows: list[tuple[str, int, float, float, float]] = []
    for family, group in sorted(by_family.items()):
        family_test_acc = [r["test_acc"] for r in group if r.get("test_acc") is not None]
        family_params = [r["num_params"] for r in group if r.get("num_params") is not None]
        if not family_test_acc:
            continue
        mean_acc = statistics.fmean(family_test_acc)
        p90_acc = _percentile(sorted(family_test_acc), 0.90)
        mean_params = statistics.fmean(family_params) if family_params else float("nan")
        family_rows.append((family, len(group), mean_acc, p90_acc, mean_params))

    family_rows.sort(key=lambda row: row[2], reverse=True)
    print("  family      n      mean_test_acc   p90_test_acc    mean_params")
    for family, count, mean_acc, p90_acc, mean_params in family_rows:
        print(
            f"  {family:10s} {count:6d}      {mean_acc:8.4f}      {p90_acc:8.4f}    {mean_params:12.0f}"
        )

    _print_group_metric_table(
        "Breakdown by optimizer (test accuracy)",
        _group_metric_rows(records, "optimizer", metric_key="test_acc", min_count=30),
        label_width=14,
    )
    _print_group_metric_table(
        "Breakdown by learning-rate bins (test accuracy)",
        _group_metric_rows(records, "lr_bin", metric_key="test_acc", min_count=30),
        label_width=14,
    )
    _print_group_metric_table(
        "Breakdown by weight-decay bins (test accuracy)",
        _group_metric_rows(records, "weight_decay_bin", metric_key="test_acc", min_count=30),
        label_width=16,
    )
    _print_group_metric_table(
        "Breakdown by label-smoothing bins (test accuracy)",
        _group_metric_rows(records, "label_smoothing_bin", metric_key="test_acc", min_count=30),
        label_width=16,
    )
    _print_group_metric_table(
        "Interaction: optimizer x LR bin (test accuracy)",
        _group_metric_rows(records, "optimizer_x_lr", metric_key="test_acc", min_count=25),
        label_width=22,
        top_n=16,
    )
    _print_group_metric_table(
        "Breakdown by weight L2 quartiles (test accuracy)",
        _group_metric_rows(records, "weight_l2_bin", metric_key="test_acc", min_count=5),
        label_width=14,
    )
    _print_group_metric_table(
        "Breakdown by weight L2/sqrt(params) quartiles (test accuracy)",
        _group_metric_rows(records, "weight_l2_per_sqrt_param_bin", metric_key="test_acc", min_count=5),
        label_width=22,
    )

    xs, ys = _collect_pairs(records, "num_params", "test_acc")
    corr_params_acc = _pearson_corr(xs, ys)
    xs, ys = _collect_pairs(records, "num_params", "wall_time_seconds")
    corr_params_time = _pearson_corr(xs, ys)

    print("\nCorrelations (Pearson)")
    print(f"  corr(num_params, test_acc)      = {_fmt_float(corr_params_acc, 4)}")
    print(f"  corr(num_params, wall_time_sec) = {_fmt_float(corr_params_time, 4)}")
    numeric_feature_keys = [
        "num_params",
        "wall_time_seconds",
        "lr",
        "lr_log10",
        "weight_decay",
        "weight_decay_log10",
        "label_smoothing",
        "hidden_dim",
        "conv_layers",
        "fc_layers",
        "depth",
        "weight_l2_norm",
        "weight_l2_per_sqrt_param",
    ]
    feature_corr_rows: list[tuple[str, float, int]] = []
    for key in numeric_feature_keys:
        xs, ys = _collect_pairs(records, key, "test_acc")
        corr = _pearson_corr(xs, ys)
        if corr is not None:
            feature_corr_rows.append((key, corr, len(xs)))
    feature_corr_rows.sort(key=lambda row: abs(row[1]), reverse=True)
    print("\nFeature signal ranking vs test_acc (Pearson, abs sorted)")
    for key, corr, n in feature_corr_rows:
        print(f"  {key:18s} corr={corr:8.4f} (n={n})")

    ranked = [rec for rec in records if rec.get("test_acc") is not None]
    ranked.sort(key=lambda rec: float(rec["test_acc"]), reverse=True)

    print(f"\nTop {top_k} models by test accuracy")
    for rec in ranked[:top_k]:
        print(
            f"  {rec['model_id']} | family={rec.get('family')} | "
            f"test_acc={_fmt_float(rec.get('test_acc'), 4)} | "
            f"val_acc={_fmt_float(rec.get('val_acc'), 4)} | "
            f"params={_fmt_float(rec.get('num_params'), 0)} | "
            f"optimizer={rec.get('optimizer')}"
        )

    print(f"\nBottom {top_k} models by test accuracy")
    for rec in ranked[-top_k:]:
        print(
            f"  {rec['model_id']} | family={rec.get('family')} | "
            f"test_acc={_fmt_float(rec.get('test_acc'), 4)} | "
            f"val_acc={_fmt_float(rec.get('val_acc'), 4)} | "
            f"params={_fmt_float(rec.get('num_params'), 0)} | "
            f"optimizer={rec.get('optimizer')}"
        )

    ranked_eff_params = [rec for rec in records if rec.get("acc_per_mparam") is not None]
    ranked_eff_params.sort(key=lambda rec: float(rec["acc_per_mparam"]), reverse=True)
    print(f"\nTop {top_k} models by parameter efficiency (test_acc / M params)")
    for rec in ranked_eff_params[:top_k]:
        print(
            f"  {rec['model_id']} | family={rec.get('family')} | "
            f"eff={_fmt_float(rec.get('acc_per_mparam'), 4)} | "
            f"test_acc={_fmt_float(rec.get('test_acc'), 4)} | "
            f"params={_fmt_float(rec.get('num_params'), 0)}"
        )

    ranked_eff_time = [rec for rec in records if rec.get("acc_per_second") is not None]
    ranked_eff_time.sort(key=lambda rec: float(rec["acc_per_second"]), reverse=True)
    print(f"\nTop {top_k} models by time efficiency (test_acc / sec)")
    for rec in ranked_eff_time[:top_k]:
        print(
            f"  {rec['model_id']} | family={rec.get('family')} | "
            f"eff={_fmt_float(rec.get('acc_per_second'), 6)} | "
            f"test_acc={_fmt_float(rec.get('test_acc'), 4)} | "
            f"time={_fmt_float(rec.get('wall_time_seconds'), 2)}s"
        )

    print("\nInsights")
    if family_rows:
        best_family = family_rows[0]
        print(
            f"  Best mean test accuracy family: {best_family[0]} "
            f"(mean={best_family[2]:.4f}, p90={best_family[3]:.4f}, n={best_family[1]})"
        )
    if corr_params_acc is not None:
        trend = "positive" if corr_params_acc > 0 else "negative"
        print(
            f"  Params vs test accuracy trend is {trend} "
            f"(corr={corr_params_acc:.4f}); magnitude indicates effect strength."
        )
    if corr_params_time is not None:
        trend = "positive" if corr_params_time > 0 else "negative"
        print(
            f"  Params vs wall-time trend is {trend} "
            f"(corr={corr_params_time:.4f}); higher usually means larger models train slower."
        )
    optimizer_rows = _group_metric_rows(records, "optimizer", metric_key="test_acc", min_count=30)
    if optimizer_rows:
        print(
            f"  Best optimizer by mean test_acc: {optimizer_rows[0][0]} "
            f"(mean={optimizer_rows[0][2]:.4f}, n={optimizer_rows[0][1]})."
        )
    lr_rows = _group_metric_rows(records, "lr_bin", metric_key="test_acc", min_count=30)
    if lr_rows:
        print(
            f"  Best LR bin by mean test_acc: {lr_rows[0][0]} "
            f"(mean={lr_rows[0][2]:.4f}, n={lr_rows[0][1]})."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute statistics and insights for a model-zoo dataset directory."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/adir.dayan/model_compression/dataset"),
        help="Path containing model_*/metadata.json files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top/bottom models to print.",
    )
    parser.add_argument(
        "--compute-weight-l2",
        action="store_true",
        help="Load each weights.pt and compute model weight L2 norm.",
    )
    parser.add_argument(
        "--weight-l2-max-models",
        type=int,
        default=None,
        help="Optional cap on number of models to compute L2 for (for quick runs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    records, parse_failures, weight_l2_computed = load_records(
        dataset_dir,
        compute_weight_l2=bool(args.compute_weight_l2),
        weight_l2_max_models=args.weight_l2_max_models,
    )
    if not records:
        raise SystemExit(f"No readable metadata.json files found under: {dataset_dir}")
    report(
        records,
        parse_failures,
        dataset_dir,
        top_k=max(1, int(args.top_k)),
        weight_l2_computed=weight_l2_computed,
    )


if __name__ == "__main__":
    main()
