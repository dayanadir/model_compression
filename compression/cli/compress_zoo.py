"""Run compression baselines across the dataset zoo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from compression.cli.common import resolve_methods
from compression.cli.compress_one import run_one_model
from compression.eval.report import aggregate_results


def _run_job(job: dict) -> tuple[str, dict]:
    result = run_one_model(
        job["model_dir"],
        methods=job["methods"],
        seed=job["seed"],
        calib_seed=job["calib_seed"],
        calib_n=job["calib_n"],
        calib_batch_size=job["calib_batch_size"],
        device=job["device"],
        data_dir=job["data_dir"],
    )
    return job["model_dir"], result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=0)
    parser.add_argument("--calib-n", type=int, default=512)
    parser.add_argument("--calib-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--out", default="results/compression_summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    methods = resolve_methods(args.methods)

    if args.report_only:
        summary = aggregate_results(str(root))
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps({"written": str(out_path), "num_models": summary["num_models"]}))
        return

    model_dirs = sorted(p for p in root.glob("model_*") if p.is_dir())
    jobs = []
    for model_dir in model_dirs:
        out_path = model_dir / "compression_result.json"
        if args.resume and out_path.exists():
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                existing = set(payload.get("methods", {}).keys())
                if all(m in existing for m in methods):
                    continue
            except Exception:
                pass
        jobs.append(
            {
                "model_dir": str(model_dir),
                "methods": methods,
                "seed": args.seed,
                "calib_seed": args.calib_seed,
                "calib_n": args.calib_n,
                "calib_batch_size": args.calib_batch_size,
                "device": args.device,
                "data_dir": args.data_dir,
            }
        )

    if args.workers <= 1:
        for job in jobs:
            model_dir, result = _run_job(job)
            out_path = Path(model_dir) / "compression_result.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_run_job, job) for job in jobs]
            for fut in as_completed(futures):
                model_dir, result = fut.result()
                out_path = Path(model_dir) / "compression_result.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

    summary = aggregate_results(str(root))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"written": str(out_path), "num_models": summary["num_models"]}))


if __name__ == "__main__":
    main()
