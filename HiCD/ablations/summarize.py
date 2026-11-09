"""Aggregate validation-selected test metrics across ablation seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence


if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ablations.run import EXPERIMENTS, PROJECT_ROOT


DEFAULT_METRICS = (
    "accuracy",
    "hit_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mcc",
    "top3_hit_accuracy",
    "top5_hit_accuracy",
    "mean_reciprocal_rank",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def completed_runs(seed_dir: Path) -> list[Path]:
    if not seed_dir.is_dir():
        return []
    return sorted(
        {
            path.parent.resolve()
            for path in seed_dir.glob("*/summary.json")
            if path.is_file() and (path.parent / "test_metrics.json").is_file()
        },
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )


def collect_rows(
    output_root: Path,
    experiment_names: Sequence[str],
    metric_names: Sequence[str],
) -> tuple[list[Dict[str, Any]], list[str]]:
    rows = []
    warnings = []
    for experiment in experiment_names:
        experiment_dir = output_root / experiment
        if not experiment_dir.is_dir():
            warnings.append(f"No output directory for {experiment}")
            continue
        for seed_dir in sorted(experiment_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            candidates = completed_runs(seed_dir)
            if not candidates:
                warnings.append(f"No completed run in {seed_dir}")
                continue
            if len(candidates) > 1:
                warnings.append(
                    f"Multiple completed runs in {seed_dir}; selected latest: "
                    f"{candidates[-1].name}"
                )
            run_dir = candidates[-1]
            metrics = json.loads(
                (run_dir / "test_metrics.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            seed_text = seed_dir.name.removeprefix("seed_")
            row: Dict[str, Any] = {
                "experiment": experiment,
                "seed": int(seed_text),
                "run_dir": str(run_dir),
                "selected_epoch": summary.get(
                    "best_epoch",
                    metrics.get("selected_epoch"),
                ),
            }
            for metric in metric_names:
                value = metrics.get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[metric] = float(value)
            rows.append(row)
    return rows, warnings


def aggregate_rows(
    rows: Sequence[Dict[str, Any]],
    metric_names: Sequence[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for experiment in dict.fromkeys(row["experiment"] for row in rows):
        experiment_rows = [
            row for row in rows if row["experiment"] == experiment
        ]
        metric_summary = {}
        for metric in metric_names:
            values = [
                float(row[metric])
                for row in experiment_rows
                if metric in row
            ]
            if not values:
                continue
            metric_summary[metric] = {
                "n": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "values": values,
            }
        result[experiment] = {
            "seeds": [row["seed"] for row in experiment_rows],
            "runs": len(experiment_rows),
            "metrics": metric_summary,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ablation test metrics over seeds."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--experiments", nargs="+", choices=tuple(EXPERIMENTS))
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_path(args.output_root)
    experiments = args.experiments or list(EXPERIMENTS)
    rows, warnings = collect_rows(output_root, experiments, args.metrics)
    if not rows:
        raise RuntimeError(f"No completed ablation runs found in {output_root}")
    payload = {
        "schema_version": "cwe-ablation-summary-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "output_root": str(output_root),
        "selection": "latest completed timestamped run per experiment/seed",
        "warnings": warnings,
        "runs": rows,
        "aggregate": aggregate_rows(rows, args.metrics),
    }
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else output_root / "summaries"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ablation_summary_{timestamp}.json"
    csv_path = output_dir / f"ablation_runs_{timestamp}.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "experiment",
        "seed",
        "run_dir",
        "selected_epoch",
        *args.metrics,
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()

