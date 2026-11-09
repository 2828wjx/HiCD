"""Summarize common-triple-support ordering rates into an auditable CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_SCHEMA = "cwe-ordering-metrics-v2"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _rate(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a rate in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a rate in [0, 1]") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a rate in [0, 1]")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result != value or result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return result


def _sample_metric(
    summary: Mapping[str, Any], name: str, context: str
) -> tuple[int, float]:
    metric = _mapping(summary.get(name), f"{context}.{name}")
    samples = _integer(
        metric.get("samples"), f"{context}.{name}.samples", minimum=1
    )
    rate = _rate(metric.get("mean_rate"), f"{context}.{name}.mean_rate")
    return samples, rate


def _parse_specs(values: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    names: set[str] = set()
    paths: set[Path] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --metric {value!r}; expected NAME=PATH")
        name, raw_path = (part.strip() for part in value.split("=", 1))
        if not name or not raw_path:
            raise ValueError(f"Invalid --metric {value!r}; expected NAME=PATH")
        path = Path(raw_path).expanduser().resolve()
        if name in names:
            raise ValueError(f"Duplicate metric name: {name}")
        if path in paths:
            raise ValueError(f"Metric paths must be distinct: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        names.add(name)
        paths.add(path)
        result.append((name, path))
    return result


def extract_row(name: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name}: metrics root must be a JSON object")
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError(
            f"{name}: expected {EXPECTED_SCHEMA}; rerun the current evaluator"
        )
    if payload.get("partial_evaluation") is not False:
        raise ValueError(f"{name}: partial evaluation cannot populate the table")

    common = _mapping(
        payload.get("common_triple_support"),
        f"{name}.common_triple_support; rerun the current evaluator",
    )
    summary = _mapping(
        common.get("sample_macro"),
        f"{name}.common_triple_support.sample_macro",
    )
    ph_n, ph_rate = _sample_metric(summary, "pos_hard_margin", name)
    hm_n, hm_rate = _sample_metric(summary, "hard_medium_margin", name)
    triple_n, triple_rate = _sample_metric(summary, "triple_margin", name)
    if not ph_n == hm_n == triple_n:
        raise ValueError(
            f"{name}: common supports differ: PH={ph_n}, HM={hm_n}, "
            f"Triple={triple_n}"
        )
    declared_samples = _integer(
        common.get("samples"),
        f"{name}.common_triple_support.samples",
        minimum=1,
    )
    if declared_samples != triple_n:
        raise ValueError(
            f"{name}: declared common samples={declared_samples}, "
            f"metric samples={triple_n}"
        )
    tolerance = 1e-12
    if triple_rate > min(ph_rate, hm_rate) + tolerance:
        raise ValueError(f"{name}: Triple exceeds min(PH, HM)")
    lower_bound = max(0.0, ph_rate + hm_rate - 1.0)
    if triple_rate + tolerance < lower_bound:
        raise ValueError(f"{name}: Triple violates the conjunction lower bound")
    bounds = _mapping(common.get("bounds"), f"{name}.common_triple_support.bounds")
    if (
        bounds.get("triple_le_min_marginals") is not True
        or bounds.get("triple_ge_frechet_lower_bound") is not True
    ):
        raise ValueError(f"{name}: evaluator-side conjunction checks did not pass")

    broad = _mapping(payload.get("sample_macro"), f"{name}.sample_macro")
    broad_ph_n, broad_ph_rate = _sample_metric(
        broad, "pos_hard_margin", f"{name}.sample_macro"
    )
    pos_hard = _mapping(payload.get("pos_hard"), f"{name}.pos_hard")
    hard_medium = _mapping(
        payload.get("hard_medium"), f"{name}.hard_medium"
    )
    ordering_start_level = _integer(
        payload.get("ordering_start_level"), f"{name}.ordering_start_level"
    )
    legacy = payload.get("legacy_ordering_start_level_defaulted") is True
    return {
        "approach": name,
        "ph_margin_rate": ph_rate,
        "hm_margin_rate": hm_rate,
        "triple_margin_rate": triple_rate,
        "common_support_samples": triple_n,
        "common_support_tuple_pairs": _integer(
            common.get("tuple_pairs"),
            f"{name}.common_triple_support.tuple_pairs",
            minimum=1,
        ),
        "triple_upper_bound_ok": True,
        "triple_lower_bound_ok": True,
        "all_ph_eligible_margin_rate_audit": broad_ph_rate,
        "all_ph_eligible_samples_audit": broad_ph_n,
        "dataset": str(payload.get("dataset", "")),
        "split": str(payload.get("split", "")),
        "seed": _integer(payload.get("seed"), f"{name}.seed"),
        "num_dataset_examples": _integer(
            payload.get("num_dataset_examples"),
            f"{name}.num_dataset_examples",
            minimum=1,
        ),
        "relation_index_sha256": str(payload.get("relation_index_sha256", "")),
        "ph_margin_threshold": float(pos_hard.get("margin")),
        "hm_margin_threshold": float(hard_medium.get("margin")),
        "ordering_start_level": ordering_start_level,
        "legacy_cutoff_defaulted": legacy,
        "formal_ready": (not legacy and ordering_start_level == 1),
        "checkpoint": str(payload.get("checkpoint", "")),
        "source_metrics": str(path),
    }


def summarize(specs: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
    if not specs:
        raise ValueError("At least one --metric is required")
    rows = [extract_row(name, path) for name, path in specs]
    common_fields = (
        "dataset",
        "split",
        "seed",
        "num_dataset_examples",
        "relation_index_sha256",
        "ph_margin_threshold",
        "hm_margin_threshold",
        "ordering_start_level",
        "common_support_samples",
        "common_support_tuple_pairs",
    )
    for field in common_fields:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(
                f"Evaluation-universe mismatch for {field}: {sorted(values)!r}"
            )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a common-support PH/HM/Triple margin table."
    )
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        metavar="NAME=METRICS_JSON",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = summarize(_parse_specs(args.metric))
    write_csv(args.output, rows)
    print(f"Saved common-support ordering table to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
