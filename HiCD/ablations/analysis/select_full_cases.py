from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablations.metrics.hierarchy import (
    CWEGraph,
    classify_prediction,
    extract_prediction_and_gold,
)
from ablations.metrics.prediction_hierarchy import (
    PROJECT_ROOT,
    read_jsonl,
    resolve_path,
    sha256_file,
    write_jsonl,
)


CATEGORY_ORDER = (
    "ordering_satisfied_and_correct",
    "ordering_satisfied_but_wrong",
)
TRIPLE_RATE_FIELD = "triple_margin_path_balanced_rate"
TRIPLE_COUNT_FIELD = "triple_margin_comparisons"
TEST_FILENAMES = (
    "megavul_hier_test.jsonl",
    "megavul_test.jsonl",
)


def _sample_id(row: Mapping[str, Any], source: str) -> str:
    value = row.get("sample_id")
    if value is None or not str(value).strip():
        raise ValueError(f"A row in {source} has no non-empty sample_id")
    return str(value).strip()


def _index_unique(
    rows: Sequence[Mapping[str, Any]], source: str
) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = _sample_id(row, source)
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {source}")
        indexed[sample_id] = row
    return indexed


def _finite_float(value: Any, *, field: str, sample_id: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} is missing for sample_id {sample_id!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} is not numeric for sample_id {sample_id!r}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"{field} is not finite for sample_id {sample_id!r}: {value!r}"
        )
    return number


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _ordering_status_summary(
    tallies: Mapping[str, Mapping[str, int]],
    *,
    triple_eligible_samples: int,
    threshold: float,
) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    for status in ("complete", "incomplete"):
        sample_count = int(tallies[status]["sample_count"])
        correct_count = int(tallies[status]["set_hit_correct_count"])
        groups[status] = {
            "sample_count": sample_count,
            "subset_share": _safe_ratio(
                sample_count,
                triple_eligible_samples,
            ),
            "set_hit_correct_count": correct_count,
            "set_hit_accuracy": _safe_ratio(correct_count, sample_count),
        }

    complete_accuracy = groups["complete"]["set_hit_accuracy"]
    incomplete_accuracy = groups["incomplete"]["set_hit_accuracy"]
    accuracy_gap_pp = (
        100.0 * (complete_accuracy - incomplete_accuracy)
        if complete_accuracy is not None and incomplete_accuracy is not None
        else None
    )
    return {
        "population": "triple_eligible_samples",
        "population_definition": f"{TRIPLE_COUNT_FIELD} > 0",
        "denominator_samples": int(triple_eligible_samples),
        "threshold": float(threshold),
        "complete_definition": f"{TRIPLE_RATE_FIELD} >= {threshold}",
        "incomplete_definition": f"{TRIPLE_RATE_FIELD} < {threshold}",
        **groups,
        "complete_minus_incomplete_set_hit_accuracy_percentage_points": (
            accuracy_gap_pp
        ),
    }


def _first_present(row: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        value = row.get(field)
        if value is not None and value != "":
            return value
    return None


def _excerpt(value: Any, max_chars: int) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    text = str(value)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n...[truncated]", True


def _compact_source_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_idx": _first_present(row, ("source_idx", "source_id", "idx")),
        "cve_id": _first_present(row, ("cve_id", "cve", "CVE")),
        "project": _first_present(row, ("project", "project_name", "repo")),
        "file_path": _first_present(
            row, ("file_path", "file_name", "filepath", "file")
        ),
        "func_name": _first_present(row, ("func_name", "function_name")),
    }


def _readable_content(row: Mapping[str, Any], max_chars: int) -> Dict[str, Any]:
    description, description_truncated = _excerpt(
        _first_present(row, ("cve_description", "description")),
        max_chars,
    )
    code, code_truncated = _excerpt(
        _first_present(row, ("code", "func", "function_code")),
        max_chars,
    )
    return {
        "description": description,
        "description_truncated": description_truncated,
        "code": code,
        "code_truncated": code_truncated,
        "text_max_chars": max_chars,
    }


def _representative_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    category: str,
    cases_per_category: int,
) -> tuple[list[str], Dict[str, Any]]:
    """Choose typical, not extreme, examples using category medians."""
    comparison_median = (
        statistics.median(float(row[TRIPLE_COUNT_FIELD]) for row in rows)
        if rows
        else None
    )
    if category == "ordering_satisfied_but_wrong":
        graph_rows = [
            row for row in rows if row["hierarchy_distance"] is not None
        ]
        distance_median = (
            statistics.median(
                float(row["hierarchy_distance"]) for row in graph_rows
            )
            if graph_rows
            else None
        )
    else:
        distance_median = 0.0 if rows else None

    def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        distance = row["hierarchy_distance"]
        if distance_median is None or distance is None:
            distance_key = (1, math.inf)
        else:
            distance_key = (0, abs(float(distance) - distance_median))
        comparison_key = (
            abs(float(row[TRIPLE_COUNT_FIELD]) - comparison_median)
            if comparison_median is not None
            else math.inf
        )
        return (*distance_key, comparison_key, str(row["sample_id"]))

    ranked = sorted(rows, key=rank_key)
    selected_ids = [
        str(row["sample_id"]) for row in ranked[:cases_per_category]
    ]
    return selected_ids, {
        "candidate_count": len(rows),
        "hierarchy_distance_median": distance_median,
        "triple_comparison_count_median": comparison_median,
        "selection_rule": (
            "choose the candidates nearest to the category median DAG distance, "
            "then nearest to the median Triple-comparison count; resolve ties "
            "lexicographically by sample_id; do not select maximum gains, "
            "maximum confidence, or maximum margin"
        ),
    }


def select_full_cases(
    prediction_rows: Sequence[Mapping[str, Any]],
    ordering_rows: Sequence[Mapping[str, Any]],
    dataset_rows: Sequence[Mapping[str, Any]],
    *,
    id2cwe: Mapping[int, str],
    graph: CWEGraph,
    triple_margin_threshold: float = 1.0,
    cases_per_category: int = 1,
    text_max_chars: int = 2_000,
    prediction_source: str = "Full predictions",
    ordering_source: str = "Full ordering sample metrics",
    dataset_source: str = "MegaVul test data",
) -> tuple[Dict[str, Any], list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Return ordering-status statistics and two auditable case groups."""
    if not 0.0 <= triple_margin_threshold <= 1.0:
        raise ValueError("triple_margin_threshold must be in [0, 1]")
    if cases_per_category < 1:
        raise ValueError("cases_per_category must be at least 1")
    if text_max_chars < 1:
        raise ValueError("text_max_chars must be at least 1")

    predictions = _index_unique(prediction_rows, prediction_source)
    ordering = _index_unique(ordering_rows, ordering_source)
    dataset = _index_unique(dataset_rows, dataset_source)

    unknown_ordering = sorted(set(ordering) - set(predictions))
    if unknown_ordering:
        raise ValueError(
            "Ordering metrics contain sample_id values absent from Full "
            f"predictions: {unknown_ordering[:10]} (n={len(unknown_ordering)})"
        )
    missing_dataset = sorted(set(predictions) - set(dataset))
    extra_dataset = sorted(set(dataset) - set(predictions))
    if missing_dataset or extra_dataset:
        raise ValueError(
            "Full predictions and MegaVul test data must contain exactly the "
            "same sample_id set: "
            f"missing dataset rows={missing_dataset[:10]} (n={len(missing_dataset)}); "
            f"extra dataset rows={extra_dataset[:10]} (n={len(extra_dataset)})"
        )

    candidates: list[Dict[str, Any]] = []
    ordering_with_triples = 0
    ordering_below_threshold = 0
    ordering_status_tallies = {
        "complete": {"sample_count": 0, "set_hit_correct_count": 0},
        "incomplete": {"sample_count": 0, "set_hit_correct_count": 0},
    }
    for sample_id in sorted(ordering):
        ordering_row = ordering[sample_id]
        raw_count = ordering_row.get(TRIPLE_COUNT_FIELD, 0)
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{TRIPLE_COUNT_FIELD} is not an integer for sample_id "
                f"{sample_id!r}: {raw_count!r}"
            ) from exc
        if count <= 0:
            continue
        ordering_with_triples += 1
        rate = _finite_float(
            ordering_row.get(TRIPLE_RATE_FIELD),
            field=TRIPLE_RATE_FIELD,
            sample_id=sample_id,
        )
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"{TRIPLE_RATE_FIELD} must be in [0, 1] for sample_id "
                f"{sample_id!r}: {rate}"
            )

        prediction, gold_values = extract_prediction_and_gold(
            predictions[sample_id], id2cwe
        )
        full_correct = prediction in set(gold_values)
        ordering_status = (
            "complete"
            if rate >= triple_margin_threshold
            else "incomplete"
        )
        ordering_status_tallies[ordering_status]["sample_count"] += 1
        ordering_status_tallies[ordering_status][
            "set_hit_correct_count"
        ] += int(full_correct)

        if ordering_status == "incomplete":
            ordering_below_threshold += 1
            continue

        hierarchy = classify_prediction(graph, prediction, gold_values)
        category = (
            "ordering_satisfied_and_correct"
            if full_correct
            else "ordering_satisfied_but_wrong"
        )
        candidates.append(
            {
                "case_category": category,
                "sample_id": sample_id,
                "pred_cwe": prediction,
                "gold_cwes": gold_values,
                "set_hit_correct": full_correct,
                TRIPLE_COUNT_FIELD: count,
                TRIPLE_RATE_FIELD: rate,
                "hierarchy_error_category": hierarchy[
                    "hierarchy_error_category"
                ],
                "hierarchy_distance": hierarchy["hierarchy_distance"],
                "matched_gold_cwe": hierarchy["matched_gold_cwe"],
                "graph_evaluable": hierarchy["graph_evaluable"],
                "source": _compact_source_metadata(dataset[sample_id]),
            }
        )

    selected: list[Dict[str, Any]] = []
    category_results: Dict[str, Any] = {}
    for category in CATEGORY_ORDER:
        members = [row for row in candidates if row["case_category"] == category]
        selected_ids, category_metrics = _representative_selection(
            members,
            category=category,
            cases_per_category=cases_per_category,
        )
        selected_set = set(selected_ids)
        rank_by_id = {
            sample_id: rank for rank, sample_id in enumerate(selected_ids, start=1)
        }
        category_results[category] = {
            **category_metrics,
            "selected_count": len(selected_ids),
            "selected_sample_ids": selected_ids,
        }
        for row in members:
            row["is_selected"] = row["sample_id"] in selected_set
            row["selection_rank"] = rank_by_id.get(row["sample_id"])
            if row["is_selected"]:
                selected.append(
                    {
                        **row,
                        "content": _readable_content(
                            dataset[str(row["sample_id"])], text_max_chars
                        ),
                    }
                )

    candidates.sort(
        key=lambda row: (
            CATEGORY_ORDER.index(str(row["case_category"])),
            str(row["sample_id"]),
        )
    )
    selected.sort(
        key=lambda row: (
            CATEGORY_ORDER.index(str(row["case_category"])),
            int(row["selection_rank"]),
            str(row["sample_id"]),
        )
    )
    metrics = {
        "schema_version": "full-ordering-case-selection-v2",
        "definitions": {
            "correctness": (
                "set-hit: Full prediction equals any final gold CWE"
            ),
            "ordering_satisfied": (
                f"{TRIPLE_COUNT_FIELD} > 0 and sample-macro "
                f"{TRIPLE_RATE_FIELD} >= {triple_margin_threshold}"
            ),
            "level_reduction": (
                "the ordering evaluator first averages Triple-margin events "
                "within each path and then averages paths within each sample"
            ),
        },
        "pairing": {
            "join_key": "sample_id",
            "row_order_used_for_pairing": False,
            "sample_ids_unique_in_each_input": True,
            "prediction_dataset_sample_id_sets_equal": True,
            "ordering_sample_ids_subset_of_predictions": True,
        },
        "support": {
            "full_prediction_samples": len(predictions),
            "ordering_rows": len(ordering),
            "ordering_rows_with_triple_comparisons": ordering_with_triples,
            "ordering_rows_below_threshold": ordering_below_threshold,
            "all_candidate_rows": len(candidates),
        },
        "ordering_status_summary": _ordering_status_summary(
            ordering_status_tallies,
            triple_eligible_samples=ordering_with_triples,
            threshold=triple_margin_threshold,
        ),
        "settings": {
            "triple_margin_threshold": triple_margin_threshold,
            "cases_per_category": cases_per_category,
            "text_max_chars": text_max_chars,
        },
        "anti_cherry_picking_policy": (
            "Candidates are fixed by ordering satisfaction and set-hit outcome. "
            "Representatives are chosen near category medians with sample_id "
            "tie-breaking, never by inspecting or maximizing improvements."
        ),
        "category_results": category_results,
    }
    return metrics, candidates, selected


def write_outputs(
    output_dir: Path,
    metrics: Mapping[str, Any],
    candidates: Iterable[Dict[str, Any]],
    selected: Iterable[Dict[str, Any]],
) -> None:
    paths = (
        output_dir / "metrics.json",
        output_dir / "ordering_status_summary.csv",
        output_dir / "all_candidates.jsonl",
        output_dir / "selected_cases.jsonl",
    )
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing case-selection outputs: "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = metrics["ordering_status_summary"]
    with paths[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "ordering_status",
                "sample_count",
                "share_of_triple_eligible_pct",
                "set_hit_accuracy_pct",
            ),
        )
        writer.writeheader()
        for status, label in (
            ("complete", "Complete"),
            ("incomplete", "Incomplete"),
        ):
            values = summary[status]
            share = values["subset_share"]
            accuracy = values["set_hit_accuracy"]
            writer.writerow(
                {
                    "ordering_status": label,
                    "sample_count": values["sample_count"],
                    "share_of_triple_eligible_pct": (
                        None if share is None else 100.0 * share
                    ),
                    "set_hit_accuracy_pct": (
                        None if accuracy is None else 100.0 * accuracy
                    ),
                }
            )
    write_jsonl(paths[2], candidates)
    write_jsonl(paths[3], selected)


def _default_test_path(data_dir: Path) -> Path:
    for filename in TEST_FILENAMES:
        candidate = data_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No MegaVul test JSONL found in {data_dir}; tried {TEST_FILENAMES}. "
        "Pass --test-jsonl explicitly if it has another filename."
    )


def _new_output_dir(parent: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"full_case_selection_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = parent / f"full_case_selection_{timestamp}_{suffix:02d}"
        suffix += 1
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select representative Full cases where Triple ordering is "
            "satisfied and final set-hit classification succeeds or fails."
        )
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ordering-sample-metrics", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--test-jsonl",
        help="Defaults to megavul_hier_test.jsonl under --data-dir.",
    )
    parser.add_argument(
        "--graph",
        default=str(PROJECT_ROOT / "processed" / "cwe_edges.json"),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--triple-margin-threshold", type=float, default=1.0)
    parser.add_argument("--cases-per-category", type=int, default=1)
    parser.add_argument("--text-max-chars", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = resolve_path(args.predictions)
    ordering_path = resolve_path(args.ordering_sample_metrics)
    data_dir = resolve_path(args.data_dir)
    test_path = (
        resolve_path(args.test_jsonl)
        if args.test_jsonl
        else _default_test_path(data_dir)
    )
    graph_path = resolve_path(args.graph)
    vocab_path = data_dir / "cwe_vocab.json"
    for label, path in (
        ("Full predictions", predictions_path),
        ("Full ordering sample metrics", ordering_path),
        ("MegaVul test JSONL", test_path),
        ("CWE vocabulary", vocab_path),
        ("Official CWE graph", graph_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    id2cwe = {
        int(key): str(value)
        for key, value in vocab["final_label_id2cwe"].items()
    }
    metrics, candidates, selected = select_full_cases(
        read_jsonl(predictions_path),
        read_jsonl(ordering_path),
        read_jsonl(test_path),
        id2cwe=id2cwe,
        graph=CWEGraph.from_json(graph_path),
        triple_margin_threshold=args.triple_margin_threshold,
        cases_per_category=args.cases_per_category,
        text_max_chars=args.text_max_chars,
        prediction_source=str(predictions_path),
        ordering_source=str(ordering_path),
        dataset_source=str(test_path),
    )
    metrics.update(
        {
            "inputs": {
                "predictions": str(predictions_path),
                "predictions_sha256": sha256_file(predictions_path),
                "ordering_sample_metrics": str(ordering_path),
                "ordering_sample_metrics_sha256": sha256_file(ordering_path),
                "test_jsonl": str(test_path),
                "test_jsonl_sha256": sha256_file(test_path),
                "cwe_vocab": str(vocab_path),
                "cwe_vocab_sha256": sha256_file(vocab_path),
                "graph": str(graph_path),
                "graph_sha256": sha256_file(graph_path),
            }
        }
    )
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else _new_output_dir(predictions_path.parent)
    )
    write_outputs(output_dir, metrics, candidates, selected)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved Full case selection to: {output_dir}")


if __name__ == "__main__":
    main()
