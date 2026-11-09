import csv
import json
import tempfile
import unittest
from pathlib import Path

from ablations.metrics.hierarchy import CWEGraph
from ablations.metrics.prediction_transition import (
    analyze_prediction_transition,
    audit_run_configs,
    write_outputs,
)


class PredictionTransitionTests(unittest.TestCase):
    def setUp(self):
        self.graph = CWEGraph(
            [
                ("R", "A"),
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "E"),
            ]
        )
        self.baseline = [
            {"sample_id": "s1", "pred_cwe": "B", "final_cwe_ids": ["B", "E"]},
            {"sample_id": "s2", "pred_cwe": "C", "final_cwe_ids": ["D"]},
            {"sample_id": "s3", "pred_cwe": "C", "final_cwe_ids": ["C"]},
            {"sample_id": "s4", "pred_cwe": "E", "final_cwe_ids": ["D"]},
            {"sample_id": "s5", "pred_cwe": "C", "final_cwe_ids": ["D"]},
            {"sample_id": "s6", "pred_cwe": "B", "final_cwe_ids": ["D"]},
        ]
        self.full = [
            {"sample_id": "s6", "pred_cwe": "E", "final_cwe_ids": ["D"]},
            {"sample_id": "s5", "pred_cwe": "C", "final_cwe_ids": ["D"]},
            {"sample_id": "s4", "pred_cwe": "B", "final_cwe_ids": ["D"]},
            {"sample_id": "s3", "pred_cwe": "B", "final_cwe_ids": ["C"]},
            {"sample_id": "s2", "pred_cwe": "D", "final_cwe_ids": ["D"]},
            {"sample_id": "s1", "pred_cwe": "E", "final_cwe_ids": ["E", "B"]},
        ]

    def analyze(self):
        return analyze_prediction_transition(
            self.baseline,
            self.full,
            id2cwe={},
            graph=self.graph,
            baseline_name="flat",
            full_name="full",
            bootstrap_samples=100,
            bootstrap_seed=7,
        )

    def test_set_hit_pairing_and_distance_direction(self):
        metrics, rows = self.analyze()
        counts = metrics["correctness"]["transition_counts"]
        self.assertEqual(
            counts,
            {
                "both_correct": 1,
                "corrected": 1,
                "regressed": 1,
                "both_wrong": 3,
            },
        )
        self.assertEqual(metrics["correctness"]["full_set_hit_accuracy"], 2 / 6)
        self.assertEqual(
            metrics["correctness"]["full_minus_baseline_set_hit_accuracy"],
            0.0,
        )
        self.assertEqual(metrics["correctness"]["mcnemar_exact"]["p_value"], 1.0)
        both_wrong = metrics["hierarchy"]["both_wrong"]
        self.assertEqual(
            both_wrong["distance_change_counts"],
            {"full_closer": 1, "tied": 1, "full_farther": 1},
        )
        by_id = {row["sample_id"]: row for row in rows}
        self.assertEqual(by_id["s4"]["distance_delta_baseline_minus_full"], 3)
        self.assertEqual(by_id["s6"]["distance_delta_baseline_minus_full"], -3)
        self.assertTrue(by_id["s1"]["baseline_correct"])
        self.assertTrue(by_id["s1"]["full_correct"])

    def test_duplicate_sample_id_is_rejected(self):
        duplicate = [*self.baseline, dict(self.baseline[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            analyze_prediction_transition(
                duplicate,
                self.full,
                id2cwe={},
                graph=self.graph,
                bootstrap_samples=0,
            )

    def test_mismatched_sample_sets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample_id sets differ"):
            analyze_prediction_transition(
                self.baseline[:-1],
                self.full,
                id2cwe={},
                graph=self.graph,
                bootstrap_samples=0,
            )

    def test_mismatched_gold_sets_are_rejected(self):
        changed = [dict(row) for row in self.full]
        changed[-1]["final_cwe_ids"] = ["B"]
        with self.assertRaisesRegex(ValueError, "Gold CWE sets differ"):
            analyze_prediction_transition(
                self.baseline,
                changed,
                id2cwe={},
                graph=self.graph,
                bootstrap_samples=0,
            )

    def test_writes_all_audit_outputs(self):
        metrics, rows = self.analyze()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "transition"
            write_outputs(output_dir, metrics, rows)
            expected = {
                "metrics.json",
                "transitions.jsonl",
                "correctness_transition.csv",
                "hierarchy_transition.csv",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected,
            )
            payload = json.loads(
                (output_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["num_examples"], 6)
            with (output_dir / "hierarchy_transition.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                hierarchy_rows = list(csv.DictReader(handle))
            self.assertEqual(len(hierarchy_rows), 49)

    def test_config_audit_rejects_input_length_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_args = root / "baseline.json"
            full_args = root / "full.json"
            baseline_args.write_text(
                json.dumps(
                    {"seed": 42, "input_mode": "code_desc", "description_max_length": 20}
                ),
                encoding="utf-8",
            )
            full_args.write_text(
                json.dumps(
                    {"seed": 42, "input_mode": "code_desc", "description_max_length": 64}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "description_max_length"):
                audit_run_configs(baseline_args, full_args)

    def test_config_audit_accepts_matching_shared_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_args = root / "baseline.json"
            full_args = root / "full.json"
            payload = {"seed": 42, "input_mode": "code_desc", "max_length": 512}
            baseline_args.write_text(json.dumps(payload), encoding="utf-8")
            full_args.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertWarnsRegex(UserWarning, "some invariant fields"):
                result = audit_run_configs(baseline_args, full_args)
            self.assertEqual(result["status"], "partial_missing_fields")
            self.assertEqual(result["mismatches"], {})

    def test_config_audit_reads_nested_training_args(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_args = root / "baseline.json"
            full_args = root / "full.json"
            baseline_args.write_text(
                json.dumps(
                    {
                        "run_id": "baseline",
                        "args": {
                            "seed": 42,
                            "input_mode": "code_desc",
                            "ordering_start_level": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            full_args.write_text(
                json.dumps(
                    {
                        "run_id": "full",
                        "args": {
                            "seed": 42,
                            "input_mode": "code_desc",
                            "ordering_start_level": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ordering_start_level"):
                audit_run_configs(baseline_args, full_args)


if __name__ == "__main__":
    unittest.main()
