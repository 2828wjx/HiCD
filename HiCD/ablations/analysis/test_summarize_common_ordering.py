from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ablations.analysis.summarize_common_ordering import (
    extract_row,
    summarize,
    write_csv,
)


def _payload(name: str, ph: float, hm: float, triple: float) -> dict:
    return {
        "schema_version": "cwe-ordering-metrics-v2",
        "dataset": "megavul",
        "split": "test",
        "seed": 42,
        "checkpoint": f"/{name}/best_model.pt",
        "relation_index_sha256": "a" * 64,
        "num_dataset_examples": 100,
        "partial_evaluation": False,
        "ordering_start_level": 1,
        "legacy_ordering_start_level_defaulted": False,
        "pos_hard": {"margin": 0.15},
        "hard_medium": {"margin": 0.05},
        "sample_macro": {
            "pos_hard_margin": {"samples": 20, "mean_rate": 0.9},
        },
        "common_triple_support": {
            "tuple_pairs": 30,
            "samples": 10,
            "sample_macro": {
                "pos_hard_margin": {"samples": 10, "mean_rate": ph},
                "hard_medium_margin": {"samples": 10, "mean_rate": hm},
                "triple_margin": {"samples": 10, "mean_rate": triple},
            },
            "bounds": {
                "triple_le_min_marginals": True,
                "triple_ge_frechet_lower_bound": True,
                "tolerance": 1e-12,
            },
        },
    }


class CommonOrderingSummaryTests(unittest.TestCase):
    def test_writes_valid_common_support_table(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            paths = []
            for name, ph, hm, triple in (
                ("ph", 0.9, 0.1, 0.05),
                ("hm", 0.1, 0.9, 0.05),
            ):
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(_payload(name, ph, hm, triple)),
                    encoding="utf-8",
                )
                paths.append((name, path))
            rows = summarize(paths)
            output = root / "table.csv"
            write_csv(output, rows)
            with output.open("r", encoding="utf-8-sig", newline="") as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["common_support_samples"], "10")
            self.assertEqual(loaded[0]["formal_ready"], "True")

    def test_rejects_invalid_conjunction_rate(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "bad.json"
            path.write_text(
                json.dumps(_payload("bad", 0.1, 0.9, 0.2)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Triple exceeds"):
                extract_row("bad", path)


if __name__ == "__main__":
    unittest.main()
