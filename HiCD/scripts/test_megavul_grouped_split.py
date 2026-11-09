import unittest

from prepare_megavul_stage1 import (
    grouped_iterative_multilabel_split,
    split_cve_overlap,
    split_hash_overlap,
)


def make_record(index, cve_id, cwe_ids, func_hash=None):
    return {
        "source_idx": index,
        "cve_id": cve_id,
        "cwe_ids": list(cwe_ids),
        "func_hash": func_hash or f"hash-{index}",
    }


class MegaVulGroupedSplitTests(unittest.TestCase):
    def test_cve_and_code_components_are_atomic_and_deterministic(self):
        records = []
        for index in range(18):
            label = "CWE-1" if index % 2 == 0 else "CWE-2"
            records.append(
                make_record(index, f"CVE-{index:04d}", [label])
            )

        # This duplicate code connects two different CVEs into one component.
        records[0]["func_hash"] = "shared-code"
        records[1]["func_hash"] = "shared-code"
        # Multiple functions from one CVE must also remain together.
        records.append(
            make_record(18, records[2]["cve_id"], ["CWE-1"])
        )

        first, targets, stats = grouped_iterative_multilabel_split(
            records,
            (0.6, 0.2, 0.2),
            42,
            required_coverage_labels={"CWE-1", "CWE-2"},
        )
        second, _, _ = grouped_iterative_multilabel_split(
            records,
            (0.6, 0.2, 0.2),
            42,
            required_coverage_labels={"CWE-1", "CWE-2"},
        )

        self.assertEqual(first, second)
        self.assertEqual(sum(targets), len(records))
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2], first[18])
        self.assertFalse(any(split_cve_overlap(records, first).values()))
        self.assertFalse(any(split_hash_overlap(records, first).values()))
        self.assertEqual(
            stats["grouping_unit"],
            "connected_components_of_cve_id_and_normalized_code_hash",
        )

        split_labels = [set(), set(), set()]
        for record, split_index in zip(records, first):
            split_labels[split_index].update(record["cwe_ids"])
        self.assertEqual(
            split_labels,
            [{"CWE-1", "CWE-2"}] * 3,
        )

    def test_impossible_required_coverage_is_rejected(self):
        records = [
            make_record(0, "CVE-A", ["CWE-1"]),
            make_record(1, "CVE-B", ["CWE-1"]),
            make_record(2, "CVE-C", ["CWE-2"]),
            make_record(3, "CVE-D", ["CWE-2"]),
            make_record(4, "CVE-E", ["CWE-2"]),
        ]
        with self.assertRaisesRegex(ValueError, "coverage impossible"):
            grouped_iterative_multilabel_split(
                records,
                (0.6, 0.2, 0.2),
                42,
                required_coverage_labels={"CWE-1", "CWE-2"},
            )

    def test_minimum_eval_group_coverage_is_enforced_and_downgraded(self):
        records = [
            make_record(index, f"CVE-A-{index}", ["CWE-1"])
            for index in range(9)
        ]
        records.extend(
            make_record(index, f"CVE-B-{index}", ["CWE-2"])
            for index in range(9, 15)
        )

        assignments, _, stats = grouped_iterative_multilabel_split(
            records,
            (0.6, 0.2, 0.2),
            42,
            required_coverage_labels={"CWE-1", "CWE-2"},
            min_eval_groups_per_label=3,
        )

        label_groups = {
            label: [0, 0, 0]
            for label in ("CWE-1", "CWE-2")
        }
        for record, split_index in zip(records, assignments):
            for label in record["cwe_ids"]:
                label_groups[label][split_index] += 1

        self.assertGreaterEqual(label_groups["CWE-1"][1], 3)
        self.assertGreaterEqual(label_groups["CWE-1"][2], 3)
        self.assertGreaterEqual(label_groups["CWE-2"][1], 2)
        self.assertGreaterEqual(label_groups["CWE-2"][2], 2)
        self.assertEqual(
            stats["effective_min_eval_atomic_groups_by_label"],
            {"CWE-1": 3, "CWE-2": 2},
        )
        self.assertEqual(
            stats["downgraded_min_eval_atomic_groups"]["CWE-2"],
            {
                "available_atomic_groups": 6,
                "requested_eval_groups": 3,
                "effective_eval_groups": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
