import copy
import unittest
from pathlib import Path

from cwe_label_normalization import (
    apply_cwe_policy,
    load_json,
    validate_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "processed" / "cwe_catalog_metadata.json"
POLICY_PATH = (
    PROJECT_ROOT / "config" / "cwe_label_normalization_v4.20.json"
)


class CweLabelNormalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Unit tests must also run in a source-only checkout without CWE data.
        cls.catalog = {
            "CWE-17": {"status": "Deprecated"},
            "CWE-18": {"status": "Deprecated"},
            "CWE-1187": {"status": "Deprecated"},
            "CWE-908": {"status": "Stable"},
            "CWE-399": {"status": "Stable", "entry_type": "Category"},
        }
        cls.policy = load_json(POLICY_PATH)
        validate_policy(
            cls.policy,
            cls.catalog,
        )

    def test_deprecated_duplicate_is_remapped_and_deduplicated(self):
        current, actions = apply_cwe_policy(
            ["CWE-1187", "CWE-908"],
            self.catalog,
            self.policy,
        )
        self.assertEqual(current, ["CWE-908"])
        self.assertEqual(actions[0]["source"], "CWE-1187")
        self.assertEqual(actions[0]["target"], "CWE-908")

    def test_deprecated_category_without_replacement_is_dropped(self):
        current, actions = apply_cwe_policy(
            ["CWE-17"],
            self.catalog,
            self.policy,
        )
        self.assertEqual(current, [])
        self.assertEqual(actions[0]["action"], "drop")

    def test_active_category_is_retained(self):
        current, actions = apply_cwe_policy(
            ["CWE-399"],
            self.catalog,
            self.policy,
        )
        self.assertEqual(current, ["CWE-399"])
        self.assertEqual(actions, [])

    def test_uncovered_deprecated_entry_fails_closed(self):
        incomplete_policy = copy.deepcopy(self.policy)
        del incomplete_policy["rules"]["CWE-1187"]
        with self.assertRaisesRegex(ValueError, "no explicit"):
            apply_cwe_policy(
                ["CWE-1187"],
                self.catalog,
                incomplete_policy,
            )


if __name__ == "__main__":
    unittest.main()
