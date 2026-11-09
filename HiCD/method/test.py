import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from method.data import CWEJsonlDataset
from HiCD.method.losses import HierarchicalCWELoss
from method.path_sampling import sample_supervision
from method.relations import build_level_candidate_mask_from_jsonl
from scripts.text_field_cleaning import clean_description_text


class MethodTests(unittest.TestCase):
    def test_identifiers_are_removed_without_removing_surrounding_text(self):
        text = "A CWE-119 bug in CVE-2025-12345; cwe_787 also applies."
        cleaned = clean_description_text(text)
        self.assertNotIn("119", cleaned)
        self.assertNotIn("12345", cleaned)
        self.assertNotIn("787", cleaned)
        self.assertIn("bug in", cleaned)
        self.assertEqual(cleaned, clean_description_text(cleaned))

    def test_path_sampling_is_repeatable_and_uses_alternative_paths(self):
        repository = {"CWE-3": {"all_sequences": [["CWE-1", "CWE-3"], ["CWE-2", "CWE-3"]]}}
        mapping = {"CWE-1": 0, "CWE-2": 1, "CWE-3": 2}
        source = {"cwe_id": "CWE-3", "hierarchy_loss_mask": 1}
        def sample(epoch):
            return sample_supervision(source, repository, mapping, 3, seed=42, epoch=epoch, sample_id="x")
        self.assertEqual(sample(4), sample(4))
        self.assertEqual({tuple(sample(epoch)["level_ids"]) for epoch in range(30)}, {(0, 2, -100), (1, 2, -100)})
        self.assertEqual(sample(4)["real_level_mask"], [1, 1, 0])
        self.assertNotIn("level_ids", source)
        with self.assertRaises(ValueError):
            sample_supervision(source, repository, mapping, 1, seed=42, epoch=0, sample_id="x")

    def test_repository_is_loaded_and_all_paths_define_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"sample_id": "s", "cwe_id": "CWE-3", "label_id": 0,
                   "hierarchy_loss_mask": 1, "contrast_loss_mask": 1,
                   "level_ids": [0, 2], "real_level_mask": [1, 1], "contrast_level_mask": [1, 1]}
            (root / "train.jsonl").write_text(json.dumps(row) + "\n")
            (root / "dataset_cwe_node_sequences.json").write_text(json.dumps({
                "CWE-3": {"all_sequences": [["CWE-1", "CWE-3"], ["CWE-2", "CWE-3"]]}}))
            (root / "cwe_vocab.json").write_text(json.dumps({"level_node_cwe2id": {"CWE-1": 0, "CWE-2": 1, "CWE-3": 2}}))
            data = CWEJsonlDataset(root / "train.jsonl", num_levels=2)
            data.set_epoch(1, 123)
            self.assertEqual(data[0], data[0])
            mask = build_level_candidate_mask_from_jsonl(root / "train.jsonl", num_level_nodes=3, num_levels=2)
            self.assertEqual(mask.tolist(), [[True, True, False], [False, False, True]])

    def test_union_probability_mass_and_positive_exclusion(self):
        scores = torch.tensor([[[0.2, 0.1, 0.0, -0.1, 0.9], [0.8, 0.6, 0.4, 0.1, 0.2]]], requires_grad=True)
        hard = torch.tensor([[1, 2], [0, 2], [-100, -100], [-100, -100], [-100, -100]])
        medium = torch.tensor([[3], [3], [-100], [-100], [-100]])
        criterion = PaperHierarchicalCWELoss(level_candidate_mask=torch.ones(2, 5, dtype=torch.bool),
            hard_negative_ids=hard, medium_negative_ids=medium, temperature=1.0,
            lambda_same_level=1.0, lambda_pos_hard=0.2, lambda_hard_medium=0.05,
            ordering_start_level=1)
        batch = {"final_label_mask": torch.tensor([[True, True, False]]),
                 "final_classification_mask": torch.ones(1),
                 "level_ids_multi": torch.tensor([[[4, 0], [4, 1]]]),
                 "real_level_masks": torch.ones(1, 2, 2), "contrast_level_masks": torch.ones(1, 2, 2),
                 "hierarchy_loss_masks": torch.ones(1, 2), "contrast_loss_masks": torch.ones(1, 2),
                 "path_mask": torch.ones(1, 2), "hierarchy_loss_weights": torch.ones(1, 2),
                 "contrast_loss_weights": torch.ones(1, 2)}
        outputs = {"final_logits": torch.zeros(1, 3, requires_grad=True),
                   "level_reprs": scores, "label_reprs": torch.eye(5)}
        result = criterion(outputs, batch)
        expected_hn = ((torch.logsumexp(scores[0, 0], 0) - scores[0, 0, 4])
                       + (torch.logsumexp(scores[0, 1], 0) - torch.logsumexp(scores[0, 1, :2], 0))) / 2
        self.assertAlmostEqual(result["loss_same_level"].item(), expected_hn.item(), places=6)
        self.assertAlmostEqual(result["loss_final"].item(), math.log(1.5), places=6)
        expected_ph = (F.softplus(torch.tensor(0.4 - 0.8 + 0.15)) + F.softplus(torch.tensor(0.4 - 0.6 + 0.15))) / 2
        self.assertAlmostEqual(result["loss_pos_hard"].item(), expected_ph.item(), places=6)
        self.assertAlmostEqual(result["loss_hard_medium"].item(), F.softplus(torch.tensor(0.1 - 0.4 + 0.05)).item(), places=6)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        # Duplicating an identical sampled path does not change the objective.
        repeated = dict(batch)
        for name in ["level_ids_multi", "real_level_masks", "contrast_level_masks",
                     "hierarchy_loss_masks", "contrast_loss_masks", "path_mask",
                     "hierarchy_loss_weights", "contrast_loss_weights"]:
            repeated[name] = torch.cat([batch[name], batch[name][:, :1]], dim=1)
        self.assertAlmostEqual(criterion(outputs, repeated)["loss"].item(), result["loss"].item(), places=6)

    def test_no_hierarchy_positions_have_zero_auxiliary_losses(self):
        criterion = PaperHierarchicalCWELoss(level_candidate_mask=torch.ones(1, 2, dtype=torch.bool))
        logits = torch.zeros(1, 2, requires_grad=True)
        batch = {"final_labels": torch.tensor([0]), "final_classification_mask": torch.ones(1),
                 "level_ids": torch.tensor([[-100]]), "real_level_mask": torch.zeros(1, 1),
                 "contrast_level_mask": torch.zeros(1, 1), "hierarchy_loss_mask": torch.zeros(1),
                 "contrast_loss_mask": torch.zeros(1), "hierarchy_loss_weight": torch.zeros(1),
                 "contrast_loss_weight": torch.zeros(1)}
        outputs = {"final_logits": logits, "level_reprs": torch.ones(1, 1, 2, requires_grad=True), "label_reprs": torch.eye(2)}
        result = criterion(outputs, batch)
        self.assertAlmostEqual(result["loss"].item(), math.log(2), places=6)
        for name in ["loss_same_level", "loss_pos_hard", "loss_hard_medium"]:
            self.assertEqual(result[name].item(), 0.0)
        result["loss"].backward()


if __name__ == "__main__":
    unittest.main()
