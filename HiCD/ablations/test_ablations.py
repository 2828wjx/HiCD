import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from ablations.metrics.hierarchy import (
    CWEGraph,
    classify_prediction,
    extract_prediction_and_gold,
    summarize_hierarchy_rows,
)
from ablations.metrics.ordering import OrderingAccumulator
from ablations.run import (
    EXPERIMENTS,
    effective_args,
    parse_common_overrides,
    select_experiments,
)
from method.data import CWEHierCollator
from method.losses import HierarchicalCWELoss
from method.modeling import HierarchicalCWEClassifier


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            is_encoder_decoder=False,
            initializer_range=0.02,
        )
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(
        self,
        input_ids,
        attention_mask,
        return_dict=True,
        token_type_ids=None,
    ):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeAutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return FakeEncoder()


class FakeTokenizer:
    model_input_names = ["input_ids", "attention_mask"]

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=True,
        max_length=None,
    ):
        values = list(range(3, 3 + len(text.split())))
        if truncation and max_length is not None:
            values = values[:max_length]
        return {"input_ids": values}

    def build_inputs_with_special_tokens(self, first, second=None):
        if second is None:
            return [1, *first, 2]
        return [1, *first, 2, *second, 2]

    def pad(self, features, *, padding, return_tensors, max_length=None):
        target = max(len(value["input_ids"]) for value in features)
        return {
            "input_ids": torch.tensor(
                [
                    value["input_ids"]
                    + [0] * (target - len(value["input_ids"]))
                    for value in features
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [1] * len(value["input_ids"])
                    + [0] * (target - len(value["input_ids"]))
                    for value in features
                ],
                dtype=torch.long,
            ),
        }


class AblationMatrixTests(unittest.TestCase):
    def test_paper_incremental_conditions_keep_crd_fixed(self):
        common = {"lambda_same_level": 1.0, "lambda_pos_hard": 0.2,
                  "lambda_hard_medium": 0.05, "lambda_geometry_alignment": 0.05}
        names = ["incremental_none", "incremental_hn", "incremental_hn_ph", "full"]
        expected = [(0, 0, 0), (1, 0, 0), (1, 0.2, 0), (1, 0.2, 0.05)]
        for name, weights in zip(names, expected):
            args = effective_args(common, EXPERIMENTS[name])
            self.assertEqual(tuple(args[key] for key in ["lambda_same_level", "lambda_pos_hard", "lambda_hard_medium"]), weights)
            self.assertEqual(args["lambda_geometry_alignment"], 0.05)

    def test_groups_are_deduplicated(self):
        selected = select_experiments(
            groups=("input", "structure", "loss"),
            names=None,
        )
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({value.name for value in selected}), 10)
        self.assertIn("loss", EXPERIMENTS["full"].groups)
        self.assertIn("structure", EXPERIMENTS["flat"].groups)
        self.assertIn("loss", EXPERIMENTS["flat"].groups)

    def test_loss_variants_only_change_intended_lambdas(self):
        common = {
            "lambda_same_level": 1.0,
            "lambda_pos_hard": 0.2,
            "lambda_hard_medium": 0.05,
            "lambda_geometry_alignment": 0.05,
            "geometry_alignment_mode": "semantic",
            "geometry_alignment_seed": 2026,
            "epochs": 50,
        }
        expected = {
            "no_alignment": (1.0, 0.2, 0.05, 0.0),
            "flat": (0.0, 0.0, 0.0, 0.0),
            "same_level_only": (1.0, 0.0, 0.0, 0.0),
            "pos_hard_only": (0.0, 0.2, 0.0, 0.0),
            "hard_medium_only": (0.0, 0.0, 0.05, 0.0),
        }
        for name, lambdas in expected.items():
            values = effective_args(common, EXPERIMENTS[name])
            self.assertEqual(
                (
                    values["lambda_same_level"],
                    values["lambda_pos_hard"],
                    values["lambda_hard_medium"],
                    values["lambda_geometry_alignment"],
                ),
                lambdas,
            )
            self.assertEqual(values["epochs"], 50)

    def test_random_alignment_only_replaces_semantic_mapping(self):
        common = {
            "input_mode": "code_desc",
            "lambda_same_level": 1.0,
            "lambda_pos_hard": 0.2,
            "lambda_hard_medium": 0.05,
            "lambda_geometry_alignment": 0.05,
            "geometry_alignment_mode": "semantic",
            "geometry_alignment_seed": 2026,
            "alignment_start_epoch": 4,
            "alignment_warmup_epochs": 3,
        }
        values = effective_args(common, EXPERIMENTS["random_alignment"])

        self.assertEqual(values["geometry_alignment_mode"], "random_fixed")
        for key, value in common.items():
            if key != "geometry_alignment_mode":
                self.assertEqual(values[key], value)

        full_values = effective_args(common, EXPERIMENTS["full"])
        self.assertEqual(full_values["geometry_alignment_mode"], "semantic")

    def test_cli_common_overrides_parse_json_values(self):
        values = parse_common_overrides(
            ["epochs=1", "local_files_only=false", "device=cuda"]
        )
        self.assertEqual(values["epochs"], 1)
        self.assertIs(values["local_files_only"], False)
        self.assertEqual(values["device"], "cuda")

class StructureAblationTests(unittest.TestCase):
    @patch("method.modeling.AutoModel", FakeAutoModel)
    def test_no_level_embedding_keeps_state_dict_and_removes_level_identity(self):
        full = HierarchicalCWEClassifier(
            "fake",
            num_final_labels=3,
            num_level_nodes=5,
            num_levels=3,
            dropout=0.0,
            use_level_embeddings=True,
        )
        no_level = HierarchicalCWEClassifier(
            "fake",
            num_final_labels=3,
            num_level_nodes=5,
            num_levels=3,
            dropout=0.0,
            use_level_embeddings=False,
        )
        self.assertEqual(list(full.state_dict()), list(no_level.state_dict()))
        no_level.load_state_dict(full.state_dict(), strict=True)
        full.eval()
        no_level.eval()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }
        full_outputs = full(**inputs)
        no_level_outputs = no_level(**inputs)
        self.assertTrue(
            torch.allclose(
                full_outputs["final_logits"],
                no_level_outputs["final_logits"],
            )
        )
        self.assertTrue(
            torch.allclose(
                no_level_outputs["level_reprs"][:, 0],
                no_level_outputs["level_reprs"][:, 1],
            )
        )

    def test_flat_loss_does_not_touch_invalid_hierarchy_outputs(self):
        criterion = HierarchicalCWELoss(
            level_candidate_mask=torch.ones((2, 3), dtype=torch.bool),
            lambda_same_level=0.0,
            lambda_pos_hard=0.0,
            lambda_hard_medium=0.0,
        )
        logits = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
        result = criterion(
            {
                "final_logits": logits,
                "level_reprs": torch.full((1, 2, 2), float("nan")),
                "label_reprs": torch.full((3, 2), float("nan")),
                "loss_geometry_alignment": torch.tensor(float("nan")),
            },
            {
                "final_labels": torch.tensor([0]),
                "final_classification_mask": torch.ones(1),
            },
        )
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(result["loss_same_level"].item(), 0.0)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class InputBudgetTests(unittest.TestCase):
    def test_single_input_modes_respect_explicit_budgets(self):
        item = {
            "code": "one two three four five six",
            "description": "alpha beta gamma delta epsilon",
        }
        code_collator = CWEHierCollator(
            FakeTokenizer(),
            input_mode="code",
            max_length=20,
            code_max_length=3,
        )
        desc_collator = CWEHierCollator(
            FakeTokenizer(),
            input_mode="desc",
            max_length=20,
            description_max_length=2,
        )
        code_feature = code_collator._tokenize_batch([item])
        desc_feature = desc_collator._tokenize_batch([item])
        self.assertEqual(code_feature["attention_mask"].sum().item(), 5)
        self.assertEqual(desc_feature["attention_mask"].sum().item(), 4)


class HierarchyMetricTests(unittest.TestCase):
    def setUp(self):
        self.graph = CWEGraph(
            [
                ("R", "A"),
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "E"),
                ("X", "Y"),
            ]
        )

    def test_error_categories_and_minimum_multilabel_distance(self):
        self.assertEqual(self.graph.relation("A", "D"), "ancestor_descendant")
        self.assertEqual(self.graph.relation("B", "C"), "sibling")
        self.assertEqual(self.graph.relation("D", "E"), "cousin")
        self.assertEqual(self.graph.relation("D", "Y"), "disconnected")
        result = classify_prediction(self.graph, "D", ["Y", "C"])
        self.assertEqual(result["matched_gold_cwe"], "C")
        self.assertEqual(result["hierarchy_distance"], 3)
        self.assertEqual(result["hierarchy_error_category"], "cross_branch")

    def test_unsupported_exact_has_no_graph_distance(self):
        row = classify_prediction(self.graph, "Remain", ["Remain"])
        self.assertEqual(row["hierarchy_error_category"], "exact")
        self.assertIsNone(row["hierarchy_distance"])
        summary = summarize_hierarchy_rows([row])
        self.assertEqual(summary["graph_coverage"], 0.0)

    def test_prediction_field_variants(self):
        prediction, gold = extract_prediction_and_gold(
            {"pred_id": 1, "true_id": 2},
            {1: "CWE-1", 2: "CWE-2"},
        )
        self.assertEqual(prediction, "CWE-1")
        self.assertEqual(gold, ["CWE-2"])

    def test_ordering_accumulator(self):
        accumulator = OrderingAccumulator(
            pos_hard_margin=0.1,
            hard_medium_margin=0.1,
        )
        accumulator.add(
            sample_id="s1",
            level=1,
            positive_score=0.8,
            hard_scores=[0.5],
            medium_scores=[0.2],
        )
        summary = accumulator.summary()
        self.assertEqual(summary["pos_hard"]["strict_rate"], 1.0)
        self.assertEqual(summary["hard_medium"]["margin_rate"], 1.0)
        self.assertEqual(summary["triple_pairs"]["strict_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
