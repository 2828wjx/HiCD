from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Sequence

from ablations.metrics.hierarchy import percentile


def _rate(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return sum(bool(value) for value in values) / len(values)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


class OrderingAccumulator:
    """Collect cosine ordering gaps using the exact selected relation pools."""

    def __init__(self, *, pos_hard_margin: float, hard_medium_margin: float):
        self.pos_hard_margin = float(pos_hard_margin)
        self.hard_medium_margin = float(hard_medium_margin)
        self.positions = 0
        self.ordering_evaluated_positions = 0
        self.positions_with_hard = 0
        self.positions_with_medium = 0
        self.positions_with_both = 0
        self.pos_hard_gaps: list[float] = []
        self.hard_medium_gaps: list[float] = []
        # Marginal events replicated on exactly the same (positive, hard,
        # medium) tuples as ``triple_margin``.  These are intentionally kept
        # separate from the broader Pos-Hard pool, which also contains
        # positions without any medium candidate.
        self.common_pos_hard_margin: list[bool] = []
        self.common_hard_medium_margin: list[bool] = []
        self.triple_strict: list[bool] = []
        self.triple_margin: list[bool] = []
        self.position_joint_strict: list[bool] = []
        self.position_joint_margin: list[bool] = []
        self.sample_path_events: Dict[
            str, Dict[str, Dict[str, list[bool]]]
        ] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.by_level: Dict[int, "OrderingAccumulator"] = {}

    def add(
        self,
        *,
        sample_id: str,
        path_id: str = "0",
        level: int,
        positive_score: float,
        hard_scores: Sequence[float],
        medium_scores: Sequence[float],
        include_ordering_comparisons: bool = True,
        add_to_level: bool = True,
    ) -> None:
        hard_scores = [float(value) for value in hard_scores]
        medium_scores = [float(value) for value in medium_scores]
        positive_score = float(positive_score)
        self.positions += 1
        if include_ordering_comparisons:
            self.ordering_evaluated_positions += 1
            self.positions_with_hard += int(bool(hard_scores))
            self.positions_with_medium += int(bool(medium_scores))
            self.positions_with_both += int(bool(hard_scores and medium_scores))

            ph_gaps = [positive_score - value for value in hard_scores]
            hm_gaps = [
                hard_value - medium_value
                for hard_value in hard_scores
                for medium_value in medium_scores
            ]
            self.pos_hard_gaps.extend(ph_gaps)
            self.hard_medium_gaps.extend(hm_gaps)
            ph_strict = [value > 0.0 for value in ph_gaps]
            ph_margin = [value >= self.pos_hard_margin for value in ph_gaps]
            hm_strict = [value > 0.0 for value in hm_gaps]
            hm_margin = [value >= self.hard_medium_margin for value in hm_gaps]
            path_events = self.sample_path_events[sample_id][str(path_id)]
            path_events["pos_hard_strict"].extend(ph_strict)
            path_events["pos_hard_margin"].extend(ph_margin)
            path_events["hard_medium_strict"].extend(hm_strict)
            path_events["hard_medium_margin"].extend(hm_margin)

            if hard_scores and medium_scores:
                triples_strict = []
                triples_margin = []
                common_ph_margin = []
                common_hm_margin = []
                for hard_value in hard_scores:
                    for medium_value in medium_scores:
                        ph_satisfied = (
                            positive_score - hard_value >= self.pos_hard_margin
                        )
                        hm_satisfied = (
                            hard_value - medium_value
                            >= self.hard_medium_margin
                        )
                        common_ph_margin.append(ph_satisfied)
                        common_hm_margin.append(hm_satisfied)
                        triples_strict.append(
                            positive_score > hard_value > medium_value
                        )
                        triples_margin.append(
                            ph_satisfied and hm_satisfied
                        )
                self.common_pos_hard_margin.extend(common_ph_margin)
                self.common_hard_medium_margin.extend(common_hm_margin)
                self.triple_strict.extend(triples_strict)
                self.triple_margin.extend(triples_margin)
                self.position_joint_strict.append(all(triples_strict))
                self.position_joint_margin.append(all(triples_margin))
                path_events["common_pos_hard_margin"].extend(
                    common_ph_margin
                )
                path_events["common_hard_medium_margin"].extend(
                    common_hm_margin
                )
                path_events["triple_strict"].extend(triples_strict)
                path_events["triple_margin"].extend(triples_margin)

        if add_to_level:
            level_bucket = self.by_level.setdefault(
                int(level),
                OrderingAccumulator(
                    pos_hard_margin=self.pos_hard_margin,
                    hard_medium_margin=self.hard_medium_margin,
                ),
            )
            level_bucket.add(
                sample_id=sample_id,
                path_id=path_id,
                level=level,
                positive_score=positive_score,
                hard_scores=hard_scores,
                medium_scores=medium_scores,
                include_ordering_comparisons=include_ordering_comparisons,
                add_to_level=False,
            )

    @staticmethod
    def _gap_summary(values: Sequence[float], margin: float) -> Dict[str, Any]:
        return {
            "pairs": len(values),
            "strict_rate": (
                sum(value > 0.0 for value in values) / len(values)
                if values
                else None
            ),
            "margin": float(margin),
            "margin_rate": (
                sum(value >= margin for value in values) / len(values)
                if values
                else None
            ),
            "gap_mean": _mean(values),
            "gap_median": percentile(values, 0.5),
            "gap_q1": percentile(values, 0.25),
            "gap_q3": percentile(values, 0.75),
        }

    def summary(self, *, include_levels: bool = True) -> Dict[str, Any]:
        sample_macro: Dict[str, Any] = {}
        event_names = (
            "pos_hard_strict",
            "pos_hard_margin",
            "hard_medium_strict",
            "hard_medium_margin",
            "triple_strict",
            "triple_margin",
            "common_pos_hard_margin",
            "common_hard_medium_margin",
        )
        for event_name in event_names:
            per_sample = []
            for paths in self.sample_path_events.values():
                per_path = [
                    _rate(events[event_name])
                    for events in paths.values()
                    if events.get(event_name)
                ]
                per_path = [value for value in per_path if value is not None]
                if per_path:
                    per_sample.append(float(_mean(per_path)))
            sample_macro[event_name] = {
                "samples": len(per_sample),
                "mean_rate": _mean(
                    [value for value in per_sample if value is not None]
                ),
            }
        common_sample_macro = {
            "pos_hard_margin": sample_macro.pop("common_pos_hard_margin"),
            "hard_medium_margin": sample_macro.pop(
                "common_hard_medium_margin"
            ),
            "triple_margin": dict(sample_macro["triple_margin"]),
        }
        common_sample_counts = {
            int(metric["samples"])
            for metric in common_sample_macro.values()
        }
        if len(common_sample_counts) != 1:
            raise AssertionError(
                "Common-support PH, HM, and Triple sample counts diverged"
            )
        if not (
            len(self.common_pos_hard_margin)
            == len(self.common_hard_medium_margin)
            == len(self.triple_margin)
        ):
            raise AssertionError(
                "Common-support PH, HM, and Triple tuple counts diverged"
            )

        common_rates = {
            name: metric["mean_rate"]
            for name, metric in common_sample_macro.items()
        }
        tolerance = 1e-12
        if common_rates["triple_margin"] is None:
            bounds = {
                "triple_le_min_marginals": None,
                "triple_ge_frechet_lower_bound": None,
            }
        else:
            ph_rate = float(common_rates["pos_hard_margin"])
            hm_rate = float(common_rates["hard_medium_margin"])
            triple_rate = float(common_rates["triple_margin"])
            upper_ok = triple_rate <= min(ph_rate, hm_rate) + tolerance
            lower_ok = (
                triple_rate + tolerance
                >= max(0.0, ph_rate + hm_rate - 1.0)
            )
            if not upper_ok or not lower_ok:
                raise AssertionError(
                    "Common-support conjunction bounds failed: "
                    f"PH={ph_rate}, HM={hm_rate}, Triple={triple_rate}"
                )
            bounds = {
                "triple_le_min_marginals": upper_ok,
                "triple_ge_frechet_lower_bound": lower_ok,
            }
        result: Dict[str, Any] = {
            "hierarchy_valid_positions": self.positions,
            # v2 compatibility alias.  Unlike v1, this means hierarchy-valid
            # (same-level) positions; see valid_positions_definition emitted
            # by evaluate_ordering.py.
            "valid_positions": self.positions,
            "ordering_evaluated_positions": self.ordering_evaluated_positions,
            "positions_with_hard": self.positions_with_hard,
            "positions_with_medium": self.positions_with_medium,
            "positions_with_both": self.positions_with_both,
            "pos_hard": self._gap_summary(
                self.pos_hard_gaps,
                self.pos_hard_margin,
            ),
            "hard_medium": self._gap_summary(
                self.hard_medium_gaps,
                self.hard_medium_margin,
            ),
            "triple_pairs": {
                "pairs": len(self.triple_strict),
                "strict_rate": _rate(self.triple_strict),
                "margin_rate": _rate(self.triple_margin),
            },
            "joint_position": {
                "positions": len(self.position_joint_strict),
                "all_pairs_strict_rate": _rate(self.position_joint_strict),
                "all_pairs_margin_rate": _rate(self.position_joint_margin),
            },
            "sample_macro_reduction": (
                "mean comparisons within path, mean paths within sample, "
                "mean across samples"
            ),
            "sample_macro": sample_macro,
            "common_triple_support": {
                "definition": (
                    "PH, HM, and Triple are evaluated on the identical "
                    "(sample, path, position, hard, medium) tuples; PH is "
                    "replicated once per hard-medium pair"
                ),
                "tuple_pairs": len(self.triple_margin),
                "samples": next(iter(common_sample_counts)),
                "sample_macro_reduction": (
                    "mean tuple events within path, mean paths within "
                    "sample, mean across samples"
                ),
                "sample_macro": common_sample_macro,
                "bounds": {**bounds, "tolerance": tolerance},
            },
        }
        if include_levels:
            result["by_level"] = {
                str(level): bucket.summary(include_levels=False)
                for level, bucket in sorted(self.by_level.items())
            }
        return result

    def sample_rows(self) -> Iterable[Dict[str, Any]]:
        event_names = (
            "pos_hard_strict",
            "pos_hard_margin",
            "hard_medium_strict",
            "hard_medium_margin",
            "triple_strict",
            "triple_margin",
            "common_pos_hard_margin",
            "common_hard_medium_margin",
        )
        for sample_id, paths in sorted(self.sample_path_events.items()):
            row: Dict[str, Any] = {"sample_id": sample_id}
            row["paths_with_comparisons"] = len(paths)
            for name in event_names:
                comparison_count = sum(
                    len(events.get(name, [])) for events in paths.values()
                )
                path_rates = [
                    _rate(events[name])
                    for events in paths.values()
                    if events.get(name)
                ]
                path_rates = [
                    value for value in path_rates if value is not None
                ]
                row[f"{name}_comparisons"] = comparison_count
                row[f"{name}_path_balanced_rate"] = _mean(path_rates)
            yield row
