from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


CATEGORY_PRIORITY = {
    "exact": 0,
    "ancestor_descendant": 1,
    "sibling": 2,
    "cousin": 3,
    "cross_branch": 4,
    "disconnected": 5,
    "unsupported": 6,
}


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class CWEGraph:
    """Official CWE View-1000 ChildOf DAG with undirected distances."""

    def __init__(self, edges: Iterable[tuple[str, str]]) -> None:
        self.parents: Dict[str, set[str]] = defaultdict(set)
        self.children: Dict[str, set[str]] = defaultdict(set)
        self.neighbors: Dict[str, set[str]] = defaultdict(set)
        for parent, child in edges:
            parent = str(parent)
            child = str(child)
            if parent == child:
                continue
            self.parents[child].add(parent)
            self.children[parent].add(child)
            self.neighbors[parent].add(child)
            self.neighbors[child].add(parent)
            self.parents.setdefault(parent, set())
            self.children.setdefault(child, set())
        self.nodes = set(self.neighbors) | set(self.parents) | set(self.children)

    @classmethod
    def from_json(cls, path: str | Path) -> "CWEGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("CWE edge file must be a JSON list")
        edges = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if str(item.get("relation", "ChildOf")) != "ChildOf":
                continue
            if str(item.get("view_id", "1000")) != "1000":
                continue
            parent = item.get("parent")
            child = item.get("child")
            if parent and child:
                edges.append((str(parent), str(child)))
        if not edges:
            raise ValueError(f"No View-1000 ChildOf edges found in {path}")
        return cls(edges)

    @lru_cache(maxsize=None)
    def ancestors(self, node: str) -> frozenset[str]:
        if node not in self.nodes:
            return frozenset()
        visited: set[str] = set()
        pending = list(self.parents[node])
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(self.parents[current] - visited)
        return frozenset(visited)

    @lru_cache(maxsize=None)
    def component(self, node: str) -> frozenset[str]:
        if node not in self.nodes:
            return frozenset()
        visited = {node}
        pending = [node]
        while pending:
            current = pending.pop()
            for neighbor in self.neighbors[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        return frozenset(visited)

    @lru_cache(maxsize=None)
    def distance(self, source: str, target: str) -> int | None:
        if source not in self.nodes or target not in self.nodes:
            return None
        if source == target:
            return 0
        visited = {source}
        pending = deque([(source, 0)])
        while pending:
            current, distance = pending.popleft()
            for neighbor in self.neighbors[current]:
                if neighbor == target:
                    return distance + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append((neighbor, distance + 1))
        return None

    def relation(self, prediction: str, gold: str) -> str:
        if prediction == gold:
            return "exact"
        if prediction not in self.nodes or gold not in self.nodes:
            return "unsupported"
        if prediction in self.ancestors(gold) or gold in self.ancestors(prediction):
            return "ancestor_descendant"
        if self.parents[prediction] & self.parents[gold]:
            return "sibling"
        prediction_grandparents = {
            ancestor
            for parent in self.parents[prediction]
            for ancestor in self.parents[parent]
        }
        gold_grandparents = {
            ancestor
            for parent in self.parents[gold]
            for ancestor in self.parents[parent]
        }
        if prediction_grandparents & gold_grandparents:
            return "cousin"
        if gold in self.component(prediction):
            return "cross_branch"
        return "disconnected"


def extract_prediction_and_gold(
    row: Mapping[str, Any],
    id2cwe: Mapping[int, str],
) -> tuple[str, list[str]]:
    prediction = row.get("pred_cwe")
    if not prediction and row.get("pred_id") is not None:
        prediction = id2cwe.get(int(row["pred_id"]), str(row["pred_id"]))
    gold_value = row.get("final_cwe_ids")
    if isinstance(gold_value, list) and gold_value:
        gold = [str(value) for value in gold_value]
    elif row.get("label_cwe"):
        gold = [str(row["label_cwe"])]
    elif row.get("final_cwe_id"):
        gold = [str(row["final_cwe_id"])]
    elif row.get("true_cwe"):
        gold = [str(row["true_cwe"])]
    elif isinstance(row.get("label_ids"), list):
        gold = [
            id2cwe.get(int(value), str(value))
            for value in row["label_ids"]
        ]
    elif row.get("label_id") is not None:
        value = int(row["label_id"])
        gold = [id2cwe.get(value, str(value))]
    elif row.get("true_id") is not None:
        value = int(row["true_id"])
        gold = [id2cwe.get(value, str(value))]
    else:
        gold = []
    if not prediction:
        raise ValueError(f"Prediction row has no predicted CWE: {row.get('sample_id')}")
    if not gold:
        raise ValueError(f"Prediction row has no final gold CWE: {row.get('sample_id')}")
    return str(prediction), list(dict.fromkeys(gold))


def classify_prediction(
    graph: CWEGraph,
    prediction: str,
    gold_values: Sequence[str],
) -> Dict[str, Any]:
    if prediction in set(gold_values):
        graph_evaluable = prediction in graph.nodes
        return {
            "hierarchy_error_category": "exact",
            "hierarchy_distance": 0 if graph_evaluable else None,
            "matched_gold_cwe": prediction,
            "graph_evaluable": graph_evaluable,
        }
    if prediction not in graph.nodes:
        return {
            "hierarchy_error_category": "unsupported",
            "hierarchy_distance": None,
            "matched_gold_cwe": None,
            "graph_evaluable": False,
        }
    supported_gold = [value for value in gold_values if value in graph.nodes]
    if not supported_gold:
        return {
            "hierarchy_error_category": "unsupported",
            "hierarchy_distance": None,
            "matched_gold_cwe": None,
            "graph_evaluable": False,
        }
    candidates = []
    for gold in supported_gold:
        category = graph.relation(prediction, gold)
        distance = graph.distance(prediction, gold)
        sort_distance = distance if distance is not None else math.inf
        candidates.append(
            (
                sort_distance,
                CATEGORY_PRIORITY[category],
                gold,
                category,
                distance,
            )
        )
    _, _, matched_gold, category, distance = min(candidates)
    return {
        "hierarchy_error_category": category,
        "hierarchy_distance": distance,
        "matched_gold_cwe": matched_gold,
        "graph_evaluable": True,
    }


def summarize_hierarchy_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    observed_counts = Counter(
        str(row["hierarchy_error_category"]) for row in rows
    )
    category_counts = {
        name: int(observed_counts.get(name, 0))
        for name in CATEGORY_PRIORITY
    }
    distances = [
        float(row["hierarchy_distance"])
        for row in rows
        if row.get("hierarchy_distance") is not None
    ]
    evaluable = sum(bool(row.get("graph_evaluable")) for row in rows)
    total = len(rows)
    return {
        "num_examples": total,
        "graph_evaluable_examples": evaluable,
        "graph_coverage": evaluable / total if total else 0.0,
        "category_counts": category_counts,
        "category_rates_all_examples": {
            key: value / total if total else 0.0
            for key, value in category_counts.items()
        },
        "distance_examples": len(distances),
        "distance_mean": sum(distances) / len(distances) if distances else None,
        "distance_median": percentile(distances, 0.5),
        "distance_q1": percentile(distances, 0.25),
        "distance_q3": percentile(distances, 0.75),
    }
