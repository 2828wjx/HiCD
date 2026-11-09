import math
from collections import Counter

from cwe_sequence_utils import cwe_sort_key


IGNORE_LEVEL_ID = -100


def classify_cwe_entry(cwe_id, hierarchy, catalog):
    catalog_info = catalog.get(cwe_id, {})
    entry_type = catalog_info.get("entry_type", "unknown")
    hierarchy_eligible = bool(
        hierarchy.get(cwe_id, {}).get("main_sequence")
    )
    return {
        "cwe_entry_type": entry_type,
        "cwe_name": catalog_info.get("name", ""),
        "cwe_status": catalog_info.get("status", ""),
        "hierarchy_eligible": hierarchy_eligible,
    }


def build_label_groups(counter, hierarchy, catalog, min_count):
    final_label_set = {
        cwe_id
        for cwe_id, count in counter.items()
        if count >= min_count
    }
    hierarchical_final_labels = {
        cwe_id for cwe_id in final_label_set if cwe_id in hierarchy
    }
    non_hierarchical_final_labels = (
        final_label_set - hierarchical_final_labels
    )
    hierarchical_rare = {
        cwe_id
        for cwe_id, count in counter.items()
        if count < min_count and cwe_id in hierarchy
    }
    excluded_non_hierarchical_rare = {
        cwe_id
        for cwe_id, count in counter.items()
        if count < min_count and cwe_id not in hierarchy
    }

    entry_metadata = {
        cwe_id: classify_cwe_entry(cwe_id, hierarchy, catalog)
        for cwe_id in counter
    }

    return {
        "final_label_set": final_label_set,
        "hierarchical_final_labels": hierarchical_final_labels,
        "non_hierarchical_final_labels": non_hierarchical_final_labels,
        "rare_train_only_cwes": hierarchical_rare,
        "excluded_non_hierarchical_rare_cwes": (
            excluded_non_hierarchical_rare
        ),
        "hierarchy_cwe_set": hierarchical_final_labels | hierarchical_rare,
        "model_cwe_set": (
            final_label_set | hierarchical_rare
        ),
        "entry_metadata": entry_metadata,
    }


def build_vocab(label_set, dataset_hierarchy):
    final_label_set = sorted(
        label_set["final_label_set"],
        key=cwe_sort_key,
    )
    rare_train_only_cwes = sorted(
        label_set["rare_train_only_cwes"],
        key=cwe_sort_key,
    )
    hierarchical_final_labels = sorted(
        label_set.get("hierarchical_final_labels", []),
        key=cwe_sort_key,
    )
    non_hierarchical_final_labels = sorted(
        label_set.get("non_hierarchical_final_labels", []),
        key=cwe_sort_key,
    )
    level_node_set = sorted(dataset_hierarchy, key=cwe_sort_key)

    final_label_cwe2id = {
        cwe: idx for idx, cwe in enumerate(final_label_set)
    }
    level_node_cwe2id = {
        cwe: idx for idx, cwe in enumerate(level_node_set)
    }

    return {
        "final_label_cwe2id": final_label_cwe2id,
        "final_label_id2cwe": {
            str(idx): cwe
            for cwe, idx in final_label_cwe2id.items()
        },
        "level_node_cwe2id": level_node_cwe2id,
        "level_node_id2cwe": {
            str(idx): cwe
            for cwe, idx in level_node_cwe2id.items()
        },
        "final_label_set": final_label_set,
        "hierarchical_final_labels": hierarchical_final_labels,
        "non_hierarchical_final_labels": non_hierarchical_final_labels,
        "rare_train_only_cwes": rare_train_only_cwes,
        "level_node_set": level_node_set,
        "num_final_labels": len(final_label_set),
        "num_hierarchical_final_labels": len(hierarchical_final_labels),
        "num_non_hierarchical_final_labels": len(
            non_hierarchical_final_labels
        ),
        "num_level_nodes": len(level_node_set),
    }


def expand_hierarchical_sequence(sequence, max_path_len):
    if not sequence:
        raise ValueError("Cannot expand an empty hierarchy sequence")
    if len(sequence) > max_path_len:
        raise ValueError(
            f"Path length {len(sequence)} exceeds "
            f"max_path_len={max_path_len}: {sequence}"
        )

    expanded = list(sequence)
    while len(expanded) < max_path_len:
        expanded.append(sequence[-1])

    real_mask = [1] * len(sequence) + [0] * (
        max_path_len - len(sequence)
    )
    return {
        "main_sequence": list(sequence),
        "expanded_sequence": expanded,
        "real_level_mask": real_mask,
        "contrast_level_mask": [1] * max_path_len,
        "is_repeated_level": [0] * len(sequence)
        + [1] * (max_path_len - len(sequence)),
        "hierarchy_loss_mask": 1,
        "contrast_loss_mask": 1,
        "path_key": "->".join(sequence),
        "path_len": len(sequence),
    }


def build_non_hierarchical_sequence(max_path_len):
    return {
        "main_sequence": [],
        "expanded_sequence": [None] * max_path_len,
        "real_level_mask": [0] * max_path_len,
        "contrast_level_mask": [0] * max_path_len,
        "is_repeated_level": [0] * max_path_len,
        "hierarchy_loss_mask": 0,
        "contrast_loss_mask": 0,
        "path_key": None,
        "path_len": 0,
    }


def build_class_prior(records_or_cwes, final_label_set):
    if records_or_cwes and isinstance(records_or_cwes[0], dict):
        cwes = [record["normalized_cwe"] for record in records_or_cwes]
    else:
        cwes = list(records_or_cwes)

    counter = Counter(cwes)
    total = len(cwes)
    raw_weights = {
        cwe_id: 1.0 / math.sqrt(max(count, 1))
        for cwe_id, count in counter.items()
    }
    mean_weight = sum(raw_weights.values()) / max(
        len(raw_weights),
        1,
    )

    result = {}
    final_label_set = set(final_label_set)
    for cwe_id, count in sorted(
        counter.items(),
        key=lambda item: cwe_sort_key(item[0]),
    ):
        result[cwe_id] = {
            "label_type": (
                "eval_label"
                if cwe_id in final_label_set
                else "train_only_rare"
            ),
            "class_count": count,
            "class_prior": count / total if total else 0,
            "class_weight": raw_weights[cwe_id] / mean_weight,
        }
    return result

