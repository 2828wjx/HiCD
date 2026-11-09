from collections import Counter

from cwe_sequence_utils import cwe_sort_key


BACKOFF_MODE_REMAIN = "remain"
BACKOFF_MODE_NEAREST_FREQUENT_ANCESTOR = (
    "nearest-frequent-ancestor"
)
BACKOFF_MODES = (
    BACKOFF_MODE_REMAIN,
    BACKOFF_MODE_NEAREST_FREQUENT_ANCESTOR,
)


def _deduplicate(values):
    return list(dict.fromkeys(values))


def build_nearest_frequent_ancestor_map(
    hierarchy,
    label_counts,
    frequent_labels,
    source_cwes=None,
):
    """Select one deterministic nearest frequent ancestor per rare CWE."""
    counts = Counter(label_counts)
    frequent = set(frequent_labels)
    sources = set(source_cwes or counts) - frequent
    result = {}

    for source in sorted(sources, key=cwe_sort_key):
        candidates = {}
        for sequence in hierarchy.get(source, {}).get(
            "all_sequences", []
        ):
            if not sequence or sequence[-1] != source:
                continue
            for index, ancestor in enumerate(sequence[:-1]):
                if ancestor not in frequent:
                    continue
                path = sequence[index:]
                distance = len(path) - 1
                candidate = {
                    "source_cwe": source,
                    "target_cwe": ancestor,
                    "distance": distance,
                    "path": path,
                    "target_count": counts[ancestor],
                }
                previous = candidates.get(ancestor)
                if previous is None or (
                    distance,
                    tuple(path),
                ) < (
                    previous["distance"],
                    tuple(previous["path"]),
                ):
                    candidates[ancestor] = candidate

        if not candidates:
            result[source] = None
            continue

        result[source] = min(
            candidates.values(),
            key=lambda item: (
                item["distance"],
                -item["target_count"],
                cwe_sort_key(item["target_cwe"]),
                tuple(item["path"]),
            ),
        )

    return result


def assign_final_cwes(
    cwe_ids,
    frequent_labels,
    remain_label,
    backoff_mode=BACKOFF_MODE_REMAIN,
    ancestor_map=None,
):
    """Assign direct frequent targets, hierarchy backoff, or Remain."""
    if backoff_mode not in BACKOFF_MODES:
        raise ValueError(f"Unsupported hierarchy backoff mode: {backoff_mode}")

    frequent = set(frequent_labels)
    direct_targets = _deduplicate(
        cwe_id for cwe_id in cwe_ids if cwe_id in frequent
    )
    rare_cwes = _deduplicate(
        cwe_id for cwe_id in cwe_ids if cwe_id not in frequent
    )
    if direct_targets:
        return {
            "final_cwe_ids": direct_targets,
            "rare_cwe_ids": rare_cwes,
            "hierarchy_backoff_actions": [],
            "unmapped_rare_cwe_ids": [],
            "used_hierarchy_backoff": False,
        }

    actions = []
    unmapped = []
    if backoff_mode == BACKOFF_MODE_NEAREST_FREQUENT_ANCESTOR:
        ancestor_map = ancestor_map or {}
        for source in rare_cwes:
            selected = ancestor_map.get(source)
            if selected is None:
                unmapped.append(source)
                continue
            actions.append(dict(selected))

    targets = _deduplicate(
        action["target_cwe"] for action in actions
    )
    return {
        "final_cwe_ids": targets or [remain_label],
        "rare_cwe_ids": rare_cwes,
        "hierarchy_backoff_actions": actions,
        "unmapped_rare_cwe_ids": (
            unmapped if actions else list(rare_cwes)
        ),
        "used_hierarchy_backoff": bool(targets),
    }


def hierarchy_backoff_manifest(
    mode,
    ancestor_map,
    threshold,
    threshold_scope,
):
    mappings = {
        source: selected
        for source, selected in sorted(
            (ancestor_map or {}).items(),
            key=lambda item: cwe_sort_key(item[0]),
        )
        if selected is not None
    }
    unmapped = sorted(
        (
            source
            for source, selected in (ancestor_map or {}).items()
            if selected is None
        ),
        key=cwe_sort_key,
    )
    return {
        "mode": mode,
        "enabled": mode == BACKOFF_MODE_NEAREST_FREQUENT_ANCESTOR,
        "applies_only_without_direct_frequent_label": True,
        "selection_order": [
            "shortest official View-1000 ancestor distance",
            "larger cleaned-dataset target count",
            "smaller numeric CWE id",
        ],
        "threshold": threshold,
        "threshold_scope": threshold_scope,
        "num_mapped_rare_cwes": len(mappings),
        "num_unmapped_rare_cwes": len(unmapped),
        "mappings": mappings,
        "unmapped_rare_cwes": unmapped,
    }
