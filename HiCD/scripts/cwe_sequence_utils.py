from collections import defaultdict


def cwe_sort_key(cwe_id):
    try:
        return int(str(cwe_id).replace("CWE-", ""))
    except Exception:
        return 10**18


def choose_main_sequence(paths):
    if not paths:
        return []
    return sorted(paths, key=lambda path: (-len(path), path))[0]


def _deduplicate_paths(paths):
    unique_paths = []
    seen = set()

    for path in paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(list(path))

    return sorted(unique_paths, key=lambda path: (len(path), path))


def build_dataset_hierarchy(source_hierarchy, dataset_cwes):
    """
    Project the source CWE hierarchy onto CWEs that actually occur in a dataset.

    Nodes outside dataset_cwes are removed from every path. Consecutive retained
    nodes form the induced dataset path, so every emitted path node is guaranteed
    to be a label that occurs in the dataset.
    """
    allowed_cwes = set(dataset_cwes)
    projected_paths_by_cwe = {}
    missing_hierarchy_cwes = []
    source_path_nodes = set()
    removed_path_nodes = set()
    affected_target_cwes = []

    for cwe_id in sorted(allowed_cwes, key=cwe_sort_key):
        info = source_hierarchy.get(cwe_id)
        if not info:
            missing_hierarchy_cwes.append(cwe_id)
            continue

        source_paths = info.get("all_sequences") or []
        if not source_paths and info.get("main_sequence"):
            source_paths = [info["main_sequence"]]

        projected_paths = []
        target_affected = False

        for source_path in source_paths:
            source_path = list(source_path)
            source_path_nodes.update(source_path)

            projected_path = [
                node for node in source_path
                if node in allowed_cwes
            ]
            removed_nodes = [
                node for node in source_path
                if node not in allowed_cwes
            ]

            if removed_nodes:
                target_affected = True
                removed_path_nodes.update(removed_nodes)

            if projected_path and projected_path[-1] == cwe_id:
                projected_paths.append(projected_path)

        projected_paths = _deduplicate_paths(projected_paths)
        if not projected_paths:
            missing_hierarchy_cwes.append(cwe_id)
            continue

        projected_paths_by_cwe[cwe_id] = projected_paths
        if target_affected:
            affected_target_cwes.append(cwe_id)

    parent_map = defaultdict(set)
    children_map = defaultdict(set)

    for paths in projected_paths_by_cwe.values():
        for path in paths:
            for parent, child in zip(path, path[1:]):
                parent_map[child].add(parent)
                children_map[parent].add(child)

    projected_hierarchy = {}

    for cwe_id in sorted(projected_paths_by_cwe, key=cwe_sort_key):
        source_info = source_hierarchy[cwe_id]
        paths = projected_paths_by_cwe[cwe_id]
        main_sequence = choose_main_sequence(paths)
        parents = sorted(parent_map.get(cwe_id, set()), key=cwe_sort_key)
        children = sorted(children_map.get(cwe_id, set()), key=cwe_sort_key)

        siblings = set()
        for parent in parents:
            siblings.update(children_map.get(parent, set()))
        siblings.discard(cwe_id)

        source_main_sequence = list(source_info.get("main_sequence", []))
        depths = [len(path) for path in paths]

        record = dict(source_info)
        record.update({
            "parents": parents,
            "children": children,
            "siblings": sorted(siblings, key=cwe_sort_key),
            "is_leaf": not children,
            "all_sequences": paths,
            "main_sequence": main_sequence,
            "min_depth": min(depths),
            "max_depth": max(depths),
            "num_sequences": len(paths),
            "sequence_node_scope": "dataset_cwes_only",
            "source_main_sequence": source_main_sequence,
            "removed_non_dataset_nodes_from_main_sequence": [
                node for node in source_main_sequence
                if node not in allowed_cwes
            ],
        })
        projected_hierarchy[cwe_id] = record

    validate_dataset_hierarchy(projected_hierarchy, allowed_cwes)

    report = {
        "num_dataset_cwes": len(allowed_cwes),
        "num_projected_hierarchy_cwes": len(projected_hierarchy),
        "num_missing_hierarchy_cwes": len(missing_hierarchy_cwes),
        "missing_hierarchy_cwes": sorted(
            set(missing_hierarchy_cwes),
            key=cwe_sort_key,
        ),
        "num_source_path_nodes": len(source_path_nodes),
        "num_removed_non_dataset_path_nodes": len(removed_path_nodes),
        "removed_non_dataset_path_nodes": sorted(
            removed_path_nodes,
            key=cwe_sort_key,
        ),
        "num_affected_target_cwes": len(set(affected_target_cwes)),
        "affected_target_cwes": sorted(
            set(affected_target_cwes),
            key=cwe_sort_key,
        ),
        "all_sequence_nodes_in_dataset": True,
    }

    return projected_hierarchy, report


def validate_dataset_hierarchy(hierarchy, dataset_cwes):
    allowed_cwes = set(dataset_cwes)

    for cwe_id, info in hierarchy.items():
        if cwe_id not in allowed_cwes:
            raise ValueError(
                f"Hierarchy target {cwe_id} is not in the dataset CWE set."
            )

        main_sequence = info.get("main_sequence", [])
        if not main_sequence:
            raise ValueError(f"Empty main_sequence for {cwe_id}.")
        if main_sequence[-1] != cwe_id:
            raise ValueError(
                f"main_sequence for {cwe_id} does not end at the target: "
                f"{main_sequence}"
            )

        all_sequences = info.get("all_sequences", [])
        if main_sequence not in all_sequences:
            raise ValueError(
                f"main_sequence for {cwe_id} is absent from all_sequences."
            )

        for sequence in all_sequences:
            outside_nodes = [
                node for node in sequence
                if node not in allowed_cwes
            ]
            if outside_nodes:
                raise ValueError(
                    f"Sequence for {cwe_id} contains non-dataset CWEs: "
                    f"{outside_nodes}"
                )
            if not sequence or sequence[-1] != cwe_id:
                raise ValueError(
                    f"Invalid target sequence for {cwe_id}: {sequence}"
                )

