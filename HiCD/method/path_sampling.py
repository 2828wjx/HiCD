import hashlib
import random


def sample_supervision(supervision, repository, node_ids, num_levels, *, seed, epoch, sample_id):
    result = dict(supervision)
    cwe = result.get("cwe_id")
    entry = repository.get(cwe, {})
    paths = entry.get("all_sequences") or []
    if not paths and entry.get("main_sequence"):
        paths = [entry["main_sequence"]]
    if not paths or not result.get("hierarchy_loss_mask", 0):
        return result
    # Sorting makes results independent of JSON object and candidate-path order.
    paths = sorted({tuple(path) for path in paths})
    key = f"{seed}|{epoch}|{sample_id}|{cwe}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    path = random.Random(local_seed).choice(paths)
    if len(path) > num_levels:
        raise ValueError(f"Path for {cwe} has {len(path)} nodes, but num_levels={num_levels}")
    unknown = set(path) - set(node_ids)
    if unknown:
        raise ValueError(f"Path for {cwe} contains unknown nodes: {sorted(unknown)}")
    count = len(path)
    result["level_ids"] = [node_ids[node] for node in path] + [-100] * (num_levels - count)
    result["real_level_mask"] = [1] * count + [0] * (num_levels - count)
    result["contrast_level_mask"] = list(result["real_level_mask"])
    return result
