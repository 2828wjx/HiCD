from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Tuple

import torch


GEOMETRY_ALIGNMENT_MODES = ("semantic", "random_fixed")
DEFAULT_GEOMETRY_ALIGNMENT_SEED = 2026


def _as_cpu_mapping(mapping: torch.Tensor) -> torch.LongTensor:
    value = torch.as_tensor(mapping, dtype=torch.long).detach().cpu().clone()
    if value.ndim != 1:
        raise ValueError("alignment mapping must have shape [C]")
    if torch.any(value < -1):
        raise ValueError("alignment mapping entries must be -1 or node ids")
    valid = value[value >= 0]
    if valid.unique().numel() != valid.numel():
        raise ValueError("alignment mapping must be one-to-one on valid classes")
    return value


def alignment_mapping_sha256(mapping: torch.Tensor) -> str:
    values = _as_cpu_mapping(mapping).tolist()
    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_alignment_mapping(
    semantic_mapping: torch.Tensor,
    *,
    mode: str = "semantic",
    seed: int = DEFAULT_GEOMETRY_ALIGNMENT_SEED,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Return the active mapping and its permutation over valid classes.

    ``random_fixed`` samples one derangement with an isolated CPU generator.
    It therefore leaves PyTorch's global RNG state unchanged.  Only valid node
    ids are permuted; excluded ``-1`` entries and the prototype multiset remain
    exactly the same.
    """

    if mode not in GEOMETRY_ALIGNMENT_MODES:
        raise ValueError(
            f"mode must be one of {GEOMETRY_ALIGNMENT_MODES}; got {mode!r}"
        )
    seed = int(seed)
    if seed < 0:
        raise ValueError("geometry alignment seed must be non-negative")

    semantic = _as_cpu_mapping(semantic_mapping)
    valid_class_ids = torch.nonzero(
        semantic >= 0,
        as_tuple=False,
    ).squeeze(-1)
    num_valid = int(valid_class_ids.numel())
    identity = torch.arange(num_valid, dtype=torch.long)
    if mode == "semantic":
        return semantic, identity
    if num_valid < 2:
        raise ValueError(
            "random_fixed alignment requires at least two valid classes"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = None
    for _ in range(10000):
        candidate = torch.randperm(num_valid, generator=generator)
        if torch.all(candidate != identity):
            permutation = candidate
            break
    if permutation is None:  # pragma: no cover - astronomically unlikely
        raise RuntimeError("failed to sample a random alignment derangement")

    semantic_node_ids = semantic.index_select(0, valid_class_ids)
    active = semantic.clone()
    active[valid_class_ids] = semantic_node_ids.index_select(0, permutation)

    active_node_ids = active.index_select(0, valid_class_ids)
    if not torch.equal(
        torch.sort(active_node_ids).values,
        torch.sort(semantic_node_ids).values,
    ):
        raise AssertionError("random alignment changed the prototype multiset")
    if torch.any(active_node_ids == semantic_node_ids):
        raise AssertionError("random alignment must have no fixed points")
    return active, permutation


def build_geometry_alignment_metadata(
    *,
    vocab: Any,
    semantic_mapping: torch.Tensor,
    active_mapping: torch.Tensor,
    permutation: torch.Tensor,
    correspondence_mode: str,
    permutation_seed: int,
    lambda_geometry_alignment: float,
    start_epoch: int,
    warmup_epochs: int,
) -> dict[str, Any]:
    semantic = _as_cpu_mapping(semantic_mapping)
    active = _as_cpu_mapping(active_mapping)
    if semantic.shape != active.shape:
        raise ValueError("semantic and active mappings must have the same shape")
    if not torch.equal(semantic < 0, active < 0):
        raise ValueError("random alignment must preserve excluded classes")

    valid_class_ids = torch.nonzero(
        semantic >= 0,
        as_tuple=False,
    ).squeeze(-1)
    semantic_nodes = semantic.index_select(0, valid_class_ids)
    active_nodes = active.index_select(0, valid_class_ids)
    if not torch.equal(
        torch.sort(semantic_nodes).values,
        torch.sort(active_nodes).values,
    ):
        raise ValueError("active mapping must preserve the prototype multiset")

    fixed_points = int((semantic_nodes == active_nodes).sum().item())
    if correspondence_mode == "random_fixed" and fixed_points != 0:
        raise ValueError("random_fixed mapping must be a derangement")
    if correspondence_mode == "semantic" and fixed_points != valid_class_ids.numel():
        raise ValueError("semantic mapping must not change correspondences")

    pairs = []
    for class_id in valid_class_ids.tolist():
        semantic_node_id = int(semantic[class_id].item())
        active_node_id = int(active[class_id].item())
        pairs.append(
            {
                "final_class_id": class_id,
                "final_cwe": vocab.final_label_id2cwe[class_id],
                "semantic_node_id": semantic_node_id,
                "semantic_node_cwe": vocab.level_node_id2cwe[semantic_node_id],
                "active_node_id": active_node_id,
                "active_node_cwe": vocab.level_node_id2cwe[active_node_id],
                "is_semantic_match": semantic_node_id == active_node_id,
            }
        )

    excluded = [
        vocab.final_label_id2cwe[class_id]
        for class_id, node_id in enumerate(semantic.tolist())
        if node_id < 0
    ]
    return {
        "mode": "one_way_centered_cosine_gram",
        "correspondence_mode": correspondence_mode,
        "enabled": float(lambda_geometry_alignment) > 0.0,
        "lambda": float(lambda_geometry_alignment),
        "start_epoch": int(start_epoch),
        "warmup_epochs": int(warmup_epochs),
        "detach_prototypes": True,
        "center_classifier_rows": True,
        "pair_loss": "off_diagonal_smooth_l1",
        "permutation_algorithm": (
            "torch_randperm_rejection_derangement_v1"
            if correspondence_mode == "random_fixed"
            else "identity"
        ),
        "permutation_seed": (
            int(permutation_seed)
            if correspondence_mode == "random_fixed"
            else None
        ),
        "derangement": correspondence_mode == "random_fixed",
        "num_aligned_final_labels": int(valid_class_ids.numel()),
        "num_fixed_points": fixed_points,
        "excluded_final_labels": excluded,
        "permutation": [int(value) for value in permutation.tolist()],
        "semantic_final_to_node_idx": semantic.tolist(),
        "active_final_to_node_idx": active.tolist(),
        "active_mapping_sha256": alignment_mapping_sha256(active),
        "pairs": pairs,
    }


def restore_alignment_mapping(
    semantic_mapping: torch.Tensor,
    *,
    saved_args: Mapping[str, Any],
    effective_configuration: Mapping[str, Any],
) -> torch.LongTensor:
    """Rebuild and verify the mapping used by a saved checkpoint."""

    metadata = effective_configuration.get("geometry_alignment", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    mode = str(
        saved_args.get(
            "geometry_alignment_mode",
            metadata.get("correspondence_mode", "semantic"),
        )
    )
    seed = int(
        saved_args.get(
            "geometry_alignment_seed",
            metadata.get(
                "permutation_seed",
                DEFAULT_GEOMETRY_ALIGNMENT_SEED,
            )
            if metadata.get("permutation_seed") is not None
            else DEFAULT_GEOMETRY_ALIGNMENT_SEED,
        )
    )
    rebuilt, _ = build_alignment_mapping(
        semantic_mapping,
        mode=mode,
        seed=seed,
    )

    recorded = metadata.get("active_final_to_node_idx")
    if recorded is not None:
        recorded_mapping = _as_cpu_mapping(torch.tensor(recorded))
        if not torch.equal(recorded_mapping, rebuilt):
            raise ValueError(
                "checkpoint random alignment mapping does not match its mode/seed"
            )
    recorded_hash = metadata.get("active_mapping_sha256")
    if recorded_hash is not None and str(recorded_hash) != alignment_mapping_sha256(rebuilt):
        raise ValueError("checkpoint alignment mapping hash mismatch")
    return rebuilt
