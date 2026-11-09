from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ablations.metrics.ordering import OrderingAccumulator
from method.data import (
    CWEHierCollator,
    CWEJsonlDataset,
    CWEVocab,
)
from method.megavul_runtime import (
    SPLIT_FILES as MEGAVUL_SPLIT_FILES,
    build_megavul_dataset_and_loader,
)
from method.modeling import HierarchicalCWEClassifier
from method.relations import build_relation_tensors, build_level_candidate_mask_from_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIMEVUL_SPLIT_FILES = {
    "train": "primevul_hier_train.jsonl",
    "validation": "primevul_hier_validation.jsonl",
    "test": "primevul_hier_test.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_encoder(value: str) -> str:
    candidate = resolve_path(value)
    return str(candidate) if candidate.exists() else value


def resolve_checkpoint(value: str | Path) -> Path:
    path = resolve_path(value)
    if path.is_dir():
        path = path / "best_model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Unsupported checkpoint: {path}")
    return payload


def verify_checkpoint_data_compatibility(
    checkpoint: Mapping[str, Any],
    vocab: CWEVocab,
    *,
    dataset: str,
) -> None:
    """Reject accidental checkpoint/task or checkpoint/vocabulary mismatches."""
    task_type = checkpoint.get("task_type")
    megavul_task_type = "megavul_set_valued_cwe_classification"
    if dataset == "megavul" and task_type not in (None, megavul_task_type):
        raise ValueError(f"Checkpoint has incompatible task_type: {task_type}")
    if dataset == "primevul" and task_type == megavul_task_type:
        raise ValueError("A MegaVul checkpoint cannot be evaluated as PrimeVul")

    checkpoint_vocab = checkpoint.get("vocab")
    if not isinstance(checkpoint_vocab, Mapping):
        return
    for key in ("final_label_cwe2id", "level_node_cwe2id"):
        saved_mapping = checkpoint_vocab.get(key)
        data_mapping = vocab.raw.get(key)
        if saved_mapping is not None and saved_mapping != data_mapping:
            raise ValueError(
                f"Checkpoint and data directory use different {key} mappings"
            )


def move_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def create_output_dir(parent: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"ordering_metrics_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = parent / f"ordering_metrics_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def resolve_ordering_start_level(
    saved_args: Mapping[str, Any],
    effective: Mapping[str, Any],
) -> tuple[int, str, bool]:
    """Restore the training-time cutoff without rewriting legacy semantics.

    New checkpoints save ``ordering_start_level`` explicitly.  Historical
    checkpoints trained PH/HM at projected position zero, so a missing value
    must resolve to zero rather than silently making the old run look as if it
    used the new position-one cutoff.
    """

    if "ordering_start_level" in effective:
        raw_value = effective["ordering_start_level"]
        source = "effective_configuration"
        legacy_defaulted = False
    elif "ordering_start_level" in saved_args:
        raw_value = saved_args["ordering_start_level"]
        source = "args"
        legacy_defaulted = False
    else:
        raw_value = 0
        source = "legacy_default_0"
        legacy_defaulted = True
        warnings.warn(
            "Checkpoint does not save ordering_start_level; evaluating it "
            "with the legacy cutoff 0 (PH/HM enabled from projected position "
            "0). Retrain with an explicit ordering_start_level=1 for the new "
            "Level-1 same-level-only design.",
            RuntimeWarning,
            stacklevel=2,
        )

    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(
            "checkpoint ordering_start_level must be a non-negative integer"
        )
    if raw_value < 0:
        raise ValueError(
            "checkpoint ordering_start_level must be a non-negative integer"
        )
    return int(raw_value), source, legacy_defaulted


def build_primevul_loader(
    data_dir: Path,
    split: str,
    tokenizer,
    saved_args: Mapping[str, Any],
    effective: Mapping[str, Any],
    *,
    num_levels: int,
    batch_size: int,
    num_workers: int,
) -> tuple[CWEJsonlDataset, DataLoader]:
    dataset = CWEJsonlDataset(
        data_dir / PRIMEVUL_SPLIT_FILES[split],
        num_levels=num_levels,
    )
    collator = CWEHierCollator(
        tokenizer,
        input_mode=str(saved_args.get("input_mode", "code_desc")),
        encoder_input_format=str(
            effective.get(
                "encoder_input_format",
                saved_args.get("encoder_input_format", "standard"),
            )
        ),
        max_length=int(saved_args.get("max_length", 512)),
        description_max_length=int(
            saved_args.get("description_max_length", 64)
        ),
        code_max_length=saved_args.get("code_max_length"),
        remain_hierarchy_loss_weight=saved_args.get(
            "remain_hierarchy_loss_weight"
        ),
    )
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )


def add_primevul_batch(
    accumulator: OrderingAccumulator,
    scores: torch.Tensor,
    batch: Mapping[str, Any],
    hard_table: torch.Tensor,
    medium_table: torch.Tensor,
    *,
    ordering_start_level: int,
    candidate_mask=None,
) -> None:
    level_ids = batch["level_ids"].detach().cpu()
    valid_hierarchy = (
        batch["real_level_mask"].bool()
        & batch["hierarchy_loss_mask"].bool().unsqueeze(1)
        & (batch["hierarchy_loss_weight"] > 0).unsqueeze(1)
        & (batch["level_ids"] >= 0)
    ).detach().cpu()
    valid_ordering = (
        valid_hierarchy.to(batch["contrast_loss_mask"].device)
        & batch["contrast_loss_mask"].bool().unsqueeze(1)
        & batch["contrast_level_mask"].bool()
        & (batch["contrast_loss_weight"] > 0).unsqueeze(1)
    ).detach().cpu()
    scores = scores.detach().cpu()
    for batch_index, level_index in valid_hierarchy.nonzero(
        as_tuple=False
    ).tolist():
        positive_id = int(level_ids[batch_index, level_index].item())
        hard_ids = hard_table[positive_id]
        hard_ids = hard_ids[hard_ids >= 0].tolist()
        medium_ids = medium_table[positive_id]
        medium_ids = medium_ids[medium_ids >= 0].tolist()
        if candidate_mask is not None:
            hard_ids = [node for node in hard_ids if candidate_mask[level_index, node]]
            medium_ids = [node for node in medium_ids if candidate_mask[level_index, node]]
        include_ordering = bool(
            valid_ordering[batch_index, level_index]
            and level_index >= ordering_start_level
        )
        accumulator.add(
            sample_id=str(batch["sample_id"][batch_index]),
            path_id="0",
            level=int(level_index),
            positive_score=float(scores[batch_index, level_index, positive_id]),
            hard_scores=[
                float(scores[batch_index, level_index, value])
                for value in hard_ids
            ],
            medium_scores=[
                float(scores[batch_index, level_index, value])
                for value in medium_ids
            ],
            include_ordering_comparisons=include_ordering,
        )


def add_megavul_batch(
    accumulator: OrderingAccumulator,
    scores: torch.Tensor,
    batch: Mapping[str, Any],
    hard_table: torch.Tensor,
    medium_table: torch.Tensor,
    *,
    ordering_start_level: int,
    candidate_mask=None,
) -> None:
    level_ids = batch["level_ids_multi"].detach().cpu()
    valid_hierarchy = (
        batch["real_level_masks"].bool()
        & batch["hierarchy_loss_masks"].bool().unsqueeze(-1)
        & batch["path_mask"].bool().unsqueeze(-1)
        & (batch["hierarchy_loss_weights"] > 0).unsqueeze(-1)
        & (batch["level_ids_multi"] >= 0)
    ).detach().cpu()
    valid_ordering = (
        valid_hierarchy.to(batch["contrast_loss_masks"].device)
        & batch["contrast_loss_masks"].bool().unsqueeze(-1)
        & batch["contrast_level_masks"].bool()
        & (batch["contrast_loss_weights"] > 0).unsqueeze(-1)
    ).detach().cpu()
    scores = scores.detach().cpu()
    seen = set()
    for batch_index, path_index, level_index in valid_hierarchy.nonzero(
        as_tuple=False
    ).tolist():
        positive_id = int(
            level_ids[batch_index, path_index, level_index].item()
        )
        key = (batch_index, level_index, positive_id)
        if key in seen:
            continue
        seen.add(key)
        positives = set(level_ids[batch_index, :, level_index][valid_hierarchy[batch_index, :, level_index]].tolist())
        hard_ids = hard_table[positive_id]
        hard_ids = hard_ids[hard_ids >= 0].tolist()
        medium_ids = medium_table[positive_id]
        medium_ids = medium_ids[medium_ids >= 0].tolist()
        hard_ids = [node for node in hard_ids if node not in positives]
        medium_ids = [node for node in medium_ids if node not in positives]
        if candidate_mask is not None:
            hard_ids = [node for node in hard_ids if candidate_mask[level_index, node]]
            medium_ids = [node for node in medium_ids if candidate_mask[level_index, node]]
        include_ordering = bool(
            valid_ordering[batch_index, :, level_index][level_ids[batch_index, :, level_index] == positive_id].any()
            and level_index >= ordering_start_level
        )
        accumulator.add(
            sample_id=str(batch["sample_id"][batch_index]),
            path_id="positive_union",
            level=int(level_index),
            positive_score=float(scores[batch_index, level_index, positive_id]),
            hard_scores=[
                float(scores[batch_index, level_index, value])
                for value in hard_ids
            ],
            medium_scores=[
                float(scores[batch_index, level_index, value])
                for value in medium_ids
            ],
            include_ordering_comparisons=include_ordering,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure learned pos > hard > medium cosine ordering."
    )
    parser.add_argument("--dataset", choices=("primevul", "megavul"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--encoder")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    saved_args = checkpoint.get("args", {})
    effective = checkpoint.get("effective_configuration", {})
    if not isinstance(saved_args, Mapping):
        raise ValueError("Checkpoint args must be a mapping")
    if not isinstance(effective, Mapping):
        effective = {}
    checkpoint_seed = saved_args.get("seed")
    if checkpoint_seed is not None and (
        isinstance(checkpoint_seed, bool)
        or not isinstance(checkpoint_seed, int)
    ):
        raise ValueError("Checkpoint seed must be an integer when present")
    lambdas = {
        "same_level": float(saved_args.get("lambda_same_level", 0.5)),
        "pos_hard": float(saved_args.get("lambda_pos_hard", 0.2)),
        "hard_medium": float(saved_args.get("lambda_hard_medium", 0.05)),
    }
    if not any(value != 0.0 for value in lambdas.values()):
        raise ValueError(
            "Flat has no trained hierarchy representation; ordering is N/A. "
            "Use prediction_hierarchy.py for its final-label DAG errors."
        )

    data_dir_value = args.data_dir or saved_args.get("data_dir")
    encoder_value = args.encoder or saved_args.get("encoder")
    if not data_dir_value or not encoder_value:
        raise ValueError("data_dir and encoder must come from CLI or checkpoint")
    data_dir = resolve_path(str(data_dir_value))
    encoder = resolve_encoder(str(encoder_value))
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    local_files_only = bool(saved_args.get("local_files_only", False))
    if local_files_only:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        encoder,
        local_files_only=local_files_only,
    )
    vocab = CWEVocab.from_json(data_dir / "cwe_vocab.json")
    verify_checkpoint_data_compatibility(
        checkpoint,
        vocab,
        dataset=args.dataset,
    )
    data_config_path = data_dir / "dataset_config.json"
    data_config = (
        json.loads(data_config_path.read_text(encoding="utf-8"))
        if data_config_path.is_file()
        else {}
    )
    num_levels = int(
        effective.get(
            "num_levels",
            saved_args.get(
                "num_levels",
                data_config.get("max_path_len", 4),
            ),
        )
    )
    (
        ordering_start_level,
        ordering_start_level_source,
        legacy_ordering_start_level_defaulted,
    ) = resolve_ordering_start_level(saved_args, effective)
    if ordering_start_level >= num_levels:
        raise ValueError(
            "checkpoint ordering_start_level must be smaller than num_levels "
            f"({ordering_start_level} >= {num_levels})"
        )
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else saved_args.get("eval_batch_size", 16)
    )
    encoder_input_format = str(
        effective.get(
            "encoder_input_format",
            saved_args.get("encoder_input_format", "standard"),
        )
    )
    if args.dataset == "primevul":
        dataset, dataloader = build_primevul_loader(
            data_dir,
            args.split,
            tokenizer,
            saved_args,
            effective,
            num_levels=num_levels,
            batch_size=batch_size,
            num_workers=args.num_workers,
        )
    else:
        dataset, dataloader = build_megavul_dataset_and_loader(
            data_dir,
            args.split,
            tokenizer,
            vocab,
            num_levels=num_levels,
            input_mode=str(saved_args.get("input_mode", "code_desc")),
            encoder_input_format=encoder_input_format,
            max_length=int(saved_args.get("max_length", 512)),
            description_max_length=int(
                saved_args.get("description_max_length", 64)
            ),
            code_max_length=saved_args.get("code_max_length"),
            remain_hierarchy_loss_weight=saved_args.get(
                "remain_hierarchy_loss_weight"
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith("cuda"),
        )

    pooling_mode = str(
        effective.get("pooling_mode", saved_args.get("pooling_mode", "auto"))
    )
    initializer_std = effective.get(
        "initializer_std",
        saved_args.get("initializer_std"),
    )
    use_level_embeddings = bool(
        effective.get(
            "use_level_embeddings",
            saved_args.get("use_level_embeddings", True),
        )
    )
    device = torch.device(args.device)
    model = HierarchicalCWEClassifier(
        encoder,
        num_final_labels=vocab.num_final_labels,
        num_level_nodes=vocab.num_level_nodes,
        num_levels=num_levels,
        pooling_mode=pooling_mode,
        initializer_std=(
            float(initializer_std) if initializer_std is not None else None
        ),
        local_files_only=local_files_only,
        use_level_embeddings=use_level_embeddings,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    relation_path = data_dir / "cwe_relation_index.json"
    relations = build_relation_tensors(
        relation_path,
        num_level_nodes=vocab.num_level_nodes,
    )
    hard_table = relations.hard_negative_ids.cpu()
    medium_table = relations.medium_negative_ids.cpu()
    candidate_mask = build_level_candidate_mask_from_jsonl(
        data_dir / (PRIMEVUL_SPLIT_FILES["train"] if args.dataset == "primevul" else MEGAVUL_SPLIT_FILES["train"]),
        num_level_nodes=vocab.num_level_nodes, num_levels=num_levels,
    )
    accumulator = OrderingAccumulator(
        pos_hard_margin=float(saved_args.get("pos_hard_margin", 0.15)),
        hard_medium_margin=float(
            saved_args.get("hard_medium_margin", 0.05)
        ),
    )
    with torch.inference_mode():
        for step, raw_batch in enumerate(tqdm(dataloader, desc="ordering")):
            if args.max_batches is not None and step >= args.max_batches:
                break
            batch = move_to_device(raw_batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            scores = torch.einsum(
                "bld,nd->bln",
                outputs["level_reprs"],
                outputs["label_reprs"],
            )
            if args.dataset == "primevul":
                add_primevul_batch(
                    accumulator,
                    scores,
                    batch,
                    hard_table,
                    medium_table,
                    ordering_start_level=ordering_start_level,
                    candidate_mask=candidate_mask,
                )
            else:
                add_megavul_batch(
                    accumulator,
                    scores,
                    batch,
                    hard_table,
                    medium_table,
                    ordering_start_level=ordering_start_level,
                    candidate_mask=candidate_mask,
                )

    metrics = {
        "schema_version": "cwe-ordering-metrics-v2",
        "dataset": args.dataset,
        "split": args.split,
        "seed": checkpoint_seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "data_dir": str(data_dir),
        "relation_index_sha256": sha256_file(relation_path),
        "num_dataset_examples": len(dataset),
        "configured_num_levels": num_levels,
        "partial_evaluation": args.max_batches is not None,
        "max_batches": args.max_batches,
        "lambdas": lambdas,
        "use_level_embeddings": use_level_embeddings,
        "ordering_start_level": ordering_start_level,
        "ordering_start_level_source": ordering_start_level_source,
        "legacy_ordering_start_level_defaulted": (
            legacy_ordering_start_level_defaulted
        ),
        "ordering_start_level_definition": (
            "zero-based dataset-projected path position; hierarchy/same-level "
            "positions below this cutoff remain in valid_positions, while "
            "Pos-Hard, Hard-Medium, Triple, and Joint comparisons are N/A"
        ),
        "valid_positions_definition": (
            "in schema v2, valid_positions is a compatibility alias of "
            "hierarchy_valid_positions: real hierarchy positions with positive "
            "hierarchy supervision weight; ordering_evaluated_positions "
            "additionally satisfy contrast masks and ordering_start_level"
        ),
        "score_definition": "cosine(z_l, label_embedding); no softmax or temperature",
        "candidate_definition": "selected hard/medium IDs from cwe_relation_index.json",
        "level_axis_definition": (
            "zero-based position in the dataset-projected CWE path; this is "
            "not official CWE DAG depth"
        ),
        **accumulator.summary(),
    }
    output_parent = (
        resolve_path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent
    )
    output_dir = create_output_dir(output_parent)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "sample_metrics.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in accumulator.sample_rows():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved ordering metrics to: {output_dir}")


if __name__ == "__main__":
    main()
