from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers import set_seed as set_transformers_seed


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from method.data import CWEVocab
from method.geometry_alignment import (
    DEFAULT_GEOMETRY_ALIGNMENT_SEED,
    GEOMETRY_ALIGNMENT_MODES,
    build_alignment_mapping,
    build_geometry_alignment_metadata,
)
from HiCD.method.losses import PaperHierarchicalCWELoss as MultiLabelHierarchicalCWELoss
from method.megavul_runtime import (
    SPLIT_FILES,
    append_jsonl,
    build_megavul_dataset_and_loader,
    evaluate_megavul,
    load_final_class_weights,
    move_to_device,
    save_json,
    save_jsonl,
)
from method.modeling import HierarchicalCWEClassifier
from method.relations import (
    build_level_candidate_mask_from_jsonl,
    build_relation_tensors,
)
from method.schedules import delayed_linear_warmup_scale


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MegaVul with set-valued final labels and CWE paths."
    )
    parser.add_argument(
        "--data_dir",
        default=str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "megavul"
        ),
    )
    parser.add_argument("--encoder", default="microsoft/codebert-base")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "checkpoints" / "megavul_hier_method"),
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--num_levels",
        type=int,
        default=None,
        help="Defaults to dataset_config.json max_path_len (5 for MegaVul).",
    )
    parser.add_argument(
        "--input_mode",
        choices=["code", "code_desc", "desc"],
        default="code_desc",
    )
    parser.add_argument(
        "--encoder_input_format",
        choices=["standard", "unixcoder"],
        default="standard",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--description_max_length", type=int, default=64)
    parser.add_argument("--code_max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--head_lr",
        type=float,
        default=None,
        help="Defaults to five times --lr.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--pooling_mode",
        choices=["auto", "cls", "mean"],
        default="auto",
    )
    parser.add_argument("--initializer_std", type=float, default=None)
    parser.set_defaults(use_level_embeddings=True)
    parser.add_argument(
        "--no_level_embeddings",
        dest="use_level_embeddings",
        action="store_false",
        help="Ablation: set every level query q_l to zero.",
    )

    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_same_level", type=float, default=0.5)
    parser.add_argument("--lambda_pos_hard", type=float, default=0.2)
    parser.add_argument("--lambda_hard_medium", type=float, default=0.05)
    parser.add_argument(
        "--lambda_geometry_alignment",
        type=float,
        default=0.0,
        help=(
            "Weight of one-way centered Gram alignment between the final "
            "classifier and hierarchy prototypes; 0 disables it."
        ),
    )
    parser.add_argument(
        "--geometry_alignment_mode",
        choices=GEOMETRY_ALIGNMENT_MODES,
        default="semantic",
        help=(
            "semantic aligns each final CWE to its same-ID hierarchy node; "
            "random_fixed uses one seeded fixed derangement as a control."
        ),
    )
    parser.add_argument(
        "--geometry_alignment_seed",
        type=int,
        default=DEFAULT_GEOMETRY_ALIGNMENT_SEED,
        help="Independent seed for the random_fixed correspondence.",
    )
    parser.add_argument(
        "--alignment_start_epoch",
        type=int,
        default=4,
        help="First epoch with a non-zero geometry alignment scale.",
    )
    parser.add_argument(
        "--alignment_warmup_epochs",
        type=int,
        default=3,
        help="Linear alignment ramp length after --alignment_start_epoch.",
    )
    parser.add_argument("--pos_hard_margin", type=float, default=0.15)
    parser.add_argument("--hard_medium_margin", type=float, default=0.05)
    parser.add_argument(
        "--ordering_start_level",
        type=int,
        default=1,
        help=(
            "Zero-based projected path position at which Pos-Hard and "
            "Hard-Medium supervision starts. The default 1 keeps position "
            "0 in same-level supervision but excludes it from ordering."
        ),
    )
    parser.add_argument("--aux_warmup_epochs", type=int, default=3)
    parser.add_argument("--use_final_class_weights", action="store_true")
    parser.add_argument("--final_class_weight_power", type=float, default=1.0)
    parser.add_argument(
        "--remain_hierarchy_loss_weight",
        type=float,
        default=None,
    )
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.set_defaults(implementation_version="hicd-paper-v1")
    return parser.parse_args()


def build_optimizer(
    model: HierarchicalCWEClassifier,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> AdamW:
    buckets: Dict[Tuple[str, bool], List[torch.nn.Parameter]] = {
        ("encoder", True): [],
        ("encoder", False): [],
        ("head", True): [],
        ("head", False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        module_group = "encoder" if name.startswith("encoder.") else "head"
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        buckets[(module_group, use_decay)].append(parameter)

    learning_rates = {"encoder": float(encoder_lr), "head": float(head_lr)}
    groups = []
    for (module_group, use_decay), parameters in buckets.items():
        if not parameters:
            continue
        groups.append(
            {
                "params": parameters,
                "lr": learning_rates[module_group],
                "weight_decay": float(weight_decay) if use_decay else 0.0,
                "module_group": module_group,
            }
        )
    return AdamW(groups)


def get_module_learning_rates(optimizer: AdamW) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for group in optimizer.param_groups:
        values.setdefault(
            str(group.get("module_group", "unknown")),
            float(group["lr"]),
        )
    return values


def validate_args(args: argparse.Namespace, data_dir: Path) -> Dict[str, Any]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"MegaVul data directory not found: {data_dir}")
    required = [
        *SPLIT_FILES.values(),
        "dataset_config.json",
        "cwe_vocab.json",
        "cwe_relation_index.json",
        "final_class_prior.json",
    ]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"MegaVul data files are missing: {missing}")

    config = json.loads(
        (data_dir / "dataset_config.json").read_text(encoding="utf-8")
    )
    configured_levels = int(config["max_path_len"])
    if args.num_levels is None:
        args.num_levels = configured_levels
    elif int(args.num_levels) != configured_levels:
        raise ValueError(
            f"--num_levels={args.num_levels} differs from dataset "
            f"max_path_len={configured_levels}"
        )

    positive_values = {
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "max_length": args.max_length,
        "description_max_length": args.description_max_length,
        "temperature": args.temperature,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.head_lr is not None and args.head_lr <= 0:
        raise ValueError("--head_lr must be positive")
    if args.code_max_length is not None and args.code_max_length <= 0:
        raise ValueError("--code_max_length must be positive")
    if args.initializer_std is not None and args.initializer_std <= 0:
        raise ValueError("--initializer_std must be positive")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("--max_train_batches must be positive")
    if args.max_eval_batches is not None and args.max_eval_batches <= 0:
        raise ValueError("--max_eval_batches must be positive")
    if (
        args.aux_warmup_epochs < 0
        or args.alignment_warmup_epochs < 0
        or args.num_workers < 0
    ):
        raise ValueError("warmup epochs and workers must be non-negative")
    if args.alignment_start_epoch < 1:
        raise ValueError("--alignment_start_epoch must be at least 1")
    if args.geometry_alignment_seed < 0:
        raise ValueError("--geometry_alignment_seed must be non-negative")
    if not 0 <= args.ordering_start_level < args.num_levels:
        raise ValueError(
            "--ordering_start_level must be in [0, --num_levels)"
        )
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup_ratio must be between 0 and 1")

    non_negative_values = {
        "weight_decay": args.weight_decay,
        "lambda_same_level": args.lambda_same_level,
        "lambda_pos_hard": args.lambda_pos_hard,
        "lambda_hard_medium": args.lambda_hard_medium,
        "lambda_geometry_alignment": args.lambda_geometry_alignment,
        "pos_hard_margin": args.pos_hard_margin,
        "hard_medium_margin": args.hard_medium_margin,
        "final_class_weight_power": args.final_class_weight_power,
    }
    if args.remain_hierarchy_loss_weight is not None:
        non_negative_values["remain_hierarchy_loss_weight"] = (
            args.remain_hierarchy_loss_weight
        )
    for name, value in non_negative_values.items():
        if value < 0:
            raise ValueError(f"--{name} must be non-negative")
    return config


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    config = validate_args(args, data_dir)
    if config.get("split_policy", {}).get("intentional_leakage_probe"):
        print("=" * 100)
        print("[WARNING] INTENTIONAL TEST-INFORMED LEAKAGE PROBE")
        print("This run is diagnostic only and is invalid for formal evaluation.")
        print("=" * 100)
    effective_head_lr = (
        float(args.head_lr)
        if args.head_lr is not None
        else float(args.lr) * 5.0
    )
    if args.encoder_input_format == "unixcoder" and args.pooling_mode == "auto":
        args.pooling_mode = "mean"
    elif args.encoder_input_format == "unixcoder" and args.pooling_mode == "cls":
        print(
            "warning: UniXcoder normally uses masked mean pooling; "
            "--pooling_mode cls is an ablation"
        )

    set_transformers_seed(args.seed)
    if args.local_files_only:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir()
    device = torch.device(args.device)

    vocab = CWEVocab.from_json(data_dir / "cwe_vocab.json")
    semantic_final_to_node_idx = vocab.build_final_to_level_node_index()
    final_to_node_idx, alignment_permutation = build_alignment_mapping(
        semantic_final_to_node_idx,
        mode=args.geometry_alignment_mode,
        seed=args.geometry_alignment_seed,
    )
    geometry_alignment = build_geometry_alignment_metadata(
        vocab=vocab,
        semantic_mapping=semantic_final_to_node_idx,
        active_mapping=final_to_node_idx,
        permutation=alignment_permutation,
        correspondence_mode=args.geometry_alignment_mode,
        permutation_seed=args.geometry_alignment_seed,
        lambda_geometry_alignment=args.lambda_geometry_alignment,
        start_epoch=args.alignment_start_epoch,
        warmup_epochs=args.alignment_warmup_epochs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.encoder,
        local_files_only=args.local_files_only,
    )
    loader_kwargs = {
        "tokenizer": tokenizer,
        "vocab": vocab,
        "num_levels": int(args.num_levels),
        "input_mode": args.input_mode,
        "encoder_input_format": args.encoder_input_format,
        "max_length": args.max_length,
        "description_max_length": args.description_max_length,
        "code_max_length": args.code_max_length,
        "remain_hierarchy_loss_weight": args.remain_hierarchy_loss_weight,
        "num_workers": args.num_workers,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    train_dataset, train_loader = build_megavul_dataset_and_loader(
        data_dir,
        "train",
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    valid_dataset, valid_loader = build_megavul_dataset_and_loader(
        data_dir,
        "validation",
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_dataset, test_loader = build_megavul_dataset_and_loader(
        data_dir,
        "test",
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    level_candidate_mask = build_level_candidate_mask_from_jsonl(
        data_dir / SPLIT_FILES["train"],
        num_level_nodes=vocab.num_level_nodes,
        num_levels=int(args.num_levels),
    )
    relation_tensors = build_relation_tensors(
        data_dir / "cwe_relation_index.json",
        num_level_nodes=vocab.num_level_nodes,
    )
    model = HierarchicalCWEClassifier(
        args.encoder,
        num_final_labels=vocab.num_final_labels,
        num_level_nodes=vocab.num_level_nodes,
        num_levels=int(args.num_levels),
        pooling_mode=args.pooling_mode,
        initializer_std=args.initializer_std,
        local_files_only=args.local_files_only,
        use_level_embeddings=args.use_level_embeddings,
        final_to_node_idx=final_to_node_idx,
    ).to(device)

    final_class_weights = None
    if args.use_final_class_weights:
        final_class_weights = load_final_class_weights(
            data_dir,
            vocab,
            power=args.final_class_weight_power,
        ).to(device)
        print(
            "using set-loss class weights: "
            f"min={final_class_weights.min().item():.4f} "
            f"max={final_class_weights.max().item():.4f} "
            f"mean={final_class_weights.mean().item():.4f}"
        )

    criterion = MultiLabelHierarchicalCWELoss(
        level_candidate_mask=level_candidate_mask,
        hard_negative_ids=relation_tensors.hard_negative_ids,
        medium_negative_ids=relation_tensors.medium_negative_ids,
        final_class_weights=final_class_weights,
        temperature=args.temperature,
        lambda_same_level=args.lambda_same_level,
        lambda_pos_hard=args.lambda_pos_hard,
        lambda_hard_medium=args.lambda_hard_medium,
        lambda_geometry_alignment=args.lambda_geometry_alignment,
        pos_hard_margin=args.pos_hard_margin,
        hard_medium_margin=args.hard_medium_margin,
        ordering_start_level=args.ordering_start_level,
    ).to(device)
    optimizer = build_optimizer(
        model,
        encoder_lr=args.lr,
        head_lr=effective_head_lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = (
        len(train_loader)
        if args.max_train_batches is None
        else min(len(train_loader), args.max_train_batches)
    )
    accumulation_steps = int(args.gradient_accumulation_steps)
    updates_per_epoch = max(math.ceil(steps_per_epoch / accumulation_steps), 1)
    total_steps = max(args.epochs * updates_per_epoch, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    print(
        "MegaVul runtime: "
        f"dataset={config.get('dataset_version')} "
        f"records={len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)} "
        f"labels={vocab.num_final_labels} levels={args.num_levels} "
        f"aligned_labels={geometry_alignment['num_aligned_final_labels']} "
        f"alignment_mode={args.geometry_alignment_mode} "
        f"alignment_seed={args.geometry_alignment_seed} "
        f"ordering_start_level={args.ordering_start_level} "
        f"encoder_lr={args.lr:.2e} head_lr={effective_head_lr:.2e}"
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    best_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.jsonl"
    save_json(
        run_dir / "args.json",
        {
            "run_id": run_id,
            "task_type": "megavul_set_valued_cwe_classification",
            "args": vars(args),
            "device": str(device),
            "dataset_version": config.get("dataset_version"),
            "num_final_labels": vocab.num_final_labels,
            "num_level_nodes": vocab.num_level_nodes,
            "effective_configuration": {
                "encoder_input_format": args.encoder_input_format,
                "pooling_mode": model.pooling_mode,
                "initializer_std": model.initializer_std,
                "use_level_embeddings": model.use_level_embeddings,
                "encoder_lr": float(args.lr),
                "head_lr": effective_head_lr,
                "num_levels": int(args.num_levels),
                "ordering_start_level": int(args.ordering_start_level),
                "final_loss": "set_probability_nll",
                "hierarchy_reduction": "path_then_sample_mean",
                "geometry_alignment": geometry_alignment,
            },
        },
    )

    for epoch in range(1, args.epochs + 1):
        train_loader.dataset.set_epoch(epoch, args.seed)
        auxiliary_scale = (
            min(epoch / float(args.aux_warmup_epochs), 1.0)
            if args.aux_warmup_epochs > 0
            else 1.0
        )
        criterion.set_auxiliary_scale(auxiliary_scale)
        alignment_scale = delayed_linear_warmup_scale(
            epoch,
            start_epoch=args.alignment_start_epoch,
            warmup_epochs=args.alignment_warmup_epochs,
        )
        criterion.set_alignment_scale(alignment_scale)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_geometry_alignment = 0.0
        completed_batches = 0
        skipped_batches = 0
        pending_batches = 0
        progress = tqdm(train_loader, desc=f"train epoch {epoch}")

        for step, raw_batch in enumerate(progress):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break
            batch = move_to_device(raw_batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["loss"]
            if not torch.isfinite(loss):
                skipped_batches += 1
                pending_batches = 0
                optimizer.zero_grad(set_to_none=True)
                progress.set_postfix(nonfinite="loss skipped")
                continue

            (loss / accumulation_steps).backward()
            pending_batches += 1
            completed_batches += 1
            running_loss += float(loss.item())
            running_geometry_alignment += float(
                loss_dict["loss_geometry_alignment"].item()
            )

            should_update = (
                pending_batches >= accumulation_steps
                or step + 1 >= steps_per_epoch
            )
            if should_update:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )
                if not torch.isfinite(grad_norm):
                    skipped_batches += pending_batches
                    pending_batches = 0
                    optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(nonfinite="grad skipped")
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending_batches = 0

            progress.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                final=f"{float(loss_dict['loss_final']):.4f}",
                same=f"{float(loss_dict['loss_same_level']):.4f}",
                ph=f"{float(loss_dict['loss_pos_hard']):.4f}",
                hm=f"{float(loss_dict['loss_hard_medium']):.4f}",
                geom=f"{float(loss_dict['loss_geometry_alignment']):.4f}",
                aux=f"{auxiliary_scale:.2f}",
                align=f"{alignment_scale:.2f}",
            )

        validation_metrics, validation_rows = evaluate_megavul(
            model,
            criterion,
            valid_loader,
            device=device,
            vocab=vocab,
            max_batches=args.max_eval_batches,
            description="validation",
        )
        train_loss = running_loss / max(completed_batches, 1)
        print(
            f"[epoch {epoch}] aux_scale={auxiliary_scale:.3f} "
            f"alignment_scale={alignment_scale:.3f} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_hit={validation_metrics['hit_accuracy']:.4f} "
            f"val_f1_macro={validation_metrics['f1_macro']:.4f} "
            f"val_mcc={validation_metrics['mcc']:.4f} "
            f"val_top3={validation_metrics['top3_hit_accuracy']:.4f}"
        )
        append_jsonl(
            history_path,
            {
                "run_id": run_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "completed_train_batches": completed_batches,
                "skipped_train_batches": skipped_batches,
                "auxiliary_scale": auxiliary_scale,
                "alignment_scale": alignment_scale,
                "train_loss_geometry_alignment": (
                    running_geometry_alignment / max(completed_batches, 1)
                ),
                "learning_rates": get_module_learning_rates(optimizer),
                "validation": validation_metrics,
            },
        )

        if validation_metrics["f1_macro"] > best_macro_f1:
            best_macro_f1 = validation_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "vocab": vocab.raw,
                    "valid_metrics": validation_metrics,
                    "run_id": run_id,
                    "best_epoch": best_epoch,
                    "task_type": "megavul_set_valued_cwe_classification",
                    "dataset_version": config.get("dataset_version"),
                    "selection_metric": "validation.f1_macro",
                    "effective_configuration": {
                        "encoder_input_format": args.encoder_input_format,
                        "pooling_mode": model.pooling_mode,
                        "initializer_std": model.initializer_std,
                        "use_level_embeddings": model.use_level_embeddings,
                        "encoder_lr": float(args.lr),
                        "head_lr": effective_head_lr,
                        "num_levels": int(args.num_levels),
                        "ordering_start_level": int(
                            args.ordering_start_level
                        ),
                        "final_loss": "set_probability_nll",
                        "hierarchy_reduction": "path_then_sample_mean",
                        "geometry_alignment": geometry_alignment,
                    },
                },
                best_path,
            )
            save_json(
                run_dir / "best_validation_metrics.json",
                {
                    **validation_metrics,
                    "run_id": run_id,
                    "selected_epoch": best_epoch,
                    "selection_metric": "validation.f1_macro",
                },
            )
            save_jsonl(
                run_dir / "best_validation_predictions.jsonl",
                validation_rows,
            )
            print(f"saved best checkpoint to {best_path}")

    if not best_path.exists():
        raise RuntimeError("training did not produce a finite checkpoint")
    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion.set_auxiliary_scale(1.0)
    criterion.set_alignment_scale(1.0)
    test_metrics, test_rows = evaluate_megavul(
        model,
        criterion,
        test_loader,
        device=device,
        vocab=vocab,
        max_batches=args.max_eval_batches,
        description="test",
    )
    print(
        f"[test] loss={test_metrics['loss']:.4f} "
        f"hit={test_metrics['hit_accuracy']:.4f} "
        f"f1_macro={test_metrics['f1_macro']:.4f} "
        f"mcc={test_metrics['mcc']:.4f} "
        f"top3={test_metrics['top3_hit_accuracy']:.4f} "
        f"mrr={test_metrics['mean_reciprocal_rank']:.4f}"
    )
    test_payload = {
        **test_metrics,
        "run_id": run_id,
        "selected_epoch": best_epoch,
        "selection_metric": "validation.f1_macro",
    }
    save_json(run_dir / "test_metrics.json", test_payload)
    save_jsonl(run_dir / "test_predictions.jsonl", test_rows)
    save_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_path),
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_macro_f1,
            "selection_metric": "validation.f1_macro",
            "dataset_version": config.get("dataset_version"),
            "effective_configuration": {
                "encoder_input_format": args.encoder_input_format,
                "pooling_mode": model.pooling_mode,
                "initializer_std": model.initializer_std,
                "use_level_embeddings": model.use_level_embeddings,
                "encoder_lr": float(args.lr),
                "head_lr": effective_head_lr,
                "num_levels": int(args.num_levels),
                "ordering_start_level": int(args.ordering_start_level),
                "final_loss": "set_probability_nll",
                "hierarchy_reduction": "path_then_sample_mean",
                "geometry_alignment": geometry_alignment,
            },
            "test_metrics": test_payload,
            "files": {
                "args": "args.json",
                "history": "history.jsonl",
                "best_model": "best_model.pt",
                "best_validation_metrics": "best_validation_metrics.json",
                "best_validation_predictions": (
                    "best_validation_predictions.jsonl"
                ),
                "test_metrics": "test_metrics.json",
                "test_predictions": "test_predictions.jsonl",
            },
        },
    )
    print(f"saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
