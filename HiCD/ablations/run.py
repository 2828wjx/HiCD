"""Plan and execute the controlled ablation matrix.

The default mode is dry-run: commands are printed but training is not started.
Pass ``--execute`` only after reviewing the plan and the selected config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("input", "structure", "loss")


@dataclass(frozen=True)
class Experiment:
    name: str
    groups: tuple[str, ...]
    description: str
    overrides: Mapping[str, Any]
    extra_flags: tuple[str, ...] = ()
    single_auxiliary: str | None = None


EXPERIMENTS: Dict[str, Experiment] = {
    "incremental_none": Experiment(
        name="incremental_none", groups=("paper_incremental",),
        description="Table 2: no HiCL terms; CRD and remaining settings unchanged",
        overrides={"lambda_same_level": 0.0, "lambda_pos_hard": 0.0, "lambda_hard_medium": 0.0},
    ),
    "incremental_hn": Experiment(
        name="incremental_hn", groups=("paper_incremental",),
        description="Table 2: node discrimination; CRD unchanged",
        overrides={"lambda_pos_hard": 0.0, "lambda_hard_medium": 0.0},
    ),
    "incremental_hn_ph": Experiment(
        name="incremental_hn_ph", groups=("paper_incremental",),
        description="Table 2: node discrimination + positive-hard; CRD unchanged",
        overrides={"lambda_hard_medium": 0.0},
    ),
    "full": Experiment(
        name="full",
        groups=("input", "structure", "loss"),
        description=(
            "code+description, hierarchy losses, and classifier/prototype "
            "geometry alignment"
        ),
        overrides={"input_mode": "code_desc"},
    ),
    "code_only": Experiment(
        name="code_only",
        groups=("input",),
        description="complete method with code-only input",
        overrides={"input_mode": "code"},
    ),
    "desc_only": Experiment(
        name="desc_only",
        groups=("input",),
        description="complete method with description-only input",
        overrides={"input_mode": "desc"},
    ),
    "no_level_embedding": Experiment(
        name="no_level_embedding",
        groups=("structure",),
        description="set q_l=0 while retaining the hierarchy branch/loss",
        overrides={"input_mode": "code_desc"},
        extra_flags=("--no_level_embeddings",),
    ),
    "no_alignment": Experiment(
        name="no_alignment",
        groups=("loss",),
        description=(
            "retain all hierarchy losses but remove classifier/prototype "
            "geometry alignment"
        ),
        overrides={
            "input_mode": "code_desc",
            "lambda_geometry_alignment": 0.0,
        },
    ),
    "random_alignment": Experiment(
        name="random_alignment",
        groups=("loss",),
        description=(
            "retain the complete method but replace semantic CWE/prototype "
            "matching with one fixed random derangement"
        ),
        overrides={
            "input_mode": "code_desc",
            "geometry_alignment_mode": "random_fixed",
        },
    ),
    "flat": Experiment(
        name="flat",
        groups=("structure", "loss"),
        description="L_final only; functional removal of hierarchy supervision",
        overrides={
            "input_mode": "code_desc",
            "lambda_same_level": 0.0,
            "lambda_pos_hard": 0.0,
            "lambda_hard_medium": 0.0,
            "lambda_geometry_alignment": 0.0,
        },
    ),
    "same_level_only": Experiment(
        name="same_level_only",
        groups=("loss",),
        description="L_final + L_same_level",
        overrides={
            "input_mode": "code_desc",
            "lambda_pos_hard": 0.0,
            "lambda_hard_medium": 0.0,
            "lambda_geometry_alignment": 0.0,
        },
        single_auxiliary="lambda_same_level",
    ),
    "pos_hard_only": Experiment(
        name="pos_hard_only",
        groups=("loss",),
        description="L_final + L_pos_hard",
        overrides={
            "input_mode": "code_desc",
            "lambda_same_level": 0.0,
            "lambda_hard_medium": 0.0,
            "lambda_geometry_alignment": 0.0,
        },
        single_auxiliary="lambda_pos_hard",
    ),
    "hard_medium_only": Experiment(
        name="hard_medium_only",
        groups=("loss",),
        description="L_final + L_hard_medium",
        overrides={
            "input_mode": "code_desc",
            "lambda_same_level": 0.0,
            "lambda_pos_hard": 0.0,
            "lambda_geometry_alignment": 0.0,
        },
        single_auxiliary="lambda_hard_medium",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(path: str | Path) -> tuple[Path, Dict[str, Any]]:
    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Ablation config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Ablation config must be one JSON object")
    required = {"dataset", "data_dir", "encoder", "output_root", "seeds", "common_args"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Ablation config is missing keys: {missing}")
    if config["dataset"] not in {"primevul", "megavul"}:
        raise ValueError("config.dataset must be primevul or megavul")
    if not isinstance(config["common_args"], dict):
        raise ValueError("config.common_args must be an object")
    forbidden = {"data_dir", "encoder", "output_dir", "seed"}
    overlap = sorted(forbidden & set(config["common_args"]))
    if overlap:
        raise ValueError(f"Move reserved args out of common_args: {overlap}")
    return config_path, config


def select_experiments(
    *,
    groups: Sequence[str],
    names: Sequence[str] | None,
) -> list[Experiment]:
    if names:
        unknown = sorted(set(names) - set(EXPERIMENTS))
        if unknown:
            raise ValueError(f"Unknown experiments: {unknown}")
        return [EXPERIMENTS[name] for name in names]
    selected_groups = set(groups or GROUPS)
    return [
        experiment
        for experiment in EXPERIMENTS.values()
        if selected_groups.intersection(experiment.groups)
    ]


def effective_args(
    common_args: Mapping[str, Any],
    experiment: Experiment,
) -> Dict[str, Any]:
    values = dict(common_args)
    values.update(experiment.overrides)
    if experiment.single_auxiliary is not None:
        if experiment.single_auxiliary not in common_args:
            raise ValueError(
                f"common_args must define {experiment.single_auxiliary} "
                f"for {experiment.name}"
            )
        values[experiment.single_auxiliary] = common_args[
            experiment.single_auxiliary
        ]
    return values


def args_to_cli(values: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        flag = f"--{key}"
        if value is True:
            result.append(flag)
        elif isinstance(value, list):
            result.append(flag)
            result.extend(str(item) for item in value)
        else:
            result.extend((flag, str(value)))
    return result


def command_text(command: Sequence[str]) -> str:
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def dataset_metadata(data_dir: Path) -> Dict[str, Any]:
    names = (
        "dataset_config.json",
        "cwe_vocab.json",
        "cwe_relation_index.json",
        "final_class_prior.json",
        "artifact_manifest.json",
    )
    result: Dict[str, Any] = {}
    for name in names:
        path = data_dir / name
        if path.is_file():
            result[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return result


def build_jobs(
    config_path: Path,
    config: Mapping[str, Any],
    experiments: Sequence[Experiment],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    dataset = str(config["dataset"])
    train_module = (
        "method.train_primevul"
        if dataset == "primevul"
        else "method.train_megavul"
    )
    data_dir = resolve_project_path(str(config["data_dir"]))
    encoder_value = str(config["encoder"])
    encoder_path = resolve_project_path(encoder_value)
    encoder = str(encoder_path) if encoder_path.exists() else encoder_value
    output_root = resolve_project_path(str(config["output_root"]))
    common_args = dict(config["common_args"])
    warnings: list[str] = []
    if common_args.get("code_max_length") is None:
        warnings.append(
            "code_max_length is unset: code-only and code+desc may expose "
            "different code-token budgets"
        )
    if not data_dir.is_dir():
        warnings.append(f"data_dir does not exist yet: {data_dir}")
    jobs = []
    for experiment in experiments:
        values = effective_args(common_args, experiment)
        for seed in seeds:
            job_root = output_root / experiment.name / f"seed_{seed}"
            command = [
                sys.executable,
                "-m",
                train_module,
                "--data_dir",
                str(data_dir),
                "--encoder",
                encoder,
                "--output_dir",
                str(job_root),
                "--seed",
                str(seed),
                *args_to_cli(values),
                *experiment.extra_flags,
            ]
            jobs.append(
                {
                    "experiment": experiment.name,
                    "groups": list(experiment.groups),
                    "description": experiment.description,
                    "seed": int(seed),
                    "job_root": str(job_root),
                    "effective_args": values,
                    "extra_flags": list(experiment.extra_flags),
                    "command": command,
                    "command_text": command_text(command),
                }
            )
    return {
        "schema_version": "cwe-ablation-plan-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset": dataset,
        "data_dir": str(data_dir),
        "encoder": encoder,
        "output_root": str(output_root),
        "dataset_metadata": dataset_metadata(data_dir) if data_dir.is_dir() else {},
        "warnings": warnings,
        "jobs": jobs,
    }


def completed_run_directories(job_root: Path) -> set[Path]:
    if not job_root.is_dir():
        return set()
    return {
        path.parent.resolve()
        for path in job_root.glob("*/summary.json")
        if path.is_file()
    }


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def execute_jobs(plan: Mapping[str, Any], *, continue_on_error: bool) -> None:
    for index, raw_job in enumerate(plan["jobs"], start=1):
        job = dict(raw_job)
        job_root = Path(job["job_root"])
        job_root.mkdir(parents=True, exist_ok=True)
        before = completed_run_directories(job_root)
        launch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        manifest_path = job_root / f"ablation_launch_{launch_id}.json"
        manifest = {
            **job,
            "plan_config_sha256": plan["config_sha256"],
            "dataset_metadata": plan["dataset_metadata"],
            "status": "running",
            "started_at": datetime.now().astimezone().isoformat(),
        }
        save_json(manifest_path, manifest)
        print(
            f"[{index}/{len(plan['jobs'])}] {job['experiment']} "
            f"seed={job['seed']}",
            flush=True,
        )
        result = subprocess.run(job["command"], cwd=PROJECT_ROOT, check=False)
        after = completed_run_directories(job_root)
        created = sorted(str(path) for path in after - before)
        manifest.update(
            {
                "status": "completed" if result.returncode == 0 else "failed",
                "returncode": int(result.returncode),
                "finished_at": datetime.now().astimezone().isoformat(),
                "created_run_directories": created,
            }
        )
        save_json(manifest_path, manifest)
        if result.returncode != 0 and not continue_on_error:
            raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or execute the controlled CWE ablation matrix."
    )
    parser.add_argument("--config")
    parser.add_argument("--data-dir", help="Override the processed dataset directory")
    parser.add_argument("--encoder", help="Override the Hugging Face model ID or local model directory")
    parser.add_argument("--output-root", help="Override the root directory for run artifacts")
    parser.add_argument("--group", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--experiments", nargs="+", choices=tuple(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--set",
        nargs="+",
        metavar="KEY=JSON_VALUE",
        help=(
            "Temporary common-arg overrides, for example: "
            "--set epochs=1 max_train_batches=1"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Start training. Without this flag, only print the plan.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--plan-output")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def parse_common_overrides(values: Sequence[str] | None) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--set has an empty key: {item}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        result[key] = value
    return result


def main() -> None:
    args = parse_args()
    if args.list:
        for experiment in EXPERIMENTS.values():
            print(
                f"{experiment.name:22s} "
                f"groups={','.join(experiment.groups):16s} "
                f"{experiment.description}"
            )
        return

    if not args.config:
        raise ValueError("--config is required unless --list is used")

    config_path, config = load_config(args.config)
    config = dict(config)
    for key in ("data_dir", "encoder", "output_root"):
        override = getattr(args, key)
        if override is not None:
            config[key] = override
    config["common_args"] = dict(config["common_args"])
    common_overrides = parse_common_overrides(args.set)
    config["common_args"].update(common_overrides)
    experiments = select_experiments(
        groups=args.group,
        names=args.experiments,
    )
    seeds = args.seeds if args.seeds else [int(value) for value in config["seeds"]]
    if not seeds:
        raise ValueError("At least one seed is required")
    plan = build_jobs(config_path, config, experiments, seeds)
    plan["cli_common_overrides"] = common_overrides
    for warning in plan["warnings"]:
        print(f"WARNING: {warning}")
    for index, job in enumerate(plan["jobs"], start=1):
        print(f"\n[{index}/{len(plan['jobs'])}] {job['experiment']} seed={job['seed']}")
        print(job["command_text"])
    if args.plan_output:
        plan_output = resolve_project_path(args.plan_output)
        plan_output.parent.mkdir(parents=True, exist_ok=True)
        save_json(plan_output, plan)
        print(f"\nPlan written to: {plan_output}")
    if args.execute:
        execute_jobs(plan, continue_on_error=args.continue_on_error)
    else:
        print("\nDry-run only. Re-run with --execute after reviewing the commands.")


if __name__ == "__main__":
    main()
