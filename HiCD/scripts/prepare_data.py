"""Build HiCD inputs from separately downloaded PrimeVul/MegaVul and CWE XML."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def commands_for(args):
    raw = Path(args.raw_dir)
    processed = Path(args.output_dir)
    catalog = processed / "cwe_catalog_metadata.json"
    hierarchy = processed / "cwe" / "cwe_node_sequences.json"
    stage1 = processed / "intermediate" / args.dataset
    dataset_dir = processed / args.dataset
    def command(script, *values):
        return [sys.executable, str(ROOT / "scripts" / (script + ".py")), *map(str, values)]
    commands = [
        command("build_cwe_catalog_metadata", "--xml", args.cwe_catalog, "--out", catalog),
        command("build_cwe_sequences", "--xml", args.cwe_view, "--out", hierarchy.parent),
    ]
    common = ["--hierarchy", hierarchy, "--catalog", catalog, "--out-dir", stage1,
              "--normalization-policy", args.normalization_policy, "--min-count", args.min_count]
    if args.dataset == "primevul":
        commands += [command("prepare_primevul_stage1", *common, "--use-remain-label",
                             "--train", raw / "primevul" / "primevul_train.jsonl",
                             "--validation", raw / "primevul" / "primevul_valid.jsonl",
                             "--test", raw / "primevul" / "primevul_test.jsonl")]
    else:
        commands += [command("prepare_megavul_stage1", *common, "--group-by-cve", "--seed", args.seed,
                             "--rare-label-policy", "remain",
                             "--input", raw / "megavul" / "cve_with_graph_abstract_commit.json")]
    commands += [
        command(f"build_{args.dataset}_hier_stage3", "--stage1-dir", stage1,
                "--hierarchy", stage1 / f"{args.dataset}_dataset_cwe_node_sequences.json",
                "--out-dir", dataset_dir, "--max-path-len", 0),
        command("build_cwe_relation_stage4", "--dataset-dir", dataset_dir,
                "--official-hierarchy", hierarchy),
        command("generate_dataset_manifest", "--dataset-dir", dataset_dir),
    ]
    return commands


def verify_cwe_sources(args):
    policy = json.loads(Path(args.normalization_policy).read_text(encoding="utf-8"))
    for path, key in [(args.cwe_catalog, "catalog_sha256"), (args.cwe_view, "view_1000_sha256")]:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Download the CWE XML first: {source}. See docs/DATA.md")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if policy.get(key) and digest != policy[key]:
            raise ValueError(f"CWE source checksum mismatch: {source}. Use the documented CWE 4.20 files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("primevul", "megavul"), required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--cwe-catalog", default="data/raw/cwe/cwec_v4.20.xml")
    parser.add_argument("--cwe-view", default="data/raw/cwe/1000.xml")
    parser.add_argument("--normalization-policy", default="config/cwe_label_normalization_v4.20.json")
    parser.add_argument("--min-count", type=int, help="Rare-label threshold (PrimeVul: 20; MegaVul: 50)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without reading input data")
    args = parser.parse_args()
    args.min_count = args.min_count if args.min_count is not None else (20 if args.dataset == "primevul" else 50)
    if args.min_count < 1:
        parser.error("--min-count must be positive")
    # All published relative paths are anchored at the repository root.
    import os
    os.chdir(ROOT)
    if not args.dry_run:
        verify_cwe_sources(args)
    for command in commands_for(args):
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        print(f"Ready: {args.output_dir}/{args.dataset}")


if __name__ == "__main__":
    main()
