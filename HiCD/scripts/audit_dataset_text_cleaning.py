import argparse
import json
from pathlib import Path

from text_field_cleaning import build_text_cleaning_audit


SPLIT_NAMES = ("train", "validation", "test")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def detect_files(dataset_dir):
    candidates = {
        "MegaVul": "megavul_hier_{split}.jsonl",
        "PrimeVul": "primevul_hier_{split}.jsonl",
    }
    for dataset, pattern in candidates.items():
        paths = {
            split: dataset_dir / pattern.format(split=split)
            for split in SPLIT_NAMES
        }
        if all(path.exists() for path in paths.values()):
            return dataset, paths
    raise FileNotFoundError(
        f"No complete MegaVul or PrimeVul final JSONL set in {dataset_dir}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset, paths = detect_files(dataset_dir)
    records = {
        split: load_jsonl(path) for split, path in paths.items()
    }
    audit = build_text_cleaning_audit(records)
    audit.update({
        "dataset": dataset,
        "dataset_dir": ".",
        "source_files": {
            split: path.name for split, path in paths.items()
        },
    })
    audit_path = dataset_dir / "text_cleaning_audit.json"
    save_json(audit, audit_path)

    config_path = dataset_dir / "dataset_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.setdefault("input_text", "code + cve_description")
        config["text_cleaning_policy"] = audit["policy"]
        config["text_cleaning_audit_file"] = audit_path.name
        save_json(config, config_path)

    stage3_summary_path = dataset_dir / "stage3_summary.json"
    if stage3_summary_path.exists():
        stage3_summary = json.loads(
            stage3_summary_path.read_text(encoding="utf-8")
        )
        stage3_summary["text_cleaning_audit"] = audit["total"]
        save_json(stage3_summary, stage3_summary_path)

    total = audit["total"]
    print(f"[OK] dataset: {dataset}")
    print(f"[OK] records: {total['records']}")
    print(f"[OK] code changed: {total['code_changed']}")
    print(
        "[OK] description changed: "
        f"{total['description_changed']}"
    )
    print(f"[OK] audit: {audit_path}")


if __name__ == "__main__":
    main()
