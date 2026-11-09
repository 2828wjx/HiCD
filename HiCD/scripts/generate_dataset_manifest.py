"""Generate or verify SHA256 manifests for a portable dataset directory."""

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_NAME = "artifact_manifest.json"


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_manifest(dataset_dir):
    dataset_dir = Path(dataset_dir)
    config_path = dataset_dir / "dataset_config.json"
    config = load_json(config_path) if config_path.exists() else {}
    files = []
    for path in sorted(dataset_dir.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        files.append({
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return {
        "schema_version": 1,
        "dataset_dir": ".",
        "dataset_version": config.get("dataset_version"),
        "source_dataset": config.get("source_dataset"),
        "files": files,
        "total_files": len(files),
        "total_size_bytes": sum(value["size_bytes"] for value in files),
    }


def verify_manifest(dataset_dir):
    dataset_dir = Path(dataset_dir)
    manifest = load_json(dataset_dir / MANIFEST_NAME)
    errors = []
    for entry in manifest.get("files", []):
        relative = entry.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            errors.append(f"non-portable manifest path: {relative!r}")
            continue
        path = dataset_dir / relative
        if not path.is_file():
            errors.append(f"missing file: {relative}")
            continue
        size = path.stat().st_size
        if size != int(entry["size_bytes"]):
            errors.append(
                f"size mismatch: {relative} expected={entry['size_bytes']} "
                f"actual={size}"
            )
            continue
        digest = file_sha256(path)
        if digest.lower() != str(entry["sha256"]).lower():
            errors.append(f"SHA256 mismatch: {relative}")
    if errors:
        print(f"[FAILED] {len(errors)} manifest error(s)")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(
        f"[MANIFEST VERIFIED] files={manifest['total_files']} "
        f"bytes={manifest['total_size_bytes']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    if args.verify:
        verify_manifest(dataset_dir)
        return
    manifest = build_manifest(dataset_dir)
    output = dataset_dir / MANIFEST_NAME
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[MANIFEST WRITTEN] {output} files={manifest['total_files']} "
        f"bytes={manifest['total_size_bytes']}"
    )


if __name__ == "__main__":
    main()
