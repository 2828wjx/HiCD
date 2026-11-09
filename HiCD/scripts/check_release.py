"""Check that the source release contains no data, checkpoints, or large files."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".jsonl", ".zip", ".pdf"}


def main():
    errors = []
    count = total = 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in {".git", "__pycache__", ".pytest_cache", ".venv", "venv"} for part in relative.parts):
            continue
        count += 1
        total += path.stat().st_size
        if path.suffix.lower() in FORBIDDEN or path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"Remove runtime artifact from source upload: {relative}")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    for message in errors:
        print(message)
    print(f"Source release: {count} files, {total / 1024 / 1024:.2f} MiB")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
