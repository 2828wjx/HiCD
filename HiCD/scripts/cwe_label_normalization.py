import hashlib
import json
import re
from pathlib import Path

from cwe_sequence_utils import cwe_sort_key


CWE_PATTERN = re.compile(r"^CWE-\d+$")
SUPPORTED_ACTIONS = {"drop", "remap"}


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_cwe_ids(value):
    if not isinstance(value, list):
        value = [value]
    return sorted(
        {
            str(item).strip()
            for item in value
            if CWE_PATTERN.fullmatch(str(item).strip())
        },
        key=cwe_sort_key,
    )


def validate_policy(policy, catalog, project_root=None):
    required = {
        "policy_version",
        "catalog_version",
        "strict_deprecated_coverage",
        "rules",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"Normalization policy is missing fields: {missing}")

    rules = policy["rules"]
    if not isinstance(rules, dict):
        raise ValueError("Normalization policy rules must be an object")

    for source, rule in rules.items():
        if not CWE_PATTERN.fullmatch(source):
            raise ValueError(f"Invalid normalization source CWE: {source}")
        if source not in catalog:
            raise ValueError(f"Normalization source is absent from catalog: {source}")
        if catalog[source].get("status") != "Deprecated":
            raise ValueError(
                f"Normalization source is not Deprecated in the catalog: {source}"
            )
        action = rule.get("action")
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action for {source}: {action}")
        target = rule.get("target")
        if action == "drop" and target is not None:
            raise ValueError(f"Drop rule must not define a target: {source}")
        if action == "remap":
            if not isinstance(target, str) or not CWE_PATTERN.fullmatch(target):
                raise ValueError(f"Remap rule has invalid target: {source}")
            if target not in catalog:
                raise ValueError(f"Remap target is absent from catalog: {target}")
            if catalog[target].get("status") == "Deprecated":
                raise ValueError(f"Remap target is also Deprecated: {target}")

    if project_root is not None:
        project_root = Path(project_root)
        for source_key, hash_key in (
            ("catalog_source", "catalog_sha256"),
            ("view_1000_source", "view_1000_sha256"),
        ):
            source = policy.get(source_key)
            expected = str(policy.get(hash_key) or "").lower()
            if not source or not expected:
                continue
            source_path = project_root / source
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Normalization policy source is missing: {source_path}"
                )
            actual = file_sha256(source_path)
            if actual != expected:
                raise ValueError(
                    f"Normalization policy hash mismatch for {source}: "
                    f"expected {expected}, got {actual}"
                )


def apply_cwe_policy(cwe_ids, catalog, policy):
    normalized = []
    actions = []
    rules = policy["rules"]
    strict = bool(policy.get("strict_deprecated_coverage", True))

    for cwe_id in normalize_cwe_ids(cwe_ids):
        metadata = catalog.get(cwe_id)
        if metadata is None:
            raise ValueError(f"CWE is absent from the current catalog: {cwe_id}")

        rule = rules.get(cwe_id)
        is_deprecated = metadata.get("status") == "Deprecated"
        if is_deprecated and rule is None and strict:
            raise ValueError(
                "Deprecated CWE has no explicit normalization rule: "
                f"{cwe_id}"
            )
        if rule is None:
            normalized.append(cwe_id)
            continue

        action = rule["action"]
        event = {
            "source": cwe_id,
            "action": action,
            "target": rule.get("target"),
            "reason": rule.get("reason", ""),
        }
        actions.append(event)
        if action == "remap":
            normalized.append(rule["target"])

    return sorted(set(normalized), key=cwe_sort_key), actions


def policy_manifest(policy, policy_path):
    return {
        "policy_file": str(policy_path).replace("\\", "/"),
        "policy_version": policy["policy_version"],
        "catalog_version": policy["catalog_version"],
        "catalog_release_date": policy.get("catalog_release_date"),
        "catalog_source": policy.get("catalog_source"),
        "catalog_sha256": policy.get("catalog_sha256"),
        "view_1000_source": policy.get("view_1000_source"),
        "view_1000_sha256": policy.get("view_1000_sha256"),
        "scope": policy.get("scope", ""),
        "strict_deprecated_coverage": bool(
            policy.get("strict_deprecated_coverage", True)
        ),
        "rules": policy["rules"],
    }
