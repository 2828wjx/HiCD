import re

TEXT_CLEANING_POLICY_VERSION = "hicd-code-description-v2-id-sanitization"
IDENTIFIER_PATTERN = re.compile(r"\b(?:CWE\s*[-_:]\s*\d+|CVE\s*[-_:]\s*\d{4}\s*[-_:]\s*\d{4,})\b", re.IGNORECASE)


def clean_code_text(text):
    """Normalize layout noise without changing code tokens or indentation."""
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    previous_blank = False
    for line in text.split("\n"):
        line = line.rstrip()
        is_blank = line.strip() == ""
        if is_blank:
            if previous_blank:
                continue
            cleaned_lines.append("")
            previous_blank = True
        else:
            cleaned_lines.append(line)
            previous_blank = False

    return "\n".join(cleaned_lines).strip()


def clean_description_text(text):
    """Remove explicit CWE/CVE identifiers and normalize whitespace (Sec. 5.2)."""
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(IDENTIFIER_PATTERN.sub(" ", text).split()).strip()


def clean_record_text(
    record,
    code_fields=("code", "func"),
    description_fields=(
        "cve_description",
        "description",
        "commit_message",
    ),
):
    def first_non_empty(fields):
        for field in fields:
            value = record.get(field)
            if value is not None and str(value).strip():
                return str(value)
        return ""

    raw_code = first_non_empty(code_fields)
    raw_description = first_non_empty(description_fields)
    return {
        "raw_code": raw_code,
        "raw_description": raw_description,
        "code": clean_code_text(raw_code),
        "description": clean_description_text(raw_description),
    }


def text_cleaning_policy_manifest():
    return {
        "version": TEXT_CLEANING_POLICY_VERSION,
        "storage_policy": (
            "retain source text in JSONL; clean deterministically when "
            "constructing model input"
        ),
        "application_stage": "training_and_evaluation_data_loader",
        "implementation": {
            "module": "scripts.text_field_cleaning",
            "code_function": "clean_code_text",
            "description_function": "clean_description_text",
        },
        "field_priority": {
            "code": ["code", "func"],
            "description": [
                "cve_description",
                "description",
                "commit_message",
            ],
        },
        "code_rules": [
            "normalize CRLF and CR to LF",
            "remove trailing whitespace from every line",
            "collapse consecutive blank lines to at most one blank line",
            "remove leading and trailing whitespace around the full field",
        ],
        "description_rules": [
            "remove explicit CWE and CVE identifiers, case-insensitively",
            "normalize CRLF and CR to LF",
            "collapse all whitespace runs to one ASCII space",
            "remove leading and trailing whitespace",
        ],
        "explicitly_not_performed": [
            "remove code comments",
            "remove code indentation",
            "rewrite identifiers or string literals",
            "stemming, stop-word removal, or description paraphrasing",
        ],
        "idempotent": True,
    }


def build_text_cleaning_audit(
    records_by_split,
    code_fields=("code", "func"),
    description_fields=(
        "cve_description",
        "description",
        "commit_message",
    ),
):
    split_stats = {}
    for split, records in records_by_split.items():
        stats = {
            "records": 0,
            "code_changed": 0,
            "description_changed": 0,
            "code_and_description_changed": 0,
            "empty_code_before": 0,
            "empty_code_after": 0,
            "empty_description_before": 0,
            "empty_description_after": 0,
            "raw_code_characters": 0,
            "cleaned_code_characters": 0,
            "raw_description_characters": 0,
            "cleaned_description_characters": 0,
        }
        for record in records:
            text = clean_record_text(
                record,
                code_fields=code_fields,
                description_fields=description_fields,
            )
            raw_code = text["raw_code"]
            raw_description = text["raw_description"]
            code = text["code"]
            description = text["description"]
            code_changed = code != raw_code
            description_changed = description != raw_description

            stats["records"] += 1
            stats["code_changed"] += int(code_changed)
            stats["description_changed"] += int(description_changed)
            stats["code_and_description_changed"] += int(
                code_changed and description_changed
            )
            stats["empty_code_before"] += int(not raw_code)
            stats["empty_code_after"] += int(not code)
            stats["empty_description_before"] += int(
                not raw_description
            )
            stats["empty_description_after"] += int(not description)
            stats["raw_code_characters"] += len(raw_code)
            stats["cleaned_code_characters"] += len(code)
            stats["raw_description_characters"] += len(
                raw_description
            )
            stats["cleaned_description_characters"] += len(description)
        split_stats[split] = stats

    total = {}
    if split_stats:
        for key in next(iter(split_stats.values())):
            total[key] = sum(stats[key] for stats in split_stats.values())
    return {
        "policy": text_cleaning_policy_manifest(),
        "split_stats": split_stats,
        "total": total,
        "all_cleaned_codes_non_empty": (
            total.get("empty_code_after", 0) == 0
        ),
        "all_cleaned_descriptions_non_empty": (
            total.get("empty_description_after", 0) == 0
        ),
    }
