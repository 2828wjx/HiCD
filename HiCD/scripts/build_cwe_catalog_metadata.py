import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def strip_ns(tag):
    return tag.split("}", 1)[-1]


def text_of(element):
    if element is None:
        return ""
    return " ".join(element.itertext()).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument(
        "--out",
        type=str,
        default="processed/cwe_catalog_metadata.json",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml)
    out_path = Path(args.out)
    if not xml_path.exists():
        raise FileNotFoundError(f"Catalog XML not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    metadata = {}

    for element in root.iter():
        entry_type = strip_ns(element.tag)
        if entry_type not in {"Weakness", "Category", "View"}:
            continue

        raw_id = element.attrib.get("ID")
        if not raw_id:
            continue

        description = ""
        summary = ""
        mapping_usage = ""
        mapping_rationale = ""
        mapping_comments = ""
        mapping_reasons = []
        for child in element:
            child_type = strip_ns(child.tag)
            if child_type == "Description":
                description = text_of(child)
            elif child_type == "Summary":
                summary = text_of(child)
            elif child_type == "Mapping_Notes":
                for note in child:
                    note_type = strip_ns(note.tag)
                    if note_type == "Usage":
                        mapping_usage = text_of(note)
                    elif note_type == "Rationale":
                        mapping_rationale = text_of(note)
                    elif note_type == "Comments":
                        mapping_comments = text_of(note)
                    elif note_type == "Reasons":
                        mapping_reasons = sorted({
                            reason.attrib.get("Type", "")
                            for reason in note
                            if strip_ns(reason.tag) == "Reason"
                            and reason.attrib.get("Type")
                        })

        cwe_id = f"CWE-{raw_id}"
        metadata[cwe_id] = {
            "cwe_id": cwe_id,
            "raw_id": raw_id,
            "entry_type": entry_type.lower(),
            "name": element.attrib.get("Name", ""),
            "status": element.attrib.get("Status", ""),
            "abstraction": element.attrib.get("Abstraction", ""),
            "description": description or summary,
            "mapping_usage": mapping_usage,
            "mapping_reasons": mapping_reasons,
            "mapping_rationale": mapping_rationale,
            "mapping_comments": mapping_comments,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    counts = {}
    for info in metadata.values():
        entry_type = info["entry_type"]
        counts[entry_type] = counts.get(entry_type, 0) + 1

    print(f"[OK] Catalog metadata: {out_path}")
    print(f"[INFO] Entries: {len(metadata)}")
    print(f"[INFO] Types: {counts}")


if __name__ == "__main__":
    main()
