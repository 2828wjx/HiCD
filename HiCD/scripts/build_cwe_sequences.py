import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def strip_ns(tag: str) -> str:
    """Remove XML namespace."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def text_of(elem) -> str:
    """Get clean text from XML element."""
    if elem is None:
        return ""
    return " ".join(elem.itertext()).strip()


def parse_cwe_xml(xml_path: Path):
    """
    Parse CWE View-1000 XML.

    Returns:
        nodes: {
          "CWE-89": {
             "cwe_id": "CWE-89",
             "raw_id": "89",
             "name": "...",
             "description": "...",
             "extended_description": "...",
             "abstraction": "Base",
             "status": "Stable"
          }
        }

        edges: [
          {
            "parent": "CWE-943",
            "child": "CWE-89",
            "relation": "ChildOf",
            "view_id": "1000"
          }
        ]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nodes = {}
    edges = []

    for elem in root.iter():
        if strip_ns(elem.tag) != "Weakness":
            continue

        raw_id = elem.attrib.get("ID")
        if not raw_id:
            continue

        cwe_id = f"CWE-{raw_id}"

        name = elem.attrib.get("Name", "")
        abstraction = elem.attrib.get("Abstraction", "")
        status = elem.attrib.get("Status", "")

        description = ""
        extended_description = ""

        for child in elem:
            tag = strip_ns(child.tag)
            if tag == "Description":
                description = text_of(child)
            elif tag == "Extended_Description":
                extended_description = text_of(child)

        nodes[cwe_id] = {
            "cwe_id": cwe_id,
            "raw_id": raw_id,
            "name": name,
            "description": description,
            "extended_description": extended_description,
            "abstraction": abstraction,
            "status": status,
        }

        # Parse Related_Weaknesses.
        for rw in elem.iter():
            if strip_ns(rw.tag) != "Related_Weakness":
                continue

            nature = rw.attrib.get("Nature", "")
            view_id = rw.attrib.get("View_ID", "")
            target_raw_id = rw.attrib.get("CWE_ID", "")

            # Only use View-1000 ChildOf relation.
            if nature != "ChildOf":
                continue
            if view_id != "1000":
                continue
            if not target_raw_id:
                continue

            parent = f"CWE-{target_raw_id}"
            child_cwe = cwe_id

            edges.append({
                "parent": parent,
                "child": child_cwe,
                "relation": "ChildOf",
                "view_id": view_id,
            })

    return nodes, edges


def build_graph(nodes, edges):
    parent_map = defaultdict(list)
    children_map = defaultdict(list)

    for e in edges:
        parent = e["parent"]
        child = e["child"]

        # Keep only edges whose endpoints exist as Weakness nodes.
        # This avoids accidentally mixing View/Category/invalid IDs.
        if parent not in nodes or child not in nodes:
            continue

        parent_map[child].append(parent)
        children_map[parent].append(child)

    # Deduplicate and sort.
    parent_map = {k: sorted(set(v)) for k, v in parent_map.items()}
    children_map = {k: sorted(set(v)) for k, v in children_map.items()}

    return parent_map, children_map


def find_roots(nodes, parent_map, children_map):
    """
    Meaningful roots:
    nodes that appear in View-1000 hierarchy and have no parent inside View-1000.
    These are usually Pillar-level CWEs, but we do not force that.
    """
    involved = set(parent_map.keys()) | set(children_map.keys())
    roots = []

    for cwe_id in involved:
        if cwe_id not in parent_map:
            roots.append(cwe_id)

    return sorted(roots)


def dfs_paths_from_roots(roots, children_map):
    """
    Generate all root-to-node paths.

    Returns:
        node_to_paths: {
          "CWE-89": [
             ["CWE-707", "CWE-74", "CWE-943", "CWE-89"]
          ]
        }
    """
    node_to_paths = defaultdict(list)

    def dfs(cur, path):
        node_to_paths[cur].append(path[:])

        for child in children_map.get(cur, []):
            # Avoid cycle.
            if child in path:
                continue
            dfs(child, path + [child])

    for root in roots:
        dfs(root, [root])

    return dict(node_to_paths)


def choose_main_sequence(paths):
    """
    Select one main sequence.

    Policy:
    1. Prefer the longest path, because it preserves more hierarchy.
    2. If tied, choose lexicographically stable path.
    """
    if not paths:
        return []

    return sorted(paths, key=lambda p: (-len(p), p))[0]


def build_siblings(parent_map, children_map):
    siblings_map = {}

    all_nodes = set(parent_map.keys()) | set(children_map.keys())

    for node in all_nodes:
        siblings = set()

        for parent in parent_map.get(node, []):
            for child in children_map.get(parent, []):
                if child != node:
                    siblings.add(child)

        siblings_map[node] = sorted(siblings)

    return siblings_map


def build_node_sequence_records(nodes, parent_map, children_map, node_to_paths, siblings_map):
    records = {}

    for cwe_id, info in nodes.items():
        paths = node_to_paths.get(cwe_id, [])
        if not paths:
            continue

        main_sequence = choose_main_sequence(paths)
        depths = [len(p) for p in paths]

        records[cwe_id] = {
            "target_cwe": cwe_id,
            "name": info.get("name", ""),
            "abstraction": info.get("abstraction", ""),
            "status": info.get("status", ""),

            "parents": parent_map.get(cwe_id, []),
            "children": children_map.get(cwe_id, []),
            "siblings": siblings_map.get(cwe_id, []),

            "is_leaf": len(children_map.get(cwe_id, [])) == 0,

            "all_sequences": paths,
            "main_sequence": main_sequence,

            "min_depth": min(depths),
            "max_depth": max(depths),
            "num_sequences": len(paths),
        }

    return records


def write_outputs(out_dir, nodes, edges, node_records):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Node information.
    with open(out_dir / "cwe_nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)

    # 2. Edge information.
    with open(out_dir / "cwe_edges.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    # 3. Node -> sequences.
    with open(out_dir / "cwe_node_sequences.json", "w", encoding="utf-8") as f:
        json.dump(node_records, f, ensure_ascii=False, indent=2)

    # 4. Sequence JSONL.
    seq_path = out_dir / "cwe_sequences.jsonl"
    seq_id = 0

    with open(seq_path, "w", encoding="utf-8") as f:
        for cwe_id, rec in sorted(node_records.items()):
            for seq in rec["all_sequences"]:
                seq_id += 1

                names = [
                    nodes.get(x, {}).get("name", "")
                    for x in seq
                ]
                abstractions = [
                    nodes.get(x, {}).get("abstraction", "")
                    for x in seq
                ]

                item = {
                    "sequence_id": f"seq_{seq_id:06d}",
                    "target_cwe": cwe_id,
                    "sequence": seq,
                    "sequence_names": names,
                    "abstractions": abstractions,
                    "depth": len(seq),
                    "is_main_sequence": seq == rec["main_sequence"],
                    "target_name": rec["name"],
                    "target_abstraction": rec["abstraction"],
                    "target_is_leaf": rec["is_leaf"],
                }

                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 5. CSV stats.
    stats_path = out_dir / "cwe_sequence_stats.csv"
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cwe_id",
            "name",
            "abstraction",
            "status",
            "is_leaf",
            "num_parents",
            "num_children",
            "num_siblings",
            "num_sequences",
            "min_depth",
            "max_depth",
            "main_sequence",
            "main_abstractions",
        ])

        for cwe_id, rec in sorted(node_records.items()):
            main_seq = rec["main_sequence"]
            main_abs = [
                nodes.get(x, {}).get("abstraction", "")
                for x in main_seq
            ]

            writer.writerow([
                cwe_id,
                rec["name"],
                rec["abstraction"],
                rec["status"],
                rec["is_leaf"],
                len(rec["parents"]),
                len(rec["children"]),
                len(rec["siblings"]),
                rec["num_sequences"],
                rec["min_depth"],
                rec["max_depth"],
                " -> ".join(main_seq),
                " -> ".join(main_abs),
            ])


def print_debug(nodes, parent_map, children_map, node_records):
    print("=" * 80)
    print("[SUMMARY]")
    print(f"nodes in XML: {len(nodes)}")
    print(f"nodes with sequences: {len(node_records)}")

    num_edges = sum(len(v) for v in children_map.values())
    print(f"valid ChildOf edges: {num_edges}")

    roots = [
        cwe_id for cwe_id, rec in node_records.items()
        if len(rec["parents"]) == 0
    ]
    leaves = [
        cwe_id for cwe_id, rec in node_records.items()
        if rec["is_leaf"]
    ]

    print(f"meaningful roots: {len(roots)}")
    print(f"leaf nodes: {len(leaves)}")

    max_depth = max((rec["max_depth"] for rec in node_records.values()), default=0)
    print(f"max depth: {max_depth}")

    print("=" * 80)
    print("[CHECK COMMON CWES]")

    for cwe in ["CWE-89", "CWE-79", "CWE-119", "CWE-125", "CWE-787", "CWE-20", "CWE-416"]:
        rec = node_records.get(cwe)
        print("-" * 80)
        print(cwe)

        if not rec:
            print("not found in View-1000 sequence records")
            continue

        print("name:", rec["name"])
        print("abstraction:", rec["abstraction"])
        print("parents:", rec["parents"])
        print("children:", rec["children"][:10])
        print("siblings:", rec["siblings"][:10])
        print("num_sequences:", rec["num_sequences"])
        print("main_sequence:", " -> ".join(rec["main_sequence"]))
        print(
            "main_abstractions:",
            " -> ".join(nodes.get(x, {}).get("abstraction", "") for x in rec["main_sequence"])
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=str,
        default=str(PROJECT_ROOT / "raw" / "1000.xml"),
        help="Path to downloaded CWE View-1000 XML file."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "processed"),
        help="Output directory."
    )
    args = parser.parse_args()

    xml_path = Path(args.xml)
    out_dir = Path(args.out)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    print(f"[INFO] Parsing XML: {xml_path}")

    nodes, raw_edges = parse_cwe_xml(xml_path)
    parent_map, children_map = build_graph(nodes, raw_edges)
    roots = find_roots(nodes, parent_map, children_map)

    node_to_paths = dfs_paths_from_roots(roots, children_map)
    siblings_map = build_siblings(parent_map, children_map)

    node_records = build_node_sequence_records(
        nodes=nodes,
        parent_map=parent_map,
        children_map=children_map,
        node_to_paths=node_to_paths,
        siblings_map=siblings_map,
    )

    # Only write valid edges whose endpoints exist in nodes.
    valid_edges = []
    for e in raw_edges:
        if e["parent"] in nodes and e["child"] in nodes:
            valid_edges.append(e)

    write_outputs(
        out_dir=out_dir,
        nodes=nodes,
        edges=valid_edges,
        node_records=node_records,
    )

    print_debug(nodes, parent_map, children_map, node_records)

    print("=" * 80)
    print("[OK] Files written to:", out_dir)
    print(" - cwe_nodes.json")
    print(" - cwe_edges.json")
    print(" - cwe_node_sequences.json")
    print(" - cwe_sequences.jsonl")
    print(" - cwe_sequence_stats.csv")


if __name__ == "__main__":
    main()
