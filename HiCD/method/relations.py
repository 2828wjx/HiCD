import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch


PAD_RELATION_ID = -100


@dataclass
class RelationTensors:

    hard_negative_ids: torch.LongTensor
    medium_negative_ids: torch.LongTensor


def _pad_ids(ids: Sequence[int], max_len: int, pad_value: int = PAD_RELATION_ID) -> List[int]:
    ids = [int(x) for x in ids if x is not None and int(x) >= 0]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([pad_value] * (max_len - len(ids)))
    return ids


def build_relation_tensors(
    relation_index_path: str | Path,
    *,
    num_level_nodes: int,
    max_hard_negatives: int = 5,
    max_medium_negatives: int = 5,
    pad_value: int = PAD_RELATION_ID,
) -> RelationTensors:
    """读取 cwe_relation_index.json，并转成固定形状 tensor。

    这个函数只做格式转换，不重新判断 CWE 结构。
    hard / medium 的定义已经在 Stage4 中完成：

    - hard: 当前节点同父节点的兄弟节点。
    - medium: 当前节点父节点的兄弟分支下、与当前节点同层的子节点。
    """

    relation_index_path = Path(relation_index_path)
    relation_data = json.loads(relation_index_path.read_text(encoding="utf-8"))

    hard = torch.full((num_level_nodes, max_hard_negatives), pad_value, dtype=torch.long)
    medium = torch.full((num_level_nodes, max_medium_negatives), pad_value, dtype=torch.long)

    for _, info in relation_data.items():
        target_level_id = info.get("target_level_id")
        if target_level_id is None:
            continue
        target_level_id = int(target_level_id)
        if target_level_id < 0 or target_level_id >= num_level_nodes:
            continue

        hard[target_level_id] = torch.tensor(
            _pad_ids(info.get("hard_negative_level_ids", []), max_hard_negatives, pad_value),
            dtype=torch.long,
        )
        medium[target_level_id] = torch.tensor(
            _pad_ids(info.get("medium_negative_level_ids", []), max_medium_negatives, pad_value),
            dtype=torch.long,
        )

    return RelationTensors(hard_negative_ids=hard, medium_negative_ids=medium)


def _normalize_paths(jsonl_paths: str | Path | Iterable[str | Path]) -> List[Path]:
    if isinstance(jsonl_paths, (str, Path)):
        return [Path(jsonl_paths)]
    return [Path(x) for x in jsonl_paths]


def build_level_candidate_mask_from_jsonl(
    jsonl_paths: str | Path | Iterable[str | Path],
    *,
    num_level_nodes: int,
    num_levels: int = 4,
) -> torch.BoolTensor:


    mask = torch.zeros((num_levels, num_level_nodes), dtype=torch.bool)

    for path in _normalize_paths(jsonl_paths):
        repository_path = path.parent / "dataset_cwe_node_sequences.json"
        vocab_path = path.parent / "cwe_vocab.json"
        repository = json.loads(repository_path.read_text(encoding="utf-8")) if repository_path.is_file() else {}
        vocab = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path.is_file() else {}
        node_ids = vocab.get("level_node_cwe2id", {})
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                supervisions = record.get("hierarchy_supervisions")
                if not isinstance(supervisions, list) or not supervisions:
                    supervisions = [record]

                for supervision in supervisions:
                    if int(supervision.get("hierarchy_loss_mask", 0)) <= 0:
                        continue

                    entry = repository.get(supervision.get("cwe_id"), {})
                    for sequence in entry.get("all_sequences", []):
                        if len(sequence) > num_levels:
                            raise ValueError("Candidate path exceeds configured hierarchy depth")
                        for level_idx, node in enumerate(sequence):
                            if node in node_ids:
                                mask[level_idx, int(node_ids[node])] = True

                    level_ids = list(
                        supervision.get("level_ids", [])
                    )[:num_levels]
                    real_level_mask = list(
                        supervision.get("real_level_mask", [])
                    )[:num_levels]
                    for level_idx, (node_id, is_real) in enumerate(
                        zip(level_ids, real_level_mask)
                    ):
                        if not is_real or node_id is None:
                            continue
                        node_id = int(node_id)
                        if 0 <= node_id < num_level_nodes:
                            mask[level_idx, node_id] = True

    return mask
