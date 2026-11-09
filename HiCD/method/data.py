import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from scripts.text_field_cleaning import (
    clean_code_text,
    clean_description_text,
)
from method.path_sampling import sample_supervision


IGNORE_INDEX = -100
UNIXCODER_ENCODER_ONLY_TOKEN = "<encoder-only>"


@dataclass
class CWEVocab:

    final_label_cwe2id: Dict[str, int]
    final_label_id2cwe: Dict[int, str]
    level_node_cwe2id: Dict[str, int]
    level_node_id2cwe: Dict[int, str]
    raw: Dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "CWEVocab":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            final_label_cwe2id={str(k): int(v) for k, v in raw["final_label_cwe2id"].items()},
            final_label_id2cwe={int(k): str(v) for k, v in raw["final_label_id2cwe"].items()},
            level_node_cwe2id={str(k): int(v) for k, v in raw["level_node_cwe2id"].items()},
            level_node_id2cwe={int(k): str(v) for k, v in raw["level_node_id2cwe"].items()},
            raw=raw,
        )

    @property
    def num_final_labels(self) -> int:
        return len(self.final_label_cwe2id)

    @property
    def num_level_nodes(self) -> int:
        return len(self.level_node_cwe2id)

    def build_final_to_level_node_index(self) -> torch.LongTensor:
        """Map hierarchy-eligible final labels to their prototype node ids.

        ``hierarchical_final_labels`` is the eligibility authority. This is
        stricter than a vocabulary intersection: a non-hierarchical Category or
        Remain label stays excluded even if it is accidentally present in the
        node vocabulary. Older vocabularies without that field fall back to the
        intersection after removing ``non_hierarchical_final_labels``.
        """

        final_ids = sorted(self.final_label_cwe2id.values())
        if final_ids != list(range(self.num_final_labels)):
            raise ValueError("final label ids must be contiguous from 0")
        node_ids = sorted(self.level_node_cwe2id.values())
        if node_ids != list(range(self.num_level_nodes)):
            raise ValueError("level node ids must be contiguous from 0")

        raw_excluded = self.raw.get("non_hierarchical_final_labels", [])
        if raw_excluded is None:
            raw_excluded = []
        if isinstance(raw_excluded, (str, bytes)):
            raise ValueError("non_hierarchical_final_labels must be a list")
        excluded = {str(label) for label in raw_excluded}
        if len(excluded) != len(raw_excluded):
            raise ValueError(
                "non_hierarchical_final_labels contains duplicates"
            )
        unknown_excluded = sorted(
            excluded - set(self.final_label_cwe2id)
        )
        if unknown_excluded:
            raise ValueError(
                "non_hierarchical_final_labels contains unknown labels: "
                f"{unknown_excluded}"
            )

        declared = self.raw.get("hierarchical_final_labels")
        if declared is None:
            eligible_labels = sorted(
                (
                    set(self.final_label_cwe2id)
                    & set(self.level_node_cwe2id)
                )
                - excluded
            )
        else:
            if isinstance(declared, (str, bytes)):
                raise ValueError("hierarchical_final_labels must be a list")
            eligible_labels = [str(label) for label in declared]
            if len(set(eligible_labels)) != len(eligible_labels):
                raise ValueError("hierarchical_final_labels contains duplicates")

        overlap = sorted(set(eligible_labels) & excluded)
        if overlap:
            raise ValueError(
                "labels cannot be both hierarchical and non-hierarchical: "
                f"{overlap}"
            )
        if (
            declared is not None
            and "non_hierarchical_final_labels" in self.raw
        ):
            partition = set(eligible_labels) | excluded
            missing_partition = sorted(
                set(self.final_label_cwe2id) - partition
            )
            unexpected_partition = sorted(
                partition - set(self.final_label_cwe2id)
            )
            if missing_partition or unexpected_partition:
                raise ValueError(
                    "hierarchical and non-hierarchical labels must partition "
                    "the final vocabulary; "
                    f"missing={missing_partition}, "
                    f"unexpected={unexpected_partition}"
                )

        missing_final = sorted(
            set(eligible_labels) - set(self.final_label_cwe2id)
        )
        missing_node = sorted(
            set(eligible_labels) - set(self.level_node_cwe2id)
        )
        if missing_final or missing_node:
            raise ValueError(
                "hierarchical_final_labels must exist in both vocabularies; "
                f"missing_final={missing_final}, missing_node={missing_node}"
            )

        expected_count = self.raw.get("num_hierarchical_final_labels")
        if expected_count is not None and int(expected_count) != len(eligible_labels):
            raise ValueError(
                "num_hierarchical_final_labels disagrees with its label list"
            )

        mapping = torch.full(
            (self.num_final_labels,),
            -1,
            dtype=torch.long,
        )
        for label in eligible_labels:
            mapping[self.final_label_cwe2id[label]] = self.level_node_cwe2id[label]
        return mapping


class CWEJsonlDataset(Dataset):
    """读取处理后的 CWE JSONL 样本。

    Dataset 只负责读取、轻量清洗和返回标签字段；
    tokenizer 截断和 padding 放在 collator 里做。
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        num_levels: int = 4,
        code_fields: Sequence[str] = ("code", "func"),
        description_fields: Sequence[str] = ("cve_description", "description", "commit_message"),
        keep_raw_record: bool = False,
        clean_text: bool = True,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.num_levels = num_levels
        self.code_fields = tuple(code_fields)
        self.description_fields = tuple(description_fields)
        self.keep_raw_record = keep_raw_record
        self.clean_text = clean_text
        self.epoch = 0
        self.path_seed = 42
        hierarchy_path = self.jsonl_path.parent / "dataset_cwe_node_sequences.json"
        vocab_path = self.jsonl_path.parent / "cwe_vocab.json"
        self.path_repository = json.loads(hierarchy_path.read_text(encoding="utf-8")) if hierarchy_path.is_file() else {}
        vocab = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path.is_file() else {}
        self.path_node_ids = vocab.get("level_node_cwe2id", {})

        self.records: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int, seed: int = 42) -> None:
        self.epoch = int(epoch)
        self.path_seed = int(seed)

    def _sample_path(self, record, index):
        return sample_supervision(record, self.path_repository, self.path_node_ids,
                                  self.num_levels, seed=self.path_seed, epoch=self.epoch,
                                  sample_id=f"{index}:{self.records[index].get('sample_id', index)}")

    def _first_non_empty(self, record: Dict[str, Any], fields: Iterable[str]) -> str:
        for field in fields:
            value = record.get(field)
            if value is not None and str(value).strip():
                return str(value)
        return ""

    def _slice_or_pad(self, values: Sequence[Any], pad_value: Any) -> List[Any]:
        values = list(values or [])
        values = values[: self.num_levels]
        if len(values) < self.num_levels:
            values.extend([pad_value] * (self.num_levels - len(values)))
        return values

    @staticmethod
    def _safe_label_id(value: Any) -> int:
        if value is None:
            return IGNORE_INDEX
        return int(value)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self._sample_path(self.records[idx], idx)

        raw_code = self._first_non_empty(record, self.code_fields)
        raw_description = self._first_non_empty(record, self.description_fields)
        if self.clean_text:
            code = clean_code_text(raw_code)
            description = clean_description_text(raw_description)
        else:
            code = raw_code
            description = raw_description

        # level_ids 里的 -100 是 ignore_index，表示该位置不参与层级损失。
        level_ids = self._slice_or_pad(record.get("level_ids", []), IGNORE_INDEX)
        real_level_mask = self._slice_or_pad(record.get("real_level_mask", []), 0)
        contrast_level_mask = self._slice_or_pad(record.get("contrast_level_mask", []), 0)

        item = {
            "code": code,
            "description": description,
            "text": f"code: {code} description: {description}",
            "sample_id": record.get("sample_id"),
            "cwe_id": record.get("cwe_id"),
            "final_cwe_id": record.get("final_cwe_id", record.get("cwe_id")),
            "is_remain_label": int(record.get("is_remain_label", 0)),
            "final_label": self._safe_label_id(record.get("label_id", IGNORE_INDEX)),
            "final_classification_mask": int(record.get("final_classification_mask", 1)),
            "level_ids": [int(x) if x is not None else IGNORE_INDEX for x in level_ids],
            "real_level_mask": [int(x) for x in real_level_mask],
            "contrast_level_mask": [int(x) for x in contrast_level_mask],
            "hierarchy_loss_mask": int(record.get("hierarchy_loss_mask", 0)),
            "contrast_loss_mask": int(record.get("contrast_loss_mask", 0)),
            "hierarchy_loss_weight": float(
                record.get(
                    "hierarchy_loss_weight",
                    record.get("hierarchy_loss_mask", 0),
                )
            ),
            "contrast_loss_weight": float(
                record.get(
                    "contrast_loss_weight",
                    record.get("contrast_loss_mask", 0),
                )
            ),
            "hierarchy_eligible": bool(record.get("hierarchy_eligible", False)),
        }

        if self.keep_raw_record:
            item["raw_record"] = record
        return item

# 数据加载
class MegaVulJsonlDataset(CWEJsonlDataset):
    """Read MegaVul set-valued labels and all CWE hierarchy paths.

    MegaVul keeps scalar compatibility fields only for single-label rows. This
    dataset intentionally reads ``label_ids`` and ``hierarchy_supervisions`` so
    native multi-label rows are never converted to ``IGNORE_INDEX``.
    """

    def _supervision_item(self, supervision: Dict[str, Any]) -> Dict[str, Any]:
        level_ids = self._slice_or_pad(
            supervision.get("level_ids", []),
            IGNORE_INDEX,
        )
        real_level_mask = self._slice_or_pad(
            supervision.get("real_level_mask", []),
            0,
        )
        contrast_level_mask = self._slice_or_pad(
            supervision.get("contrast_level_mask", []),
            0,
        )
        return {
            "cwe_id": supervision.get("cwe_id"),
            "level_ids": [
                int(value) if value is not None else IGNORE_INDEX
                for value in level_ids
            ],
            "real_level_mask": [int(value) for value in real_level_mask],
            "contrast_level_mask": [
                int(value) for value in contrast_level_mask
            ],
            "hierarchy_loss_mask": int(
                supervision.get("hierarchy_loss_mask", 0)
            ),
            "contrast_loss_mask": int(
                supervision.get("contrast_loss_mask", 0)
            ),
            "hierarchy_loss_weight": float(
                supervision.get(
                    "hierarchy_loss_weight",
                    supervision.get("hierarchy_loss_mask", 0),
                )
            ),
            "contrast_loss_weight": float(
                supervision.get(
                    "contrast_loss_weight",
                    supervision.get("contrast_loss_mask", 0),
                )
            ),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        raw_code = self._first_non_empty(record, self.code_fields)
        raw_description = self._first_non_empty(
            record,
            self.description_fields,
        )
        if self.clean_text:
            code = clean_code_text(raw_code)
            description = clean_description_text(raw_description)
        else:
            code = raw_code
            description = raw_description

        label_ids = [int(value) for value in record.get("label_ids", [])]
        if not label_ids:
            raise ValueError(
                f"MegaVul sample {record.get('sample_id', idx)!r} has no label_ids"
            )
        if len(label_ids) != len(set(label_ids)):
            raise ValueError(
                f"MegaVul sample {record.get('sample_id', idx)!r} has duplicate label_ids"
            )

        supervisions = record.get("hierarchy_supervisions") or []
        if not supervisions:
            supervisions = [record]
        paths = [self._supervision_item(self._sample_path(value, idx)) for value in supervisions]

        item = {
            "code": code,
            "description": description,
            "text": f"code: {code} description: {description}",
            "sample_id": record.get("sample_id"),
            "cwe_ids": list(record.get("cwe_ids") or []),
            "final_cwe_ids": list(record.get("final_cwe_ids") or []),
            "label_ids": label_ids,
            "is_remain_label": int(record.get("is_remain_label", 0)),
            "final_classification_mask": int(
                record.get("final_classification_mask", 1)
            ),
            "hierarchy_paths": paths,
        }
        if self.keep_raw_record:
            item["raw_record"] = record
        return item


class CWEHierCollator:

    def __init__(
        self,
        tokenizer,
        *,
        input_mode: str = "code_desc",
        encoder_input_format: str = "standard",
        max_length: int = 512,
        description_max_length: int = 64,
        code_max_length: Optional[int] = None,
        remain_hierarchy_loss_weight: Optional[float] = None,
        padding: str | bool = "longest",
        code_prefix: str = "code: ",
        description_prefix: str = "description: ",
    ) -> None:
        if input_mode not in {"code", "code_desc", "desc"}:
            raise ValueError("input_mode must be one of: code, code_desc, desc")
        if encoder_input_format not in {"standard", "unixcoder"}:
            raise ValueError(
                "encoder_input_format must be one of: standard, unixcoder"
            )
        if encoder_input_format == "unixcoder" and int(max_length) < 4:
            raise ValueError("UniXcoder input formatting requires max_length >= 4")
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.encoder_input_format = encoder_input_format
        self.max_length = max_length
        self.description_max_length = description_max_length
        self.code_max_length = code_max_length
        self.remain_hierarchy_loss_weight = remain_hierarchy_loss_weight
        self.padding = padding
        self.code_prefix = code_prefix
        self.description_prefix = description_prefix

        self.unixcoder_mode_token_id: Optional[int] = None
        if self.encoder_input_format == "unixcoder":
            vocabulary = self.tokenizer.get_vocab()
            if UNIXCODER_ENCODER_ONLY_TOKEN not in vocabulary:
                raise ValueError(
                    "The selected tokenizer does not contain UniXcoder's "
                    f"{UNIXCODER_ENCODER_ONLY_TOKEN!r} mode token"
                )
            if self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None:
                raise ValueError(
                    "UniXcoder input formatting requires cls_token_id and sep_token_id"
                )
            self.unixcoder_mode_token_id = int(
                vocabulary[UNIXCODER_ENCODER_ONLY_TOKEN]
            )

    def _pair_special_token_count(self) -> int:
        try:
            return int(self.tokenizer.num_special_tokens_to_add(pair=True))
        except TypeError:
            return int(self.tokenizer.num_special_tokens_to_add(True))

    def _effective_code_max_length(self) -> int:
        if self.code_max_length is not None:
            return int(self.code_max_length)
        if self.encoder_input_format == "unixcoder":
            # <s> <encoder-only> </s> content </s>
            fixed_token_count = 4
        else:
            fixed_token_count = self._pair_special_token_count()
        code_budget = self.max_length - self.description_max_length - fixed_token_count
        if code_budget <= 0:
            raise ValueError(
                f"Invalid token budget: max_length={self.max_length}, "
                f"description_max_length={self.description_max_length}, "
                f"pair_special_tokens={self._pair_special_token_count()}"
            )
        return code_budget

    def _encode_without_special_tokens(self, text: str, max_length: int) -> List[int]:
        if int(max_length) <= 0:
            return []
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        return list(encoded["input_ids"])

    def _encode_code_description_pair(self, code: str, description: str) -> Dict[str, List[int]]:
        code_limit = self._effective_code_max_length()
        desc_limit = int(self.description_max_length)

        code_ids = self._encode_without_special_tokens(self.code_prefix + code, code_limit)
        desc_ids = self._encode_without_special_tokens(self.description_prefix + description, desc_limit)

        input_ids = self.tokenizer.build_inputs_with_special_tokens(code_ids, desc_ids)

        # 正常情况下不会超，因为 code_limit 已经扣除了 pair special tokens。
        # 这里做一个防御式兜底，兼容少数 tokenizer 的特殊行为。
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            code_ids = code_ids[: max(len(code_ids) - overflow, 0)]
            input_ids = self.tokenizer.build_inputs_with_special_tokens(code_ids, desc_ids)

        feature: Dict[str, List[int]] = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

        # BERT 类模型可能需要 token_type_ids；RoBERTa/CodeT5 通常不需要。
        if "token_type_ids" in getattr(self.tokenizer, "model_input_names", []):
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(code_ids, desc_ids)
            if len(token_type_ids) == len(input_ids):
                feature["token_type_ids"] = token_type_ids

        return feature

    def _build_unixcoder_feature(
        self,
        content_ids: Sequence[int],
    ) -> Dict[str, List[int]]:
        if self.unixcoder_mode_token_id is None:
            raise RuntimeError("UniXcoder mode token was not initialized")

        cls_token_id = int(self.tokenizer.cls_token_id)
        sep_token_id = int(self.tokenizer.sep_token_id)
        content_budget = max(int(self.max_length) - 4, 0)
        visible_content = [int(value) for value in content_ids[:content_budget]]
        input_ids = [
            cls_token_id,
            self.unixcoder_mode_token_id,
            sep_token_id,
            *visible_content,
            sep_token_id,
        ]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }

    def _encode_unixcoder_code_description(
        self,
        code: str,
        description: str,
    ) -> Dict[str, List[int]]:
        """Build the official encoder-only prefix and reserve the desc budget."""

        desc_limit = int(self.description_max_length)
        # The leading space preserves a lexical boundary when the two separately
        # tokenized fields are concatenated into UniXcoder's single content span.
        desc_ids = self._encode_without_special_tokens(
            " " + self.description_prefix + description,
            desc_limit,
        )
        maximum_code_budget = max(int(self.max_length) - 4 - len(desc_ids), 0)
        requested_code_budget = self._effective_code_max_length()
        code_limit = min(requested_code_budget, maximum_code_budget)
        code_ids = self._encode_without_special_tokens(
            self.code_prefix + code,
            code_limit,
        )
        return self._build_unixcoder_feature([*code_ids, *desc_ids])

    def _encode_unixcoder_single(
        self,
        text: str,
        *,
        prefix: str = "",
        max_content_length: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        content_budget = max(int(self.max_length) - 4, 0)
        if max_content_length is not None:
            content_budget = min(content_budget, int(max_content_length))
        content_ids = self._encode_without_special_tokens(
            prefix + text,
            content_budget,
        )
        return self._build_unixcoder_feature(content_ids)

    def _encode_single(
        self,
        text: str,
        *,
        prefix: str = "",
        max_content_length: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        try:
            special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        except TypeError:
            special_tokens = int(self.tokenizer.num_special_tokens_to_add(False))
        content_budget = max(self.max_length - special_tokens, 0)
        if max_content_length is not None:
            content_budget = min(content_budget, int(max_content_length))
        token_ids = self._encode_without_special_tokens(
            prefix + text,
            content_budget,
        )
        input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        feature: Dict[str, List[int]] = {
            "input_ids": input_ids[: self.max_length],
            "attention_mask": [1] * min(len(input_ids), self.max_length),
        }
        if "token_type_ids" in getattr(self.tokenizer, "model_input_names", []):
            token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids)
            if len(token_type_ids) == len(input_ids):
                feature["token_type_ids"] = token_type_ids[: self.max_length]
        return feature

    def _tokenize_batch(
        self,
        batch: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.encoder_input_format == "unixcoder" and self.input_mode == "code_desc":
            features = [
                self._encode_unixcoder_code_description(
                    x["code"],
                    x["description"],
                )
                for x in batch
            ]
        elif self.encoder_input_format == "unixcoder" and self.input_mode == "code":
            features = [
                self._encode_unixcoder_single(
                    x["code"],
                    max_content_length=self.code_max_length,
                )
                for x in batch
            ]
        elif self.encoder_input_format == "unixcoder":
            features = [
                self._encode_unixcoder_single(
                    x["description"],
                    prefix=self.description_prefix,
                    max_content_length=self.description_max_length,
                )
                for x in batch
            ]
        elif self.input_mode == "code_desc":
            features = [
                self._encode_code_description_pair(x["code"], x["description"])
                for x in batch
            ]
        elif self.input_mode == "code":
            features = [
                self._encode_single(
                    x["code"],
                    max_content_length=self.code_max_length,
                )
                for x in batch
            ]
        else:
            features = [
                self._encode_single(
                    x["description"],
                    prefix=self.description_prefix,
                    max_content_length=self.description_max_length,
                )
                for x in batch
            ]

        pad_kwargs: Dict[str, Any] = {
            "padding": self.padding,
            "return_tensors": "pt",
        }
        if self.padding == "max_length":
            pad_kwargs["max_length"] = self.max_length

        return dict(self.tokenizer.pad(features, **pad_kwargs))

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        result = self._tokenize_batch(batch)
        result["final_labels"] = torch.tensor([x["final_label"] for x in batch], dtype=torch.long)
        result["final_classification_mask"] = torch.tensor(
            [x["final_classification_mask"] for x in batch],
            dtype=torch.float,
        )
        result["level_ids"] = torch.tensor([x["level_ids"] for x in batch], dtype=torch.long)
        result["real_level_mask"] = torch.tensor([x["real_level_mask"] for x in batch], dtype=torch.float)
        result["contrast_level_mask"] = torch.tensor(
            [x["contrast_level_mask"] for x in batch],
            dtype=torch.float,
        )
        result["hierarchy_loss_mask"] = torch.tensor(
            [x["hierarchy_loss_mask"] for x in batch],
            dtype=torch.float,
        )
        result["contrast_loss_mask"] = torch.tensor(
            [x["contrast_loss_mask"] for x in batch],
            dtype=torch.float,
        )
        hierarchy_weights = []
        contrast_weights = []
        for x in batch:
            hierarchy_weight = float(x["hierarchy_loss_weight"])
            contrast_weight = float(x["contrast_loss_weight"])
            if (
                self.remain_hierarchy_loss_weight is not None
                and int(x.get("is_remain_label", 0)) == 1
                and int(x.get("hierarchy_loss_mask", 0)) == 1
            ):
                hierarchy_weight = float(self.remain_hierarchy_loss_weight)
                contrast_weight = float(self.remain_hierarchy_loss_weight)
            hierarchy_weights.append(hierarchy_weight)
            contrast_weights.append(contrast_weight)

        result["hierarchy_loss_weight"] = torch.tensor(
            hierarchy_weights,
            dtype=torch.float,
        )
        result["contrast_loss_weight"] = torch.tensor(
            contrast_weights,
            dtype=torch.float,
        )

        # 非 tensor 字段主要用于 debug / 打印错误样本。
        result["sample_id"] = [x["sample_id"] for x in batch]
        result["cwe_id"] = [x["cwe_id"] for x in batch]
        result["final_cwe_id"] = [x["final_cwe_id"] for x in batch]
        result["is_remain_label"] = torch.tensor(
            [x["is_remain_label"] for x in batch],
            dtype=torch.long,
        )
        result["hierarchy_eligible"] = [x["hierarchy_eligible"] for x in batch]
        return result


class MegaVulHierCollator(CWEHierCollator):
    """Collate set-valued final labels and variable-count hierarchy paths."""

    def __init__(
        self,
        tokenizer,
        *,
        num_final_labels: int,
        num_levels: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(tokenizer, **kwargs)
        if int(num_final_labels) <= 0:
            raise ValueError("num_final_labels must be positive")
        if int(num_levels) <= 0:
            raise ValueError("num_levels must be positive")
        self.num_final_labels = int(num_final_labels)
        self.num_levels = int(num_levels)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("MegaVul collator received an empty batch")

        result = self._tokenize_batch(batch)
        batch_size = len(batch)
        max_paths = max(
            max(len(item.get("hierarchy_paths") or []), 1)
            for item in batch
        )

        final_label_mask = torch.zeros(
            (batch_size, self.num_final_labels),
            dtype=torch.bool,
        )
        level_ids = torch.full(
            (batch_size, max_paths, self.num_levels),
            IGNORE_INDEX,
            dtype=torch.long,
        )
        real_level_masks = torch.zeros(
            (batch_size, max_paths, self.num_levels),
            dtype=torch.float,
        )
        contrast_level_masks = torch.zeros_like(real_level_masks)
        path_mask = torch.zeros((batch_size, max_paths), dtype=torch.float)
        hierarchy_loss_masks = torch.zeros_like(path_mask)
        contrast_loss_masks = torch.zeros_like(path_mask)
        hierarchy_loss_weights = torch.zeros_like(path_mask)
        contrast_loss_weights = torch.zeros_like(path_mask)

        for batch_index, item in enumerate(batch):
            for label_id in item["label_ids"]:
                label_id = int(label_id)
                if not 0 <= label_id < self.num_final_labels:
                    raise ValueError(
                        f"label id {label_id} is outside [0, {self.num_final_labels})"
                    )
                final_label_mask[batch_index, label_id] = True

            paths = item.get("hierarchy_paths") or []
            for path_index, path in enumerate(paths):
                current_level_ids = list(path["level_ids"])
                current_real_mask = list(path["real_level_mask"])
                current_contrast_mask = list(path["contrast_level_mask"])
                if not (
                    len(current_level_ids)
                    == len(current_real_mask)
                    == len(current_contrast_mask)
                    == self.num_levels
                ):
                    raise ValueError(
                        "MegaVul hierarchy path length differs from num_levels"
                    )

                level_ids[batch_index, path_index] = torch.tensor(
                    current_level_ids,
                    dtype=torch.long,
                )
                real_level_masks[batch_index, path_index] = torch.tensor(
                    current_real_mask,
                    dtype=torch.float,
                )
                contrast_level_masks[batch_index, path_index] = torch.tensor(
                    current_contrast_mask,
                    dtype=torch.float,
                )
                path_mask[batch_index, path_index] = 1.0
                hierarchy_loss_masks[batch_index, path_index] = float(
                    path["hierarchy_loss_mask"]
                )
                contrast_loss_masks[batch_index, path_index] = float(
                    path["contrast_loss_mask"]
                )

                hierarchy_weight = float(path["hierarchy_loss_weight"])
                contrast_weight = float(path["contrast_loss_weight"])
                if (
                    self.remain_hierarchy_loss_weight is not None
                    and int(item.get("is_remain_label", 0)) == 1
                    and int(path["hierarchy_loss_mask"]) == 1
                ):
                    hierarchy_weight = float(
                        self.remain_hierarchy_loss_weight
                    )
                    contrast_weight = float(
                        self.remain_hierarchy_loss_weight
                    )
                hierarchy_loss_weights[batch_index, path_index] = (
                    hierarchy_weight
                )
                contrast_loss_weights[batch_index, path_index] = (
                    contrast_weight
                )

        if not final_label_mask.any(dim=1).all():
            raise ValueError("Every MegaVul sample must contain a final label")

        result["final_label_mask"] = final_label_mask
        result["final_label_counts"] = final_label_mask.sum(dim=1)
        result["final_classification_mask"] = torch.tensor(
            [item["final_classification_mask"] for item in batch],
            dtype=torch.float,
        )
        result["level_ids_multi"] = level_ids
        result["real_level_masks"] = real_level_masks
        result["contrast_level_masks"] = contrast_level_masks
        result["path_mask"] = path_mask
        result["hierarchy_loss_masks"] = hierarchy_loss_masks
        result["contrast_loss_masks"] = contrast_loss_masks
        result["hierarchy_loss_weights"] = hierarchy_loss_weights
        result["contrast_loss_weights"] = contrast_loss_weights

        result["sample_id"] = [item["sample_id"] for item in batch]
        result["cwe_ids"] = [item["cwe_ids"] for item in batch]
        result["final_cwe_ids"] = [
            item["final_cwe_ids"] for item in batch
        ]
        result["label_ids"] = [item["label_ids"] for item in batch]
        result["is_remain_label"] = torch.tensor(
            [item["is_remain_label"] for item in batch],
            dtype=torch.long,
        )
        return result
