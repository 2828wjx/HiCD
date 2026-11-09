import inspect
import os
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


def one_way_gram_alignment_loss(
    final_classifier_weight: torch.Tensor,
    prototype_weight: torch.Tensor,
    final_to_node_idx: torch.LongTensor,
) -> torch.Tensor:
    """Transfer prototype relations to the flat classifier without an adapter.

    Only mapped final classes participate. Classifier rows are centered across
    classes to remove the common-translation freedom of a softmax classifier.
    Both sides are then row-normalized and their off-diagonal cosine Gram
    matrices are matched. Prototypes are detached deliberately, making the
    transfer one-way from hierarchy geometry to the final classifier.
    """

    if final_classifier_weight.ndim != 2:
        raise ValueError("final_classifier_weight must have shape [C, H]")
    if prototype_weight.ndim != 2:
        raise ValueError("prototype_weight must have shape [N, D]")

    mapping = final_to_node_idx.to(
        device=final_classifier_weight.device,
        dtype=torch.long,
    )
    if mapping.ndim != 1 or mapping.numel() != final_classifier_weight.shape[0]:
        raise ValueError("final_to_node_idx must have shape [C]")
    if torch.any(mapping < -1):
        raise ValueError("final_to_node_idx entries must be -1 or valid node ids")

    valid_class_ids = torch.nonzero(mapping >= 0, as_tuple=False).squeeze(-1)
    if valid_class_ids.numel() > 0:
        valid_node_ids = mapping.index_select(0, valid_class_ids)
        if int(valid_node_ids.max().item()) >= prototype_weight.shape[0]:
            raise ValueError("final_to_node_idx contains an out-of-range node id")
    else:
        valid_node_ids = mapping.new_empty((0,))

    # Preserve a differentiable zero for disabled or degenerate vocabularies.
    if valid_class_ids.numel() < 2:
        return final_classifier_weight.sum() * 0.0

    work_dtype = (
        torch.float32
        if final_classifier_weight.dtype in (torch.float16, torch.bfloat16)
        else final_classifier_weight.dtype
    )
    classifier_rows = final_classifier_weight.index_select(
        0,
        valid_class_ids,
    ).to(dtype=work_dtype)
    classifier_rows = classifier_rows - classifier_rows.mean(
        dim=0,
        keepdim=True,
    )
    classifier_rows = F.normalize(classifier_rows, p=2, dim=-1)

    prototype_rows = prototype_weight.index_select(
        0,
        valid_node_ids.to(prototype_weight.device),
    ).detach().to(device=classifier_rows.device, dtype=work_dtype)
    prototype_rows = F.normalize(prototype_rows, p=2, dim=-1)

    classifier_gram = classifier_rows @ classifier_rows.transpose(0, 1)
    prototype_gram = prototype_rows @ prototype_rows.transpose(0, 1)
    off_diagonal = ~torch.eye(
        valid_class_ids.numel(),
        dtype=torch.bool,
        device=classifier_gram.device,
    )
    return F.smooth_l1_loss(
        classifier_gram[off_diagonal],
        prototype_gram[off_diagonal],
        reduction="mean",
    )


class HierarchicalCWEClassifier(nn.Module):

    def __init__(
        self,
        encoder_name_or_path: str,
        *,
        num_final_labels: int,
        num_level_nodes: int,
        num_levels: int = 4,
        contrast_dim: Optional[int] = None,
        dropout: float = 0.1,
        pooling_mode: str = "auto",
        initializer_std: Optional[float] = None,
        freeze_encoder: bool = False,
        local_files_only: bool = False,
        use_level_embeddings: bool = True,
        final_to_node_idx: Optional[torch.LongTensor] = None,
    ) -> None:
        super().__init__()

        if local_files_only:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        base_model = AutoModel.from_pretrained(
            encoder_name_or_path,
            local_files_only=local_files_only,
        )
        base_config = base_model.config

        # CodeT5 这类模型是 encoder-decoder。做分类时我们只需要 encoder，
        # 否则直接调用完整 T5Model 会要求 decoder_input_ids。
        if getattr(base_config, "is_encoder_decoder", False) and hasattr(base_model, "get_encoder"):
            self.encoder = base_model.get_encoder()
        else:
            self.encoder = base_model

        hidden_size = getattr(base_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(base_config, "d_model", None)
        if hidden_size is None:
            raise ValueError(f"Cannot infer hidden size from encoder config: {base_config}")
        hidden_size = int(hidden_size)
        contrast_dim = int(contrast_dim or hidden_size)
        if pooling_mode not in {"auto", "cls", "mean"}:
            raise ValueError("pooling_mode must be one of: auto, cls, mean")
        if pooling_mode == "auto":
            pooling_mode = (
                "mean" if getattr(base_config, "is_encoder_decoder", False) else "cls"
            )
        if pooling_mode == "cls" and hasattr(self.encoder, "pooler"):
            self.encoder.pooler = None
        if initializer_std is None:
            initializer_std = getattr(base_config, "initializer_range", None)
        if initializer_std is None:
            initializer_std = hidden_size ** -0.5
        initializer_std = float(initializer_std)
        if initializer_std <= 0:
            raise ValueError("initializer_std must be positive")
        self.encoder_accepts_token_type_ids = "token_type_ids" in inspect.signature(self.encoder.forward).parameters

        self.num_final_labels = num_final_labels
        self.num_level_nodes = num_level_nodes
        self.num_levels = num_levels
        self.hidden_size = hidden_size
        self.contrast_dim = contrast_dim
        self.pooling_mode = pooling_mode
        self.initializer_std = initializer_std
        self.use_level_embeddings = bool(use_level_embeddings)

        self.dropout = nn.Dropout(dropout)

        # 最终分类头：所有样本都会训练它，包括 Category 样本。
        self.final_classifier = nn.Linear(hidden_size, num_final_labels)

        # 第 l 层的可训练查询向量。它不代表某个 CWE，只代表“当前要看第几层”。
        self.level_embeddings = nn.Embedding(num_levels, hidden_size) #（4，512?）

        # 把 h_cls + q_l 投影到对比空间，得到 z_l。
        self.level_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, contrast_dim),
        )

        # 每个层级 CWE 节点一个可训练 label embedding。
        # CWE-ID / CWE name 只用于建映射；这里真正参与训练的是向量。
        self.label_embeddings = nn.Embedding(num_level_nodes, contrast_dim)

        if final_to_node_idx is None:
            final_to_node_idx = torch.full(
                (num_final_labels,),
                -1,
                dtype=torch.long,
            )
        else:
            final_to_node_idx = torch.as_tensor(
                final_to_node_idx,
                dtype=torch.long,
            ).detach().clone()
        if final_to_node_idx.ndim != 1 or final_to_node_idx.numel() != num_final_labels:
            raise ValueError("final_to_node_idx must have shape [num_final_labels]")
        if torch.any(final_to_node_idx < -1):
            raise ValueError("final_to_node_idx entries must be -1 or valid node ids")
        valid_node_ids = final_to_node_idx[final_to_node_idx >= 0]
        if valid_node_ids.numel() and int(valid_node_ids.max().item()) >= num_level_nodes:
            raise ValueError("final_to_node_idx contains an out-of-range node id")
        if valid_node_ids.unique().numel() != valid_node_ids.numel():
            raise ValueError("final_to_node_idx must not map two classes to one node")
        # This map is derived from cwe_vocab.json. Keeping it out of state_dict
        # preserves strict loading of checkpoints created before alignment.
        self.register_buffer(
            "final_to_node_idx",
            final_to_node_idx,
            persistent=False,
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self._reset_new_parameters()

    def compute_geometry_alignment_loss(self) -> torch.Tensor:
        return one_way_gram_alignment_loss(
            self.final_classifier.weight,
            self.label_embeddings.weight,
            self.final_to_node_idx,
        )

    def _reset_new_parameters(self) -> None:
        # encoder 自己已经从预训练模型加载了；这里只初始化新增模块。
        def init_new_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.initializer_std)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.final_classifier.apply(init_new_module)
        self.level_embeddings.apply(init_new_module)
        self.level_projector.apply(init_new_module)
        self.label_embeddings.apply(init_new_module)

    def _get_sequence_representation(
        self,
        encoder_outputs,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """BERT-like 模型取原始首 token；T5 类模型做 mask mean pooling。"""

        hidden_states = encoder_outputs.last_hidden_state
        if self.pooling_mode == "cls":
            # CodeBERT/RoBERTa 的 pooler 不参与 MLM 预训练，直接使用原始
            # <s>/CLS 表示可以避免把近似随机的 dense+tanh 当作预训练表示。
            return hidden_states[:, 0]

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None and self.encoder_accepts_token_type_ids:
            encoder_kwargs["token_type_ids"] = token_type_ids

        encoder_outputs = self.encoder(**encoder_kwargs)
        # h_cls = [B,H]
        h_cls = self._get_sequence_representation(encoder_outputs, attention_mask)

        # 1) 最终 CWE 分类分支。
        final_logits = self.final_classifier(self.dropout(h_cls))

        # 2) 层级表示分支。
        # level_query: [L, H]，4 个层级提示向量
        level_ids = torch.arange(self.num_levels, device=h_cls.device)
        if self.use_level_embeddings:
            level_query = self.level_embeddings(level_ids)
        else:
            # Keep the full branch/state_dict but remove level-specific q_l.
            # This isolates level conditioning without changing parameter
            # shapes, initialization order, or checkpoint compatibility.
            level_query = h_cls.new_zeros((self.num_levels, self.hidden_size))

        # level_inputs: [B, L, H]，h_cls 在“层级维度”上扩成 4 份，然后每一份分别加上对应层的 level_query 原始向量
        level_inputs = h_cls.unsqueeze(1) + level_query.unsqueeze(0) # 每个样本在每个层级下的输入表示，如第1层: h_cls + q1
        # 每个样本的逐层表示 z1, z2, z3, z4；形状: [B, L, D]，例子: [8, 4, 256]，然后进行一系列变换，监督捕获层级语义
        level_reprs = self.level_projector(level_inputs)

        # 对比学习里一般归一化后用 cosine similarity 更稳。
        # level_reprs 是样本层级表示
        level_reprs = F.normalize(level_reprs, dim=-1)
        # CWE 层级节点 embedding，形状: [N, D]，例子: [106, 256]，目标是做每个 CWE 的语义锚点，后续为为每个样本的每一层语义来学习具体的cwe信息
        label_reprs = F.normalize(self.label_embeddings.weight, dim=-1)
 
        return {
            "loss_geometry_alignment": self.compute_geometry_alignment_loss(),
            "final_logits": final_logits, # 最终 CWE 分类用，形状 [B, 28]；分类头输出的分数
            "h_cls": h_cls, # encoder 的整体样本表示；encoder 输出，最终分类和层级分支共同的源头
            "level_reprs": level_reprs, # z1-z4，逐层样本表示，用于层级对比损失，学到样本在不同 CWE 层级上的位置
            "label_reprs": label_reprs, # CWE 层级节点 embedding，可训练，与样本的逐层表示 level_reprs 做对比，约束 level_reprs 学到 CWE 层级语义。
        }
