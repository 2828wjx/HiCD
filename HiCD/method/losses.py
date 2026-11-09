"""HiCD Eqs. 10--16 and 23--24 with level-wise positive-node unions.

The compatibility losses in losses.py operate on individual paths. This module
is the objective used by the published training and evaluation entry points.
"""

import torch
from torch.nn import functional as F

from method.losses import MultiLabelHierarchicalCWELoss, _masked_mean


class PaperHierarchicalCWELoss(MultiLabelHierarchicalCWELoss):
    """MML/CE, multi-positive node discrimination, ranking, and one-way CRD."""

    @staticmethod
    def _positive_sets(level_ids, valid, num_nodes):
        # [B,P,L] -> [B,L,N]. Repeated nodes from distinct paths count once.
        ids = level_ids.transpose(1, 2)
        mask = valid.transpose(1, 2)
        if ((ids[mask] < 0) | (ids[mask] >= num_nodes)).any():
            raise ValueError("Hierarchy supervision contains an invalid node id")
        counts = torch.zeros((*ids.shape[:2], num_nodes), device=ids.device)
        counts.scatter_add_(2, ids.clamp(0, num_nodes - 1), mask.float())
        return counts > 0

    @staticmethod
    def _position_weights(valid, weights):
        mask = valid.to(weights.dtype)
        return (mask * weights.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1)

    @staticmethod
    def _reduce_positions(values, valid, weights):
        """Average valid levels within a sample, then samples (padding excluded)."""
        mask = valid.to(values.dtype)
        per_sample = (values * weights * mask).sum(1) / mask.sum(1).clamp_min(1)
        return _masked_mean(per_sample, valid.any(1))

    def _ranking(self, scores, positives, candidates, valid_positions, weights, anchors):
        zero = scores.sum() * 0.0
        if self.hard_negative_ids is None or self.hard_negative_ids.shape[-1] == 0:
            return zero, zero
        hard_ids = self.hard_negative_ids.to(scores.device)
        hard_safe = hard_ids.clamp_min(0)
        hard_scores = scores[:, :, hard_safe]
        # Restrict to same-level candidates and exclude EVERY sampled positive,
        # including nodes belonging to another valid label of the same sample.
        hard_valid = ((hard_ids >= 0)[None, None]
                      & candidates[:, :, hard_safe]
                      & ~positives[:, :, hard_safe])
        pair_valid = (hard_valid & anchors.unsqueeze(-1)
                      & valid_positions[:, :, None, None])
        ph_values = F.softplus((hard_scores - scores.unsqueeze(-1) + self.pos_hard_margin).clamp(-50, 50))
        counts = pair_valid.sum((-1, -2))
        ph = self._reduce_positions(
            (ph_values * pair_valid).sum((-1, -2)) / counts.clamp_min(1),
            counts > 0, weights,
        ) if self.lambda_pos_hard else zero

        if not self.lambda_hard_medium or self.medium_negative_ids is None or self.medium_negative_ids.shape[-1] == 0:
            return ph, zero
        medium_ids = self.medium_negative_ids.to(scores.device)
        medium_safe = medium_ids.clamp_min(0)
        medium_scores = scores[:, :, medium_safe]
        medium_valid = ((medium_ids >= 0)[None, None]
                        & candidates[:, :, medium_safe]
                        & ~positives[:, :, medium_safe])
        triple_valid = pair_valid.unsqueeze(-1) & medium_valid.unsqueeze(-2)
        hm_values = F.softplus((medium_scores.unsqueeze(-2) - hard_scores.unsqueeze(-1)
                                + self.hard_medium_margin).clamp(-50, 50))
        counts = triple_valid.sum((-1, -2, -3))
        hm = self._reduce_positions(
            (hm_values * triple_valid).sum((-1, -2, -3)) / counts.clamp_min(1),
            counts > 0, weights,
        )
        return ph, hm

    def forward(self, outputs, batch):
        logits = outputs["final_logits"]
        geometry = self._geometry_alignment_loss(outputs, logits)
        if "final_label_mask" in batch:
            final = self._set_classification_loss(logits, batch)
            ids = batch.get("level_ids_multi")
            if ids is not None:
                valid = (batch["real_level_masks"].bool()
                         & batch["hierarchy_loss_masks"].bool().unsqueeze(-1)
                         & batch["path_mask"].bool().unsqueeze(-1) & (ids != -100))
                contrast = (valid & batch["contrast_loss_masks"].bool().unsqueeze(-1)
                            & batch["contrast_level_masks"].bool())
                hierarchy_weights = batch["hierarchy_loss_weights"]
                contrast_weights = batch["contrast_loss_weights"]
        else:
            labels = batch["final_labels"]
            final = _masked_mean(F.cross_entropy(logits, labels, weight=self.final_class_weights,
                                                 ignore_index=-100, reduction="none"),
                                 (labels != -100) & batch["final_classification_mask"].bool())
            ids = batch.get("level_ids")
            if ids is not None:
                ids = ids.unsqueeze(1)
                valid = (batch["real_level_mask"].bool().unsqueeze(1)
                         & batch["hierarchy_loss_mask"].bool()[:, None, None] & (ids != -100))
                contrast = (valid & batch["contrast_loss_mask"].bool()[:, None, None]
                            & batch["contrast_level_mask"].bool().unsqueeze(1))
                hierarchy_weights = batch["hierarchy_loss_weight"].unsqueeze(1)
                contrast_weights = batch["contrast_loss_weight"].unsqueeze(1)
        if not self.auxiliary_losses_enabled():
            return self._final_only_result(logits, final, geometry)
        if ids is None:
            raise ValueError("Hierarchy supervision is required when hierarchy losses are enabled")
        scores = torch.einsum("bld,nd->bln", outputs["level_reprs"], outputs["label_reprs"])
        if scores.shape[1] != ids.shape[-1]:
            raise ValueError("Model and supervision use different hierarchy depths")
        positives = self._positive_sets(ids, valid, scores.shape[-1])
        candidates = self.level_candidate_mask[:scores.shape[1]].to(scores.device)[None] | positives
        weights = self._position_weights(valid, hierarchy_weights)
        zero = scores.sum() * 0.0
        if self.lambda_same_level:
            scaled = scores / self.temperature
            minimum = torch.finfo(scaled.dtype).min / 4
            position_loss = (torch.logsumexp(scaled.masked_fill(~candidates, minimum), -1)
                             - torch.logsumexp(scaled.masked_fill(~positives, minimum), -1))
            same = self._reduce_positions(position_loss, positives.any(-1), weights)
        else:
            same = zero
        ordering_positions = (contrast.any(1) & self._ordering_position_mask(scores.shape[1], scores.device)[None])
        contrast_positives = self._positive_sets(ids, contrast, scores.shape[-1])
        # All positives remain excluded from negatives, including nodes whose
        # individual path has contrast disabled.
        ph, hm = self._ranking(scores, positives, candidates, ordering_positions,
                               self._position_weights(contrast, contrast_weights), contrast_positives)
        total = (self.lambda_final * final + self.auxiliary_scale *
                 (self.lambda_same_level * same + self.lambda_pos_hard * ph + self.lambda_hard_medium * hm)
                 + self._scaled_geometry_alignment(geometry, logits))
        return {"loss": total, "loss_final": final.detach(), "loss_same_level": same.detach(),
                "loss_pos_hard": ph.detach(), "loss_hard_medium": hm.detach(),
                "loss_geometry_alignment": geometry.detach()}
