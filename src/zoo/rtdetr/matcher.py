"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from typing import Dict 

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from ...core import register


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(
        self,
        weight_dict,
        use_focal_loss=False,
        alpha=0.25,
        gamma=2.0,
        small_object_threshold=0.0,
        small_object_weight=0.0,
        small_object_max_scale=1.0,
        small_object_cost_bonus=0.0,
        small_object_center_radius=2.0,
        eps=1e-6,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.small_object_threshold = float(small_object_threshold)
        self.small_object_weight = float(small_object_weight)
        self.small_object_max_scale = float(small_object_max_scale)
        self.small_object_cost_bonus = float(small_object_cost_bonus)
        self.small_object_center_radius = float(small_object_center_radius)
        self.eps = float(eps)

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    def _get_target_scale(self, tgt_bbox: torch.Tensor) -> torch.Tensor:
        if self.small_object_threshold <= 0 or self.small_object_weight <= 0:
            return torch.ones((tgt_bbox.shape[0],), dtype=tgt_bbox.dtype, device=tgt_bbox.device)

        area = (tgt_bbox[:, 2] * tgt_bbox[:, 3]).clamp(min=self.eps)
        ratio = torch.sqrt(self.small_object_threshold / area)
        scale = 1.0 + self.small_object_weight * torch.clamp(ratio - 1.0, min=0.0)
        return scale.clamp(max=max(1.0, self.small_object_max_scale))

    def _get_small_object_bonus(self, out_bbox: torch.Tensor, tgt_bbox: torch.Tensor) -> torch.Tensor:
        if self.small_object_threshold <= 0 or self.small_object_cost_bonus <= 0:
            return torch.zeros((out_bbox.shape[0], tgt_bbox.shape[0]), dtype=out_bbox.dtype, device=out_bbox.device)

        area = (tgt_bbox[:, 2] * tgt_bbox[:, 3]).clamp(min=self.eps)
        small_mask = area < self.small_object_threshold
        if not torch.any(small_mask):
            return torch.zeros((out_bbox.shape[0], tgt_bbox.shape[0]), dtype=out_bbox.dtype, device=out_bbox.device)

        center_dist = torch.cdist(out_bbox[:, :2], tgt_bbox[:, :2], p=1)
        target_scale = torch.sqrt(area).unsqueeze(0).clamp(min=self.eps)
        radius = max(self.small_object_center_radius, self.eps)
        normalized_dist = center_dist / (target_scale * radius)
        bonus = self.small_object_cost_bonus * torch.exp(-normalized_dist)
        bonus = bonus * small_mask.to(out_bbox.dtype).unsqueeze(0)
        return bonus

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class        
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        bonus = self._get_small_object_bonus(out_bbox, tgt_bbox)
        if torch.count_nonzero(bonus) > 0:
            C = C - bonus
        elif self.small_object_weight > 0:
            # Backward-compatible fallback for older configs.
            target_scale = self._get_target_scale(tgt_bbox)
            C = C * target_scale.unsqueeze(0)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return {'indices': indices}
        