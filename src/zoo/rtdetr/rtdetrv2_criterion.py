"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision

import copy

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher, 
        weight_dict, 
        losses, 
        alpha=0.2, 
        gamma=2.0, 
        num_classes=80, 
        boxes_weight_format=None,
        share_matched_indices=False,
        small_object_threshold=0.0,
        small_object_weight=0.0,
        small_object_max_scale=1.0,
        small_nwd_threshold=0.0,
        small_nwd_tau=0.2,
        amc_distance='cosine',
        eps=1e-6):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.small_object_threshold = float(small_object_threshold)
        self.small_object_weight = float(small_object_weight)
        self.small_object_max_scale = float(small_object_max_scale)
        self.small_nwd_threshold = float(small_nwd_threshold)
        self.small_nwd_tau = float(small_nwd_tau)
        self.amc_distance = str(amc_distance)
        self.eps = float(eps)

    def loss_amc(self, outputs, targets, indices, num_boxes, **kwargs):
        # Minimal center-assignment regularization:
        # L_amc = mean_i min_j D(z_i, c_{y_i,j}) over matched positives in ETF classes.
        if 'pred_embeds' not in outputs:
            # For aux/dn branches that don't expose embeddings, return zero.
            ref = outputs.get('pred_boxes', None)
            if ref is None:
                ref = next(iter(outputs.values()))
                return {'loss_amc': ref.sum() * 0.0}
            return {'loss_amc': ref.sum() * 0.0}

        if ('etf_class_index' not in outputs) or ('etf_centers' not in outputs):
            return {'loss_amc': outputs['pred_embeds'].sum() * 0.0}

        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {'loss_amc': outputs['pred_embeds'].sum() * 0.0}

        src_embeds = outputs['pred_embeds'][idx]  # [N, D]
        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # [N]

        etf_class_index = outputs['etf_class_index'].to(device=src_embeds.device, dtype=torch.long)
        etf_centers = outputs['etf_centers'].to(device=src_embeds.device, dtype=src_embeds.dtype)  # [K, M, D]

        num_classes = int(self.num_classes)
        map_table = torch.full((num_classes,), -1, dtype=torch.long, device=src_embeds.device)
        map_table[etf_class_index] = torch.arange(etf_class_index.numel(), device=src_embeds.device)

        local_cls = map_table[target_labels]
        keep = local_cls >= 0
        if not torch.any(keep):
            return {'loss_amc': src_embeds.sum() * 0.0}

        z = src_embeds[keep]
        c = etf_centers[local_cls[keep]]  # [N_keep, M, D]

        if self.amc_distance == 'cosine':
            z = F.normalize(z, dim=-1, eps=self.eps)
            c = F.normalize(c, dim=-1, eps=self.eps)
            dist = 1.0 - torch.einsum('nd,nmd->nm', z, c)
        elif self.amc_distance == 'l2':
            dist = (z.unsqueeze(1) - c).pow(2).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported amc_distance: {self.amc_distance}")

        loss_amc = dist.min(dim=1).values.mean()
        return {'loss_amc': loss_amc}

    def _small_nwd_loss(self, src_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        if self.small_nwd_threshold <= 0 or src_boxes.numel() == 0 or target_boxes.numel() == 0:
            return src_boxes.sum() * 0.0

        area = (target_boxes[:, 2] * target_boxes[:, 3]).clamp(min=self.eps)
        small_mask = area < self.small_nwd_threshold
        if not torch.any(small_mask):
            return src_boxes.sum() * 0.0

        src_small = src_boxes[small_mask]
        tgt_small = target_boxes[small_mask]

        center_delta_sq = (src_small[:, :2] - tgt_small[:, :2]).pow(2).sum(dim=-1)
        wh_delta_sq = (src_small[:, 2:] - tgt_small[:, 2:]).pow(2).sum(dim=-1) * 0.25
        w2_distance = (center_delta_sq + wh_delta_sq).clamp(min=self.eps)

        nwd = torch.exp(-torch.sqrt(w2_distance) / max(self.small_nwd_tau, self.eps))
        return (1.0 - nwd).mean()

    def _get_size_weights(self, target_boxes: torch.Tensor) -> torch.Tensor:
        if self.small_object_threshold <= 0 or self.small_object_weight <= 0 or target_boxes.numel() == 0:
            return torch.ones((target_boxes.shape[0],), dtype=target_boxes.dtype, device=target_boxes.device)

        area = (target_boxes[:, 2] * target_boxes[:, 3]).clamp(min=self.eps)
        ratio = torch.sqrt(self.small_object_threshold / area)
        scale = 1.0 + self.small_object_weight * torch.clamp(ratio - 1.0, min=0.0)
        return scale.clamp(max=max(1.0, self.small_object_max_scale))

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        query_weight = torch.ones(src_logits.shape[:2], dtype=src_logits.dtype, device=src_logits.device)
        query_weight[idx] = self._get_size_weights(target_boxes).to(query_weight.dtype)
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = (loss.mean(-1) * query_weight).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        size_weights = self._get_size_weights(target_boxes)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = (loss_bbox * size_weights.unsqueeze(-1)).sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        loss_giou = loss_giou * size_weights
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['loss_nwd_small'] = self._small_nwd_loss(src_boxes, target_boxes)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'amc': self.loss_amc,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)            
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of encoder auxiliary losses. For rtdetr v2
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, enc_targets)
                indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
