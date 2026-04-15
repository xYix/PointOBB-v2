# Copyright (c) OpenMMLab. All rights reserved.
"""Analytic-sigma Pseudo-Label Head (backup / reference).

Uses analytical sigma computed on-the-fly from GT point geometry instead
of a learned conv_sigma. Resolution-independent but cannot encode
object-size-dependent uncertainty (see design notes in training_logic.md).

    sigma(p) = d1(p)/s * (1 + w_amb * d1(p)/d2(p))

where d1, d2 are distances to the nearest and second-nearest GT centers.

NOTE: This approach has a fundamental limitation — a single sigma_edge_ratio
cannot work for both large objects (ships ~80px) and small objects (vehicles
~15px). Use VPDPseudoLabelHead (learned sigma) for production.
"""

import torch
from mmcv.runner import force_fp32

from ..builder import ROTATED_HEADS
from .pseudo_label_head import PseudoLabelHead

INF = 1e8


@ROTATED_HEADS.register_module()
class AnalyticSigmaPseudoLabelHead(PseudoLabelHead):
    """PseudoLabelHead with analytical sigma for boundary-aware edge detection.

    During the boundary walk, computes an analytical sigma at each step
    from GT point geometry. When sigma exceeds sigma_at_center * ratio,
    the walk stops — producing tighter boxes between nearby objects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_cfg = kwargs.get('train_cfg', {})
        self.sigma_edge_ratio = train_cfg.get('sigma_edge_ratio', 1.5)
        self.ambiguity_weight = train_cfg.get('ambiguity_weight', 2.0)

    def get_edge_boundary_simple(self, gt_labels, eigvec, center_point_gt,
                                 cls_scores_all, thresh3=0.1,
                                 default_max_length=128, mode='near',
                                 is_secondary=False,
                                 is_nearest_same_class=None,
                                 nearest_gt_point=None,
                                 first_axis_range=None,
                                 all_gt_centers_px=None):
        """Override: add analytical sigma stop condition.

        Args:
            all_gt_centers_px (Tensor): (M, 2) all GT centers in pixel coords
                for this image, used to compute analytical sigma on-the-fly.
        """
        num_gts = center_point_gt.shape[0]
        stride = self.strides[0]
        center_point_gt_feat = center_point_gt / stride

        eigvec_norm = eigvec / torch.norm(eigvec, dim=1, keepdim=True)
        top_bottom = torch.zeros(num_gts, 2).to(center_point_gt.device)
        cls_scores_sigmoid = cls_scores_all[0].sigmoid()
        H, W = cls_scores_all[0].shape[1], cls_scores_all[0].shape[2]

        if first_axis_range is not None:
            first_axis_range = first_axis_range / stride

        use_sigma = (all_gt_centers_px is not None
                     and all_gt_centers_px.shape[0] >= 2)

        for i in range(num_gts):
            ctr = center_point_gt_feat[i]
            ctr_px = center_point_gt[i]
            eigvec_i = eigvec_norm[i]
            is_same_class = is_nearest_same_class[i]
            nearest_gt_point_i = nearest_gt_point[i] / stride
            direction = nearest_gt_point_i - ctr
            direction_norm = direction / torch.norm(direction)
            distance = torch.abs((direction * eigvec_i).sum())
            if not is_secondary:
                valid_dup_remove = torch.abs(
                    (direction_norm * eigvec_i).sum()) > 0.866
            else:
                valid_dup_remove = torch.abs(
                    (direction_norm * eigvec_i).sum()) > 0.5

            sigma_center = None
            if use_sigma:
                sigma_center = 0.5
                sigma_thr = sigma_center * self.sigma_edge_ratio

            # Walk forward
            for j in range(default_max_length):
                point = (ctr + j * eigvec_i).round().long()
                if point[0] < 0 or point[0] >= W or point[1] < 0 or point[1] >= H:
                    top_bottom[i, 0] = j
                    break
                if cls_scores_sigmoid[gt_labels[i], point[1], point[0]] < \
                        self.thresh3[gt_labels[i]]:
                    top_bottom[i, 0] = j
                    break
                if use_sigma and j > 1:
                    pt_px = ctr_px + j * eigvec_i * stride
                    d_all = torch.norm(
                        all_gt_centers_px - pt_px.unsqueeze(0), dim=1)
                    d_sorted = d_all.sort()[0]
                    d1 = d_sorted[0].item()
                    d2 = d_sorted[1].item() if d_sorted.shape[0] > 1 else 1e8
                    ambiguity = d1 / max(d2, 1e-6)
                    sigma_pt = max(d1 / stride, 0.5) * (
                        1.0 + self.ambiguity_weight * min(ambiguity, 1.0))
                    if sigma_pt > sigma_thr:
                        top_bottom[i, 0] = j
                        break
                if valid_dup_remove:
                    if is_same_class and j > 0.5 * distance:
                        top_bottom[i, 0] = j
                        break
                if first_axis_range is not None:
                    if j > 0.6 * first_axis_range[i]:
                        top_bottom[i, 0] = j
                        break

            # Walk backward
            for j in range(default_max_length):
                point = (ctr - j * eigvec_i).round().long()
                if point[0] < 0 or point[0] >= W or point[1] < 0 or point[1] >= H:
                    top_bottom[i, 1] = j
                    break
                if cls_scores_sigmoid[gt_labels[i], point[1], point[0]] < \
                        self.thresh3[gt_labels[i]]:
                    top_bottom[i, 1] = j
                    break
                if use_sigma and j > 1:
                    pt_px = ctr_px - j * eigvec_i * stride
                    d_all = torch.norm(
                        all_gt_centers_px - pt_px.unsqueeze(0), dim=1)
                    d_sorted = d_all.sort()[0]
                    d1 = d_sorted[0].item()
                    d2 = d_sorted[1].item() if d_sorted.shape[0] > 1 else 1e8
                    ambiguity = d1 / max(d2, 1e-6)
                    sigma_pt = max(d1 / stride, 0.5) * (
                        1.0 + self.ambiguity_weight * min(ambiguity, 1.0))
                    if sigma_pt > sigma_thr:
                        top_bottom[i, 1] = j
                        break
                if is_secondary and valid_dup_remove:
                    if is_same_class and j > 0.5 * distance:
                        top_bottom[i, 0] = j - 1
                        break
                if first_axis_range is not None:
                    if j > 0.6 * first_axis_range[i]:
                        top_bottom[i, 0] = j
                        break

        return top_bottom[:, 0], top_bottom[:, 1]

    def _get_target_single(self, gt_bboxes, gt_labels, cls_scores_all,
                           img_metas, points, regress_ranges,
                           num_points_per_lvl, alpha=1, thresh1=8,
                           thresh2_bg=4, thresh3=0.1, pca_length=28,
                           default_max_length=128, mode='near'):
        """Override to pass all GT centers to boundary walking."""
        alpha = alpha
        thresh3 = self.thresh3
        pca_length = self.pca_length
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        center_point_gt = gt_bboxes[:, :2]

        if num_gts == 0:
            filename_raw = img_metas['filename'].split('/')[-1].split('.')[0]
            with open(self.store_ann_dir + filename_raw + '.txt', 'w') as w:
                w.write('\n')
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))

        dist_sample_and_gt = torch.cdist(points, center_point_gt)
        dist_gt_and_gt = (torch.cdist(center_point_gt, center_point_gt)
                          + torch.eye(num_gts).to(dist_sample_and_gt.device) * INF)
        dist_min_gt_and_gt, dist_min_gt_and_gt_index = dist_gt_and_gt.min(dim=1)
        labels = -1 * torch.ones(num_points, dtype=gt_labels.dtype,
                                 device=gt_labels.device)

        index_neg = ((alpha * dist_sample_and_gt) > dist_min_gt_and_gt).all(
            dim=1).nonzero().squeeze(-1)
        if len(index_neg) > 0:
            labels[index_neg] = self.num_classes

        thresh1_tensor = thresh1 * torch.ones_like(dist_min_gt_and_gt)
        dist_min_thresh1_gt = torch.min(dist_min_gt_and_gt / 2, thresh1_tensor)
        index_pos = (dist_sample_and_gt < dist_min_thresh1_gt).nonzero()
        if len(index_pos) > 0:
            labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]

        is_nearest_same_class = (gt_labels[dist_min_gt_and_gt_index] == gt_labels)
        valid_middle_point = (
            center_point_gt[is_nearest_same_class]
            + center_point_gt[dist_min_gt_and_gt_index][is_nearest_same_class]) / 2
        dist_sample_and_mid = torch.cdist(points, valid_middle_point)
        index_neg_additional = (dist_sample_and_mid < thresh2_bg).any(
            dim=1).nonzero().squeeze(-1)
        if len(index_neg_additional) > 0:
            labels[index_neg_additional] = self.num_classes

        # PCA (same as parent)
        center_factor = self.get_center_factor(
            center_point_gt, gt_labels, cls_scores_all[0])
        gt_ctr_rect = self.get_rectangle_cls_prob(
            cls_scores_all[0].sigmoid(), self.strides[0], center_factor,
            center_point_gt, pca_length, mode='near')
        gt_ctr_rect_label = gt_ctr_rect[torch.arange(num_gts), gt_labels, :, :]
        gt_rect_ctr2edge = gt_ctr_rect_label.shape[-1] // 2
        points_rect_x = torch.arange(
            -gt_rect_ctr2edge, gt_rect_ctr2edge + 1, 1).to(gt_ctr_rect.device)
        points_rect_y = torch.arange(
            -gt_rect_ctr2edge, gt_rect_ctr2edge + 1, 1).to(gt_ctr_rect.device)
        points_rect_xy = torch.stack(
            torch.meshgrid(points_rect_x, points_rect_y), -1).reshape(-1, 2)
        gt_ctr_rect_label = gt_ctr_rect_label.transpose(1, 2).contiguous().view(
            num_gts, -1)
        points_rect_xy_adapt = (
            points_rect_xy.unsqueeze(0).repeat(num_gts, 1, 1)
            * torch.sqrt(gt_ctr_rect_label).unsqueeze(-1))
        points_cov_matrix = (
            torch.matmul(points_rect_xy_adapt.transpose(1, 2),
                         points_rect_xy_adapt)
            / (gt_ctr_rect_label.shape[-1] ** 2 - 1))
        eigvals, eigvecs = torch.linalg.eigh(points_cov_matrix, UPLO='L')

        larger_eigvals_index = (eigvals[:, 1] > eigvals[:, 0]).int()
        eigvec_first = (
            eigvecs[:, 0, :] * (1 - larger_eigvals_index).unsqueeze(1).repeat(1, 2)
            + eigvecs[:, 1, :] * larger_eigvals_index.unsqueeze(1).repeat(1, 2))
        mask_eigvec = (eigvec_first[:, 1] > 0).int()
        epsilon = mask_eigvec * 1e-6 + (1 - mask_eigvec) * -1e-6

        angle_targets = -torch.atan(
            eigvec_first[:, 0] / (eigvec_first[:, 1] + epsilon)).unsqueeze(-1)
        eigvec_second = torch.stack(
            [-eigvec_first[:, 1], eigvec_first[:, 0]], -1)

        first_axis_range = self.get_closest_gt_first_axis(
            gt_labels, eigvec_first, center_point_gt, angle_threshold=0.866)

        all_gt_centers_px = center_point_gt

        top_simple, bottom_simple = self.get_edge_boundary_simple(
            gt_labels, eigvec_first, center_point_gt, cls_scores_all,
            thresh3, default_max_length, mode='simple',
            is_secondary=False,
            is_nearest_same_class=is_nearest_same_class,
            nearest_gt_point=center_point_gt[dist_min_gt_and_gt_index],
            first_axis_range=first_axis_range,
            all_gt_centers_px=all_gt_centers_px)
        left_simple, right_simple = self.get_edge_boundary_simple(
            gt_labels, eigvec_second, center_point_gt, cls_scores_all,
            thresh3, default_max_length, mode='simple',
            is_secondary=True,
            is_nearest_same_class=is_nearest_same_class,
            nearest_gt_point=center_point_gt[dist_min_gt_and_gt_index],
            all_gt_centers_px=all_gt_centers_px)

        top_simple = top_simple * self.strides[0] + 1
        bottom_simple = bottom_simple * self.strides[0] + 1
        left_simple = left_simple * self.strides[0] + 1
        right_simple = right_simple * self.strides[0] + 1

        pseudo_gt_bboxes = torch.cat([
            center_point_gt,
            (left_simple + right_simple).unsqueeze(-1),
            (top_simple + bottom_simple).unsqueeze(-1),
            angle_targets], -1)
        pseudo_gt_bboxes = pseudo_gt_bboxes.detach()

        self.generate_labels(gt_labels, pseudo_gt_bboxes, angle_targets, img_metas)

        if num_gts == 1:
            index_pos = (dist_sample_and_gt < 8).nonzero().reshape(-1)
            index_neg = (dist_sample_and_gt > 128).nonzero().reshape(-1)
            labels[index_pos] = gt_labels[0]
            labels[index_neg] = self.num_classes
            return labels, gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))

        return labels, None, None
