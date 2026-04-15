# Copyright (c) OpenMMLab. All rights reserved.
"""Point-Supervised VPD Loss with Point-Conditioned Prior.

All computations are done in **stride-normalized space**:
  - bbox_mu[:, :2] = (Δx/stride, Δy/stride)   posterior mean for center offset
  - target center:  (gt_center - anchor) / stride
  - d_i_norm = d_i_pixels / stride              kNN density in normalized units

ELBO  (center-only, no w/h — point supervision has no box size signal):
  L = λ_reg * L_reg  +  λ_sigma * L_sigma  +  λ_kl(t) * KL(q||p)

  L_reg   = SmoothL1(μ_q, gt_delta)          — drives μ, σ not involved
  L_sigma = log(σ_q) + e²/(2σ_q²)            — drives σ to match |error|,
            coupled NLL allows spatial variation; μ gradient dominated by L_reg

  Prior p_ψ(z | p_i, N_i):
    center dims: N(0, σ_c²),  σ_c = σ_c_coeff * d_i_norm

  KL = Σ_d KL(N(μ_q,σ_q²) || N(μ_p,σ_p²))     (d = dx, dy only)
"""
import torch
import torch.nn as nn

from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class PointSupervisedVPDLoss(nn.Module):
    """Point-Supervised VPD Loss — center only, no w/h.

    Args:
        lambda_reg (float): Center regression weight. Default: 10.0.
        lambda_sigma (float): Uncertainty loss weight. Default: 1.0.
        lambda_kl (float): Final KL weight. Default: 0.05.
        lambda_kl_warmup (float): Initial KL weight. Default: 0.005.
        knn_k (int): kNN neighbors for density estimate. Default: 5.
        sigma_c_coeff (float): Center prior σ_c = coeff * d_i_norm. Default: 1.0.
        warmup_iters (int): Iters before KL ramp begins. Default: 1000.
        anneal_iters (int): Iters over which KL ramps to final value. Default: 3000.
        prior_delta_min (float): Min d_i_norm clamp. Default: 0.5.
        prior_delta_max (float): Max d_i_norm clamp. Default: 20.0.
        log_sigma_min (float): Clamp log_sigma from below. Default: -6.
        log_sigma_max (float): Clamp log_sigma from above. Default: 4.
    """

    def __init__(self,
                 lambda_reg=10.0,
                 lambda_sigma=1.0,
                 lambda_kl=0.05,
                 lambda_kl_warmup=0.005,
                 knn_k=5,
                 sigma_c_coeff=1.0,
                 warmup_iters=1000,
                 anneal_iters=3000,
                 prior_delta_min=0.5,
                 prior_delta_max=20.0,
                 log_sigma_min=-6.0,
                 log_sigma_max=4.0):
        super(PointSupervisedVPDLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.lambda_sigma = lambda_sigma
        self.lambda_kl = lambda_kl
        self.lambda_kl_warmup = lambda_kl_warmup
        self.knn_k = knn_k
        self.sigma_c_coeff = sigma_c_coeff
        self.warmup_iters = warmup_iters
        self.anneal_iters = anneal_iters
        self.prior_delta_min = prior_delta_min
        self.prior_delta_max = prior_delta_max
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def _curriculum(self, cur_iter):
        """Return effective lambda_kl with smooth linear ramp."""
        if cur_iter < self.warmup_iters:
            return self.lambda_kl_warmup
        ratio = min(1.0, (cur_iter - self.warmup_iters) / max(self.anneal_iters, 1))
        return self.lambda_kl_warmup + ratio * (self.lambda_kl - self.lambda_kl_warmup)

    def forward(self,
                pred_delta,
                pred_log_sigma,
                pos_points,
                pos_strides,
                gt_centers,
                gt_centers_list,
                cur_iter=0):
        """Compute loss.

        Args:
            pred_delta (Tensor): Posterior mean (N, 2) — (Δx/s, Δy/s).
            pred_log_sigma (Tensor): Posterior log-std (N, 2).
            pos_points (Tensor): Anchor points, image coords (N, 2).
            pos_strides (Tensor): Stride per anchor (N,).
            gt_centers (Tensor): Matched GT centers, image coords (N, 2).
            gt_centers_list (list[Tensor]): All GT centers per image.
            cur_iter (int): Training iteration for curriculum.

        Returns:
            dict[str, Tensor]: loss_reg, loss_sigma, loss_kl (all weighted).
        """
        N = pred_delta.shape[0]
        if N == 0:
            zero = pred_delta.sum() * 0.0
            return dict(loss_reg=zero, loss_sigma=zero, loss_kl=zero)

        stride_2d = pos_strides.float().unsqueeze(1)  # (N, 1)

        # ── Clamp log_sigma ───────────────────────────────────────────
        log_sigma_q = pred_log_sigma.clamp(
            min=self.log_sigma_min, max=self.log_sigma_max)  # (N, 2)
        sigma_q = log_sigma_q.exp()                          # (N, 2)

        # ── Center loss: decoupled regression + uncertainty ───────────
        # Two terms with decoupled purposes:
        #   l_reg:   SmoothL1(mu, target) — drives mu, sigma not involved
        #   l_sigma: log(σ) + error²/(2σ²) — coupled NLL drives sigma to
        #            match |error|; mu gradient is safe because l_reg (λ=10)
        #            dominates the NLL mu gradient (λ=1, ~e/σ²) by ~5×
        import torch.nn.functional as F
        gt_delta = (gt_centers - pos_points) / stride_2d     # (N, 2)

        l_reg = F.smooth_l1_loss(pred_delta, gt_delta,
                                 reduction='sum', beta=0.5) / N

        sq_error = (pred_delta - gt_delta).pow(2)               # (N, 2)
        l_sigma = (log_sigma_q + sq_error / (2.0 * sigma_q.pow(2))
                   ).sum(dim=-1).mean()

        # ── Point-conditioned prior (center only) ─────────────────────
        eff_lambda_kl = self._curriculum(cur_iter)

        all_gt_px = torch.cat(gt_centers_list, dim=0)        # (M, 2)
        M = all_gt_px.shape[0]
        if M > 1:
            dists = torch.cdist(gt_centers, all_gt_px)       # (N, M)
            dists = dists + (dists < 1e-2).float() * 1e8     # mask self
            k = min(self.knn_k, M - 1)
            knn_px, _ = dists.topk(k, dim=1, largest=False)
            d_i_px = knn_px.mean(dim=1)                      # (N,)
        else:
            d_i_px = pos_strides.float() * self.prior_delta_max

        d_i_norm = (d_i_px / pos_strides.float()).clamp(
            min=self.prior_delta_min, max=self.prior_delta_max)  # (N,)

        # Center prior: N(0, σ_c²)
        sigma_c = (self.sigma_c_coeff * d_i_norm).unsqueeze(1).expand(-1, 2)
        sigma_c = sigma_c.clamp(min=0.5)
        mu_c = torch.zeros(N, 2, device=pred_delta.device)
        sigma_p = sigma_c.clamp(min=1e-4)

        # ── KL divergence (2 dims only) ──────────────────────────────
        kl = (torch.log(sigma_p / sigma_q)
              + (sigma_q.pow(2) + (pred_delta - mu_c).pow(2))
              / (2.0 * sigma_p.pow(2))
              - 0.5)                                          # (N, 2)
        l_kl = kl.sum(dim=-1).mean()

        return dict(
            loss_reg=self.lambda_reg * l_reg,
            loss_sigma=self.lambda_sigma * l_sigma,
            loss_kl=eff_lambda_kl * l_kl,
        )
