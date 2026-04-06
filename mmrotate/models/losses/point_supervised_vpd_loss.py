# Copyright (c) OpenMMLab. All rights reserved.
"""Point-Supervised VPD Loss with Point-Conditioned Prior.

All computations are done in **stride-normalized space**:
  - bbox_mu[:, :2] = (Δx/stride, Δy/stride)   posterior mean for center offset
  - bbox_mu[:, 2:] = (log w, log h)             posterior mean for log-size
  - target center:  (gt_center - anchor) / stride
  - d_i_norm = d_i_pixels / stride              kNN density in normalized units

ELBO:
  L = λ_center * L_center  +  λ_kl(t) * KL(q||p)

  L_center = SmoothL1( predicted_delta, gt_delta )   [in normalized space]

  Prior p_ψ(z | p_i, N_i):
    center dims: N(0, σ_c²),  σ_c = σ_c_coeff * d_i_norm
    scale  dims: N(μ_s, σ_s²), μ_s = log(α * d_i_norm), σ_s annealed

  KL = Σ_d KL(N(μ_q,σ_q²) || N(μ_p,σ_p²))
     = Σ_d [ log(σ_p/σ_q) + (σ_q²+(μ_q-μ_p)²)/(2σ_p²) - 0.5 ]

Curriculum:
  Stage A (iter < warmup_iters): λ_kl = λ_kl_warmup, σ_s = σ_s_init
  Stage B: λ_kl and σ_s linearly interpolate to final values over anneal_iters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class PointSupervisedVPDLoss(nn.Module):
    """Point-Supervised VPD Loss in stride-normalized space.

    Key design choices:
    - No separate variance regularization: KL already regularizes σ_q toward σ_p.
    - log_sigma is clamped to [-6, 4] to prevent numerical explosion.
    - KL weight ramps up smoothly (no step jump) from warmup to final value.
    - Scale prior is intentionally broad (σ_s_init=2.0) at start so KL
      does not force scale predictions before the network has seen enough data.

    Args:
        lambda_center (float): Center loss weight. Default: 1.0.
        lambda_kl (float): Final KL weight. Default: 0.05.
        lambda_kl_warmup (float): Initial KL weight. Default: 0.005.
        knn_k (int): kNN neighbors for density estimate. Default: 5.
        sigma_c_coeff (float): Center prior σ_c = coeff * d_i_norm. Default: 1.0.
        scale_alpha_w (float): Scale prior μ_w = log(α_w * d_i_norm). Default: 1.0.
        scale_alpha_h (float): Scale prior μ_h = log(α_h * d_i_norm). Default: 1.0.
        sigma_s_init (float): Initial scale prior σ_s (broad). Default: 2.0.
        sigma_s_final (float): Final scale prior σ_s (tighter). Default: 0.8.
        warmup_iters (int): Iters before KL ramp begins. Default: 1000.
        anneal_iters (int): Iters over which KL ramps to final value. Default: 3000.
        prior_delta_min (float): Min d_i_norm clamp. Default: 0.5.
        prior_delta_max (float): Max d_i_norm clamp. Default: 20.0.
        log_sigma_min (float): Clamp log_sigma from below (prevents σ→0). Default: -6.
        log_sigma_max (float): Clamp log_sigma from above. Default: 4.
    """

    def __init__(self,
                 lambda_center=1.0,
                 lambda_kl=0.05,
                 lambda_kl_warmup=0.005,
                 knn_k=5,
                 sigma_c_coeff=1.0,
                 scale_alpha_w=1.0,
                 scale_alpha_h=1.0,
                 sigma_s_init=2.0,
                 sigma_s_final=0.8,
                 warmup_iters=1000,
                 anneal_iters=3000,
                 prior_delta_min=0.5,
                 prior_delta_max=20.0,
                 log_sigma_min=-6.0,
                 log_sigma_max=4.0,
                 center_beta=1.0):
        super(PointSupervisedVPDLoss, self).__init__()
        self.lambda_center = lambda_center
        self.lambda_kl = lambda_kl
        self.lambda_kl_warmup = lambda_kl_warmup
        self.knn_k = knn_k
        self.sigma_c_coeff = sigma_c_coeff
        self.scale_alpha_w = scale_alpha_w
        self.scale_alpha_h = scale_alpha_h
        self.sigma_s_init = sigma_s_init
        self.sigma_s_final = sigma_s_final
        self.warmup_iters = warmup_iters
        self.anneal_iters = anneal_iters
        self.prior_delta_min = prior_delta_min
        self.prior_delta_max = prior_delta_max
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.center_beta = center_beta

    def _curriculum(self, cur_iter):
        """Return (eff_lambda_kl, sigma_s) with smooth linear ramp."""
        if cur_iter < self.warmup_iters:
            return self.lambda_kl_warmup, self.sigma_s_init
        ratio = min(1.0, (cur_iter - self.warmup_iters) / max(self.anneal_iters, 1))
        eff_lkl = self.lambda_kl_warmup + ratio * (self.lambda_kl - self.lambda_kl_warmup)
        sigma_s = self.sigma_s_init - ratio * (self.sigma_s_init - self.sigma_s_final)
        return eff_lkl, sigma_s

    def forward(self,
                bbox_mu,
                bbox_log_sigma,
                pos_points,
                pos_strides,
                gt_centers,
                gt_centers_list,
                cur_iter=0):
        """Compute loss.

        Args:
            bbox_mu (Tensor): Posterior mean (N, 4).
                [:, :2] = (Δx/stride, Δy/stride)
                [:, 2:] = (log_w, log_h)
            bbox_log_sigma (Tensor): Posterior log-std (N, 4).
            pos_points (Tensor): Anchor points, image coords (N, 2).
            pos_strides (Tensor): Stride per anchor (N,).
            gt_centers (Tensor): Matched GT centers, image coords (N, 2).
            gt_centers_list (list[Tensor]): All GT centers per image (image coords).
            cur_iter (int): Training iteration for curriculum.

        Returns:
            dict[str, Tensor]: loss_center (weighted), loss_kl (weighted).
        """
        N = bbox_mu.shape[0]
        if N == 0:
            zero = bbox_mu.sum() * 0.0
            return dict(loss_center=zero, loss_kl=zero)

        stride_1d = pos_strides.float()          # (N,)
        stride_2d = stride_1d.unsqueeze(1)       # (N, 1)

        # ── Center loss ────────────────────────────────────────────────
        # Both target and prediction are in normalized (stride-divided) space.
        # Use 'sum' reduction divided by N to match detection-style loss scaling,
        # and a small beta to keep more samples in the quadratic regime where
        # gradients are proportional to the error magnitude.
        gt_delta_norm = (gt_centers - pos_points) / stride_2d   # (N, 2)
        pred_delta_norm = bbox_mu[:, :2]                         # (N, 2)
        l_center = F.smooth_l1_loss(pred_delta_norm, gt_delta_norm,
                                    reduction='sum', beta=self.center_beta) / N

        # ── Point-conditioned prior ───────────────────────────────────
        eff_lambda_kl, sigma_s = self._curriculum(cur_iter)

        # kNN distance in pixel space, then normalize by per-sample stride
        all_gt_px = torch.cat(gt_centers_list, dim=0)           # (M, 2)
        M = all_gt_px.shape[0]
        if M > 1:
            dists = torch.cdist(gt_centers, all_gt_px)          # (N, M)
            dists = dists + (dists < 1e-2).float() * 1e8        # mask self
            k = min(self.knn_k, M - 1)
            knn_px, _ = dists.topk(k, dim=1, largest=False)
            d_i_px = knn_px.mean(dim=1)                         # (N,)
        else:
            d_i_px = stride_1d * self.prior_delta_max

        d_i_norm = (d_i_px / stride_1d).clamp(
            min=self.prior_delta_min, max=self.prior_delta_max)  # (N,)

        # Center prior: N(0, σ_c²)
        # Enforce a minimum sigma to prevent KL gradient explosion when
        # the posterior mean is far from the prior mean.
        sigma_c = (self.sigma_c_coeff * d_i_norm).unsqueeze(1).expand(-1, 2)
        sigma_c = sigma_c.clamp(min=1.0)   # at least 1 stride unit
        mu_c = torch.zeros(N, 2, device=bbox_mu.device)

        # Scale prior: N(log(α·d_i_norm), σ_s²)
        # log_w, log_h are log of size in stride-normalized units.
        # Use broad sigma_s to avoid forcing scale before the model converges.
        mu_s = torch.stack([
            torch.log(self.scale_alpha_w * d_i_norm),
            torch.log(self.scale_alpha_h * d_i_norm)], dim=1)   # (N, 2)
        sigma_s_2d = bbox_mu.new_full((N, 2), max(sigma_s, 1.0))

        prior_mu    = torch.cat([mu_c, mu_s], dim=1)            # (N, 4)
        prior_sigma = torch.cat([sigma_c, sigma_s_2d], dim=1)   # (N, 4)

        # ── KL divergence ─────────────────────────────────────────────
        # Clamp log_sigma to prevent σ_q → 0 (collapse) or σ_q → ∞.
        # Using clamp with straight-through: the forward value is clamped,
        # and gradients flow through the clamp boundary.
        log_sigma_q = bbox_log_sigma.clamp(
            min=self.log_sigma_min, max=self.log_sigma_max)
        sigma_q = log_sigma_q.exp()
        sigma_p = prior_sigma.clamp(min=1e-4)

        kl = (torch.log(sigma_p / sigma_q)
              + (sigma_q.pow(2) + (bbox_mu - prior_mu).pow(2))
              / (2.0 * sigma_p.pow(2))
              - 0.5)                                             # (N, 4)

        # Apply KL only to center dims early in training, add scale dims after warmup.
        # This prevents KL explosion from poorly initialized scale predictions.
        center_kl = kl[:, :2].sum(dim=-1).mean()
        scale_kl = kl[:, 2:].sum(dim=-1).mean()

        # Scale KL contribution ramps up more slowly
        scale_kl_ratio = min(1.0, max(0.0,
            (cur_iter - self.warmup_iters) / max(self.anneal_iters, 1)))
        l_kl = center_kl + scale_kl_ratio * scale_kl

        # Return already-weighted losses.  The framework sums all returned
        # loss_* values, so each entry must carry its own coefficient.
        weighted_center = self.lambda_center * l_center
        weighted_kl = eff_lambda_kl * l_kl

        return dict(
            loss_center=weighted_center,
            loss_kl=weighted_kl,
        )
