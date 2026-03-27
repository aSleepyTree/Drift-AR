"""
Drift loss for Drifting Models.
Ported from the official JAX implementation: https://github.com/lambertae/drifting/blob/main/drift_loss.py
"""

import torch
import torch.nn.functional as F


def cdist_l2(x, y, eps=1e-8):
    """Pairwise L2 distance.
    Args:
        x: [B, N, D]
        y: [B, M, D]
    Returns:
        [B, N, M] pairwise distances.
    """
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(sq_dist.clamp(min=eps))


def drift_loss(gen, fixed_pos, fixed_neg=None,
               weight_gen=None, weight_pos=None, weight_neg=None,
               R_list=(0.02, 0.05, 0.2)):
    """Compute the drifting loss.

    All drift-field computation is done under torch.no_grad (stop-gradient).
    Only the final MSE between the (live) generator output and the
    (detached) drifted goal propagates gradients.

    Args:
        gen:       [B, C_g, S]  generated features (C_g samples, S-dim each).
        fixed_pos: [B, C_p, S]  positive (real) features from memory bank.
        fixed_neg: [B, C_n, S]  negative features (optional).
        weight_gen: [B, C_g]    per-sample weights (optional).
        weight_pos: [B, C_p]    per-sample weights (optional).
        weight_neg: [B, C_n]    per-sample weights (optional).
        R_list:    tuple of kernel bandwidth values.

    Returns:
        loss: scalar tensor (mean over batch).
        info: dict with auxiliary metrics.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = fixed_pos.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = fixed_neg.new_ones(B, C_n)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()

    # targets = [old_gen, fixed_neg, fixed_pos]  along dim=1
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    # --- Compute goal under no_grad ---
    with torch.no_grad():
        info = {}

        # Pairwise distances: [B, C_g, C_g+C_n+C_p]
        dist = cdist_l2(old_gen, targets)
        weighted_dist = dist * targets_w[:, None, :]
        scale = weighted_dist.mean() / targets_w.mean().clamp(min=1e-8)
        info["scale"] = scale.item()

        scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs

        # Normalize distance for kernel
        dist_normed = dist / scale.clamp(min=1e-3)

        # Mask self-comparisons (diagonal block for gen vs gen)
        mask_val = 100.0
        diag_mask = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block_mask = F.pad(diag_mask, (0, C_n + C_p))  # [C_g, C_g+C_n+C_p]
        block_mask = block_mask.unsqueeze(0)  # [1, C_g, C_g+C_n+C_p]
        dist_normed = dist_normed + block_mask * mask_val

        # Accumulate forces across R values
        force_across_R = torch.zeros_like(old_gen_scaled)

        for R in R_list:
            logits = -dist_normed / R

            # Symmetric affinity: sqrt(softmax(row) * softmax(col))
            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = torch.sqrt((affinity * aff_transpose).clamp(min=1e-6))

            affinity = affinity * targets_w[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)

            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled

            f_norm_val = (total_force_R ** 2).mean()
            info[f"loss_{R}"] = f_norm_val.item()

            force_scale = f_norm_val.clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R

    # --- Compute loss (gradients flow through gen only) ---
    gen_scaled = gen / scale_inputs.detach()
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))  # [B]

    info_mean = {k: v for k, v in info.items()}
    return loss.mean(), info_mean
