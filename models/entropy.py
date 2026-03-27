"""
Entropy computation for Drift-AR (Eq. 4-6 of the paper).

Provides:
  - Per-position causal-normalized entropy from attention maps (Eq. 4)
  - Entropy loss that maximizes attention entropy (Eq. 5)
  - Entropy-to-variance mapping for the drifting prior (Eq. 6)
  - Causal mask generation for the penultimate encoder block
"""

import torch
import math


def make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create an additive causal attention mask.

    Returns:
        mask: [1, 1, seq_len, seq_len] with 0 for allowed positions and
              -inf for masked positions.  Compatible with
              ``F.scaled_dot_product_attention`` and manual softmax.
    """
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]


def make_buffer_content_causal_mask(
    buffer_size: int, content_len: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a block causal mask for buffer + content sequences.

    Layout (L = buffer_size + content_len):
      - buffer  x buffer:  bidirectional (0)
      - buffer  x content: masked (-inf)  — class tokens don't see image tokens
      - content x buffer:  fully visible (0)
      - content x content: causal (upper-triangle = -inf)

    Returns:
        mask: [1, 1, L, L] additive attention mask.
    """
    L = buffer_size + content_len
    mask = torch.zeros(L, L, device=device, dtype=dtype)

    # buffer x content: -inf (buffer cannot attend to content)
    mask[:buffer_size, buffer_size:] = float('-inf')

    # content x content: causal (upper-triangle)
    content_causal = torch.full((content_len, content_len), float('-inf'), device=device, dtype=dtype)
    content_causal = torch.triu(content_causal, diagonal=1)
    mask[buffer_size:, buffer_size:] = content_causal

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]


def compute_per_position_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """Compute causal-normalized per-position entropy from an attention map.

    Eq. 4:  E^(r)_entropy = -1/log(r) * sum_{c=1}^{r} Omega_{r,c} * log(Omega_{r,c})

    The attention map must come from a causal-masked self-attention layer
    (Omega_{r,c} = 0 for c > r).  Each row r is normalized by log(r) — its
    maximum possible entropy — so that E in [0, 1].
    Position r=1 is set to 0 (single support, zero entropy by construction).

    Args:
        attn_weights: [B, H, N, N] softmax attention weights from the
                      penultimate encoder block (after causal masking).
                      H = number of heads.

    Returns:
        entropy: [B, N] per-position normalized entropy, averaged over heads.
    """
    # Average over heads: [B, N, N]
    attn = attn_weights.float().mean(dim=1)

    B, N, _ = attn.shape

    # Clamp for numerical stability in log
    attn_clamped = attn.clamp(min=1e-8)

    # Row-wise entropy: -sum_c Omega_{r,c} * log(Omega_{r,c})  -> [B, N]
    raw_entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)

    # Per-row normalization: log(r) for row r (1-indexed).
    # Row r=0 (1-indexed r=1) has a single support -> entropy is 0.
    row_indices = torch.arange(1, N + 1, device=attn.device, dtype=attn.dtype)
    log_r = row_indices.log().clamp(min=1e-8)  # [N], log(1)=0 handled by clamp
    entropy = raw_entropy / log_r.unsqueeze(0)  # [B, N]

    # Position r=1 (index 0): single support, zero entropy by construction
    entropy[:, 0] = 0.0

    return entropy.clamp(min=0.0, max=1.0)


def compute_entropy_loss(entropy_map: torch.Tensor) -> torch.Tensor:
    """Entropy loss that *maximizes* attention entropy (Eq. 5).

    L_entropy = -1/R * sum_{r=2}^{R} E^(r)_entropy

    Minimizing L_entropy maximizes the normalized entropy at every position,
    encouraging the draft/target to produce diverse, non-collapsed attention.

    Args:
        entropy_map: [B, R] per-position normalized entropy from
                     ``compute_per_position_entropy``.

    Returns:
        Scalar loss (mean over batch).
    """
    R = entropy_map.shape[1]
    if R <= 1:
        return entropy_map.new_zeros(())
    # Sum over positions r=2..R, divide by R (not R-1), average over batch
    loss = -entropy_map[:, 1:].sum(dim=1).mean() / R
    return loss


def entropy_to_variance(
    E: torch.Tensor,
    sigma_max: float = 0.5,
    tau_sigma: float = 2.0,
    E_max: float = 1.0,
) -> torch.Tensor:
    """Map normalized entropy to variance scale (Eq. 6).

    sigma(E) = sigma_max * (exp(E / tau_sigma) - 1) / (exp(E_max / tau_sigma) - 1)

    When E is small (confident AR), sigma ~ 0.
    When E -> E_max, sigma -> sigma_max.

    Args:
        E:         [B, R] per-position normalized entropy in [0, 1].
        sigma_max: Maximum variance cap.
        tau_sigma: Temperature controlling curvature.
        E_max:     Maximum entropy value for normalization (typically 1.0).

    Returns:
        sigma: [B, R] per-position standard deviation.
    """
    E_clamped = E.clamp(min=0.0, max=E_max)
    denom = math.exp(E_max / tau_sigma) - 1.0
    if abs(denom) < 1e-8:
        denom = 1e-8
    sigma = sigma_max * (torch.exp(E_clamped / tau_sigma) - 1.0) / denom
    return sigma.clamp(min=0.0, max=sigma_max)
