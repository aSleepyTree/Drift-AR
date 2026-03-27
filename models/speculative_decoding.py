"""
Speculative decoding for Drift-AR (Sec 4.3).

The draft AR model proposes features autoregressively, and the target
AR model verifies them.  Entropy-based context-aware early stopping
determines when to stop the draft-verify loop.

Inference pipeline:
  1. Target encoder produces z_AR and entropy_map.
  2. Draft model proposes features from z_AR.
  3. Target verifies: accepted positions use draft features directly
     in the entropy-parameterized prior; rejected positions use target features.
  4. Drift decoder runs once on the (partially draft-accelerated) prior.
  5. Early stopping when draft entropy drops below a threshold.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from models.entropy import compute_per_position_entropy, entropy_to_variance


class EntropyTracker:
    """Tracks running statistics of target entropy for adaptive thresholding.

    tau_global = 0.3 * mean(E) - 0.1 * std(E)  (from training stats)
    tau_c = gamma * EMA(E_target)                (context-aware)
    """

    def __init__(self, gamma: float = 0.8, ema_decay: float = 0.9):
        self.gamma = gamma
        self.ema_decay = ema_decay
        self.ema_entropy: Optional[float] = None
        self.running_mean: float = 0.0
        self.running_var: float = 0.0
        self.count: int = 0

    def update(self, entropy_values: torch.Tensor):
        """Update running statistics with new entropy values."""
        vals = entropy_values.detach().float()
        batch_mean = vals.mean().item()
        batch_var = vals.var().item() if vals.numel() > 1 else 0.0

        n = vals.numel()
        if self.count == 0:
            self.running_mean = batch_mean
            self.running_var = batch_var
        else:
            total = self.count + n
            delta = batch_mean - self.running_mean
            self.running_mean = (self.running_mean * self.count + batch_mean * n) / total
            self.running_var = (self.running_var * self.count + batch_var * n) / total + \
                               (delta ** 2) * self.count * n / (total ** 2)
        self.count += n

        # EMA initialization: E_bar_0 = tau_global / gamma (paper Sec 4.3)
        if self.ema_entropy is None:
            self.ema_entropy = self.tau_global / self.gamma if self.gamma > 0 else batch_mean
        else:
            self.ema_entropy = self.ema_decay * self.ema_entropy + (1 - self.ema_decay) * batch_mean

    @property
    def tau_global(self) -> float:
        """Global threshold from training statistics."""
        std = max(self.running_var, 0.0) ** 0.5
        return 0.3 * self.running_mean - 0.1 * std

    @property
    def tau_context(self) -> float:
        """Context-aware threshold from EMA of target entropy."""
        if self.ema_entropy is None:
            return float('inf')
        return self.gamma * self.ema_entropy


@torch.no_grad()
def speculative_decode(
    target_model: nn.Module,
    draft_model: nn.Module,
    labels: torch.Tensor,
    gamma: float = 0.8,
    entropy_tracker: Optional[EntropyTracker] = None,
    cfg: float = 1.0,
    acceptance_threshold: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """Speculative decoding with context-aware early stopping.

    For accepted positions the draft's features are used directly in the
    entropy-parameterized prior, saving the target encoder from needing
    to produce high-quality features at those positions.  For rejected
    positions the target features are used.  The drift decoder then runs
    once on the merged prior to produce the final output.

    Args:
        target_model:        The full Drift-AR model.
        draft_model:         The lightweight DraftAR model.
        labels:              [B] class labels.
        gamma:               Multiplier for context-aware entropy threshold.
        entropy_tracker:     EntropyTracker for adaptive thresholding.
        cfg:                 Classifier-free guidance scale.
        acceptance_threshold: L2 distance threshold for accepting draft tokens.

    Returns:
        generated_images: [B, C, H, W] generated latent images.
        info: dict with statistics.
    """
    if entropy_tracker is None:
        entropy_tracker = EntropyTracker(gamma=gamma)

    bsz = labels.size(0)
    device = labels.device

    # --- Step 1: Run target encoder ---
    tokens_dummy = torch.zeros(
        bsz, target_model.seq_len, target_model.token_embed_dim, device=device
    )
    class_embedding = target_model.class_emb(labels)

    if cfg != 1.0:
        tokens_dummy_cfg = torch.cat([tokens_dummy, tokens_dummy], dim=0)
        class_embedding_cfg = torch.cat(
            [class_embedding, target_model.fake_latent.repeat(bsz, 1)], dim=0
        )
        z_AR, entropy_map = target_model.forward_encoder(tokens_dummy_cfg, class_embedding_cfg)
        z_AR_cond = z_AR[:bsz]
        entropy_cond = entropy_map[:bsz]
    else:
        z_AR, entropy_map = target_model.forward_encoder(tokens_dummy, class_embedding)
        z_AR_cond = z_AR
        entropy_cond = entropy_map

    entropy_tracker.update(entropy_cond.reshape(-1))

    # --- Step 2: Draft model proposes features ---
    draft_features, draft_attn = draft_model(
        z_AR_cond, return_last_attn=True
    )  # [B, seq_len, D], [B, H, N, N]

    # --- Step 3: Verify draft against target ---
    dist = (draft_features - z_AR_cond).norm(dim=-1)  # [B, seq_len]
    accepted_mask = dist < acceptance_threshold  # [B, seq_len]

    # Find longest accepted prefix per sample
    accepted_lengths = torch.zeros(bsz, device=device, dtype=torch.long)
    for b in range(bsz):
        for pos in range(accepted_mask.shape[1]):
            if accepted_mask[b, pos]:
                accepted_lengths[b] = pos + 1
            else:
                break

    total_accepted = accepted_lengths.float().mean().item()

    # --- Step 4: Merge features — use draft where accepted, target otherwise ---
    # This is the key optimization: accepted positions use the cheaper draft
    merged_features = z_AR_cond.clone()
    merged_features[accepted_mask] = draft_features[accepted_mask]

    # --- Step 5: Early stopping check ---
    draft_entropy = compute_per_position_entropy(draft_attn) if draft_attn is not None else None
    mean_draft_entropy = draft_entropy.mean().item() if draft_entropy is not None else float('inf')
    tau_c = entropy_tracker.tau_context
    tau_g = entropy_tracker.tau_global
    early_stopped = mean_draft_entropy < min(tau_c, tau_g)

    # --- Step 6: Generate output using merged features + entropy prior ---
    seq_len = target_model.seq_len
    sigma = entropy_to_variance(
        entropy_cond,
        sigma_max=target_model.sigma_max,
        tau_sigma=target_model.tau_sigma,
    )
    noise = torch.randn(bsz, seq_len, target_model.token_embed_dim, device=device)

    # Eq. 7: x0 = z_AR_token + sigma * eps  (using merged features)
    z_AR_token = target_model.enc_to_token_proj(merged_features)
    x0 = z_AR_token + sigma.unsqueeze(-1) * noise

    if cfg != 1.0:
        # For CFG we need unconditioned features too
        z_AR_uncond = z_AR[bsz:]
        entropy_uncond = entropy_map[bsz:]
        sigma_uncond = entropy_to_variance(
            entropy_uncond,
            sigma_max=target_model.sigma_max,
            tau_sigma=target_model.tau_sigma,
        )
        z_AR_token_uncond = target_model.enc_to_token_proj(z_AR_uncond)
        x0_uncond = z_AR_token_uncond + sigma_uncond.unsqueeze(-1) * noise

        x0_double = torch.cat([x0, x0_uncond], dim=0)
        z_double = z_AR
        entropy_double = entropy_map

        gen_tokens_double = target_model.drift_decoder(x0_double, z_double, entropy_map=entropy_double)
        gen_cond, gen_uncond = gen_tokens_double.chunk(2, dim=0)
        gen_tokens = gen_uncond + cfg * (gen_cond - gen_uncond)
    else:
        gen_tokens = target_model.drift_decoder(x0, merged_features, entropy_map=entropy_cond)

    hw = target_model.img_size // 16
    generated = target_model.unpatchify(gen_tokens, hw=hw)

    info = {
        'accepted_tokens_mean': total_accepted,
        'accepted_lengths': accepted_lengths,
        'draft_entropy_mean': mean_draft_entropy,
        'tau_context': tau_c,
        'tau_global': tau_g,
        'early_stopped': early_stopped,
    }

    return generated, info


@torch.no_grad()
def speculative_decode_iterative(
    target_model: nn.Module,
    draft_model: nn.Module,
    labels: torch.Tensor,
    max_iterations: int = 4,
    gamma: float = 0.8,
    cfg: float = 1.0,
    acceptance_threshold: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """Multi-round speculative decoding with iterative refinement.

    Each round runs draft-verify and generates output.  Early stopping
    terminates the loop when the draft is confident.

    Args:
        target_model:        The full Drift-AR model.
        draft_model:         The lightweight DraftAR model.
        labels:              [B] class labels.
        max_iterations:      Maximum draft-verify rounds.
        gamma:               Entropy threshold multiplier.
        cfg:                 Classifier-free guidance scale.
        acceptance_threshold: L2 threshold for acceptance.

    Returns:
        generated_images: [B, C, H, W] generated latent images.
        info: dict with iteration statistics.
    """
    tracker = EntropyTracker(gamma=gamma)

    total_accepted = 0
    total_positions = 0
    generated = None

    for iteration in range(max_iterations):
        generated, iter_info = speculative_decode(
            target_model=target_model,
            draft_model=draft_model,
            labels=labels,
            gamma=gamma,
            entropy_tracker=tracker,
            cfg=cfg,
            acceptance_threshold=acceptance_threshold,
        )

        total_accepted += iter_info['accepted_tokens_mean']
        total_positions += target_model.seq_len

        if iter_info['early_stopped']:
            break

    info = {
        'total_iterations': iteration + 1,
        'total_accepted': total_accepted,
        'total_positions': total_positions,
        'acceptance_rate': total_accepted / max(total_positions, 1),
        'early_stopped': iter_info.get('early_stopped', False),
    }

    return generated, info
