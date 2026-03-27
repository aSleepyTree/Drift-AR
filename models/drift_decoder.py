"""
Single-step generator decoder for Drift-AR with Entropy-AdaLN.

Removes timestep conditioning since drifting models use a single forward
pass (no diffusion timestep).  Adds entropy-based adaptive layer
normalization (Entropy-AdaLN) that injects per-position prediction entropy
into the decoder blocks.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.crossattention import CrossAttentionBlock


class DriftFinalLayer(nn.Module):
    """Final projection layer without timestep modulation."""

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class EntropyAdaLN(nn.Module):
    """Entropy-based adaptive layer normalization with per-position modulation.

    Two pathways:
      1. Global: mean-pool entropy -> MLP -> [B, 6, dim] added to level_embed
         (drives the CrossAttentionBlock's adaLN scale/shift/gate).
      2. Spatial: per-position entropy -> MLP -> [B, seq_len, dim] scale & shift
         applied element-wise to the decoder hidden states before each block.
    """

    def __init__(self, model_channels: int):
        super().__init__()
        # Global pathway: mean entropy -> adaLN modulation
        self.global_mlp = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, 6 * model_channels),
        )
        # Spatial pathway: per-position entropy -> element-wise scale & shift
        self.spatial_mlp = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.global_mlp[-1].weight)
        nn.init.zeros_(self.global_mlp[-1].bias)
        nn.init.zeros_(self.spatial_mlp[-1].weight)
        nn.init.zeros_(self.spatial_mlp[-1].bias)

    def forward_global(self, entropy_map: torch.Tensor) -> torch.Tensor:
        """Global adaLN modulation from mean-pooled entropy.

        Args:
            entropy_map: [B, seq_len] per-position entropy.
        Returns:
            [B, 6, dim] modulation signal to add to level_embed.
        """
        e_mean = entropy_map.mean(dim=1, keepdim=True)  # [B, 1]
        out = self.global_mlp(e_mean)  # [B, 6*dim]
        return out.reshape(entropy_map.shape[0], 6, -1)

    def forward_spatial(self, entropy_map: torch.Tensor) -> tuple:
        """Per-position scale and shift from spatial entropy.

        Args:
            entropy_map: [B, seq_len] per-position entropy.
        Returns:
            (scale, shift): each [B, seq_len, dim].
        """
        e = entropy_map.unsqueeze(-1)  # [B, seq_len, 1]
        out = self.spatial_mlp(e)  # [B, seq_len, 2*dim]
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift


class DriftDecoderNet(nn.Module):
    """Transformer decoder that maps noise -> latent tokens in a single pass.

    Conditioned on encoder output z via cross-attention.
    Uses a learned "level embedding" plus optional Entropy-AdaLN modulation
    in place of the timestep embedding to drive the adaLN modulation in
    CrossAttentionBlock.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        blocks,
        diff_seqlen,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.grad_checkpointing = grad_checkpointing

        self.input_proj = nn.Linear(in_channels, model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.level_embed = nn.Parameter(torch.zeros(1, 6, model_channels))

        self.entropy_adaln = EntropyAdaLN(model_channels)

        self.res_blocks = blocks
        self.final_layer = DriftFinalLayer(model_channels, out_channels)

        self.pos_embed = nn.Parameter(torch.zeros(1, diff_seqlen, model_channels))

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.level_embed, std=0.02)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        self.entropy_adaln._init_weights()

    def forward(self, x, c, entropy_map=None):
        """
        Args:
            x: [B, seq_len, in_channels] noise input.
            c: [B, seq_len, z_channels] encoder conditioning.
            entropy_map: [B, seq_len] per-position entropy (optional).
        Returns:
            [B, seq_len, out_channels] predicted latent tokens.
        """
        x = self.input_proj(x)
        x = x + self.pos_embed

        c = self.cond_embed(c)

        t = self.level_embed.expand(x.shape[0], -1, -1)

        if entropy_map is not None:
            t = t + self.entropy_adaln.forward_global(entropy_map)
            spatial_scale, spatial_shift = self.entropy_adaln.forward_spatial(entropy_map)
            # Apply spatial modulation once at the input to avoid compounding
            # across blocks (same scale/shift applied N times would over-amplify).
            x = x * (1.0 + spatial_scale) + spatial_shift

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, c, t)
        else:
            for block in self.res_blocks:
                x = block(x, c, t)

        return self.final_layer(x)


class DriftDecoder(nn.Module):
    """Single-step drift decoder wrapper."""

    def __init__(
        self,
        target_channels,
        z_channels,
        diff_seqlen,
        width,
        blocks,
        grad_checkpointing=False,
    ):
        super().__init__()
        self.in_channels = target_channels
        self.net = DriftDecoderNet(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            diff_seqlen=diff_seqlen,
            blocks=blocks,
            grad_checkpointing=grad_checkpointing,
        )

    def forward(self, noise, z, entropy_map=None):
        """Generate latent tokens from noise, conditioned on encoder output z.

        Args:
            noise:       [B, seq_len, target_channels]
            z:           [B, seq_len, z_channels]
            entropy_map: [B, seq_len] per-position entropy (optional).
        Returns:
            [B, seq_len, target_channels] predicted latent tokens.
        """
        return self.net(noise, z, entropy_map=entropy_map)
