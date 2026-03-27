"""
Draft AR model for speculative decoding (Sec 4.1 of Drift-AR paper).

A lightweight 6-block Transformer that predicts the next-step feature
in continuous space using causal (autoregressive) attention.
Trained with SmoothL1 regression loss (Eq. 3).
Can be initialized from the target AR encoder's first blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.crossattention import AttentionBlock
from models.entropy import make_causal_mask


class DraftAR(nn.Module):
    """6-block draft AR model for speculative decoding.

    Takes the target encoder's hidden-state sequence and predicts
    next-position features autoregressively in continuous space.
    """

    def __init__(self, embed_dim: int, num_heads: int, depth: int = 6,
                 mlp_ratio: float = 4.0, norm_layer=nn.LayerNorm,
                 proj_drop: float = 0.1, attn_drop: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            AttentionBlock(
                embed_dim, num_heads, mlp_ratio,
                qkv_bias=True, norm_layer=norm_layer,
                proj_drop=proj_drop, attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

    @classmethod
    def from_target_encoder(cls, target_encoder_blocks, target_norm,
                            embed_dim, num_heads, depth=6, **kwargs):
        """Initialize from the first ``depth`` blocks of the target encoder."""
        model = cls(embed_dim=embed_dim, num_heads=num_heads, depth=depth, **kwargs)
        src_blocks = list(target_encoder_blocks)[:depth]
        for dst, src in zip(model.blocks, src_blocks):
            dst.load_state_dict(src.state_dict())
        model.norm.load_state_dict(target_norm.state_dict())
        return model

    def forward(self, z_seq: torch.Tensor, mask: torch.Tensor = None,
                return_last_attn: bool = False):
        """Predict next-position features autoregressively.

        Uses causal attention so that position r can only attend to 1..r,
        matching the paper's description of the draft as an AR model.

        Args:
            z_seq: [B, seq_len, embed_dim] input feature sequence from the
                   target encoder (or class-conditioned embeddings).
            mask:  optional attention mask. If None, a causal mask is created.
            return_last_attn: if True, also return [B, H, N, N] attention
                   weights from the last block (for entropy computation).

        Returns:
            pred: [B, seq_len, embed_dim] predicted features.
            attn_weights (optional): [B, H, N, N] last-block attention weights.
        """
        if mask is None:
            mask = make_causal_mask(z_seq.shape[1], device=z_seq.device, dtype=z_seq.dtype)
        x = z_seq
        attn_weights = None
        for i, block in enumerate(self.blocks):
            if return_last_attn and i == len(self.blocks) - 1:
                x, attn_weights = block(x, mask=mask, return_attn=True)
            else:
                x = block(x, mask=mask)
        x = self.norm(x)
        x = self.out_proj(x)
        if return_last_attn:
            return x, attn_weights
        return x


def compute_reg_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SmoothL1 regression loss for draft model (Eq. 3).

    L_reg = SmoothL1(pred_{1:R-1}, target_{2:R})

    The draft predicts position j+1 given positions 1..j, so we compare
    pred[:, :-1] with target[:, 1:].

    Args:
        pred:   [B, R, D] draft model predictions.
        target: [B, R, D] ground-truth encoder features.

    Returns:
        Scalar loss averaged over batch, positions, and dimensions.
    """
    return F.smooth_l1_loss(pred[:, :-1], target[:, 1:].detach())
