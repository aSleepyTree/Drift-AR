"""
Penultimate-layer feature encoder for drift loss computation.

Paper Sec 4.2: "we use a copy of the target AR model's penultimate layer
as phi, which is fixed throughout training."

φ is a frozen deep copy of the penultimate AttentionBlock only (no final
LayerNorm).  It operates in the encoder's hidden-state space [B, seq_len, dim].
"""

import copy
import torch
import torch.nn as nn


class PenultimateLayerEncoder(nn.Module):
    """Frozen copy of the target AR encoder's penultimate AttentionBlock.

    Extracts features by running a single AttentionBlock on the encoder's
    hidden states.  All parameters are frozen at construction time.
    """

    def __init__(self, encoder_block: nn.Module):
        """
        Args:
            encoder_block: the penultimate AttentionBlock from the target
                           encoder (will be deep-copied and frozen).
        """
        super().__init__()
        self.block = copy.deepcopy(encoder_block)

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (no gradient).

        Args:
            x: [B, seq_len, dim] hidden states from the encoder
               (output of the block *before* the penultimate one).

        Returns:
            [B, seq_len, dim] features in phi-space.
        """
        return self._extract(x)

    def forward_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features allowing gradients to flow through x.

        Encoder parameters remain frozen, but gradients propagate back
        to x for generator training.
        """
        return self._extract(x)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
