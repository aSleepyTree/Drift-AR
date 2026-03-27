"""
Drift-AR: Single-Step Visual Autoregressive Generation via Anti-Symmetric Drifting.

Full paper implementation with:
  - Entropy computation from penultimate encoder block attention maps (Eq. 4)
  - Entropy-parameterized prior (Eq. 7): x0 = z_AR + sigma(E) * eps
  - Entropy-AdaLN injection into the drift decoder
  - Draft AR model for speculative decoding (Eq. 3)
  - Three losses: L_drift (Eq. 8), L_reg (Eq. 3), L_entropy (Eq. 5)
  - Penultimate-layer feature encoder (phi) for drift loss
"""

from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from diffusers.utils.torch_utils import randn_tensor

from models.crossattention import CrossAttentionBlock, AttentionBlock
from models.drift_decoder import DriftDecoder
from models.drift_loss import drift_loss
from models.entropy import (
    compute_per_position_entropy,
    compute_entropy_loss,
    entropy_to_variance,
    make_buffer_content_causal_mask,
)
from models.draft_model import DraftAR, compute_reg_loss


class DriftAR(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=1,
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        vae_embed_dim=16,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.1,
        proj_dropout=0.1,
        buffer_size=64,
        grad_checkpointing=False,
        # Drift-specific params
        R_list=(0.02, 0.05, 0.2),
        # Entropy params (Eq. 6)
        sigma_max=0.5,
        tau_sigma=2.0,
        # Draft model
        draft_depth=6,
    ):
        super().__init__()

        self.vae_embed_dim = vae_embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        # VAE spatial resolution: img_size/16.  After patchify: (img_size/16/patch_size)^2
        self.seq_len = (img_size // 16 // patch_size) ** 2
        self.grad_checkpointing = grad_checkpointing
        self.token_embed_dim = vae_embed_dim * patch_size ** 2

        # Drift hyperparams
        self.R_list = R_list
        self.sigma_max = sigma_max
        self.tau_sigma = tau_sigma

        # ---------- AR Encoder ----------
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim)
        )

        self.encoder_depth = encoder_depth
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_heads = encoder_num_heads
        self.encoder_blocks = nn.ModuleList([
            AttentionBlock(
                encoder_embed_dim, encoder_num_heads, mlp_ratio,
                qkv_bias=True, norm_layer=norm_layer,
                proj_drop=proj_dropout, attn_drop=attn_dropout,
            )
            for _ in range(encoder_depth * 2)
        ])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # ---------- Draft AR Model ----------
        self.draft_model = DraftAR(
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            depth=draft_depth,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            proj_drop=proj_dropout,
            attn_drop=attn_dropout,
        )

        # ---------- Drift Decoder (single-step, with Entropy-AdaLN) ----------
        self.diffusion_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )
        decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=decoder_embed_dim,
                cross_attention_dim=encoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                proj_drop=proj_dropout,
                attn_drop=attn_dropout,
            )
            for _ in range(decoder_depth)
        ])
        self.drift_decoder = DriftDecoder(
            target_channels=self.token_embed_dim,
            z_channels=encoder_embed_dim,
            diff_seqlen=self.seq_len,
            width=decoder_embed_dim,
            blocks=decoder_blocks,
            grad_checkpointing=grad_checkpointing,
        )

        # Projection from token space to encoder space for phi
        self.token_to_enc_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim)
        # Projection from encoder space to token space for the prior mean (Eq. 7)
        self.enc_to_token_proj = nn.Linear(encoder_embed_dim, self.token_embed_dim)

        # feature_encoder (phi) is set externally after construction via
        # init_feature_encoder() because it requires a deep copy of the
        # penultimate encoder block which must happen after weight init.
        self.feature_encoder = None

        self.initialize_weights()
        self.drift_decoder.net.initialize_weights()

    def init_feature_encoder(self):
        """Create the frozen penultimate-layer feature encoder from this
        model's own encoder.  Must be called after weights are loaded.

        Paper Sec 4.2: φ is a copy of the penultimate layer only (no norm).
        """
        from models.feature_encoder import PenultimateLayerEncoder
        penultimate_block = self.encoder_blocks[-2]
        self.feature_encoder = PenultimateLayerEncoder(penultimate_block)

    def init_draft_from_encoder(self):
        """Initialize draft model from the target encoder's first blocks."""
        self.draft_model = DraftAR.from_target_encoder(
            target_encoder_blocks=self.encoder_blocks,
            target_norm=self.encoder_norm,
            embed_dim=self.encoder_embed_dim,
            num_heads=self.encoder_num_heads,
            depth=self.draft_model.depth,
        )

    def initialize_weights(self):
        torch.nn.init.normal_(self.class_emb.weight, std=0.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=0.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x, hw=16, patch_size=None):
        bsz, c, h, w = x.shape
        p = patch_size if patch_size else self.patch_size
        h_ = w_ = hw
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum("nchpwq->nhwcpq", x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x, hw=16, patch_size=None):
        bsz = x.shape[0]
        p = patch_size if patch_size else self.patch_size
        c = self.vae_embed_dim
        h_ = w_ = hw
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum("nhwcpq->nchpwq", x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def forward_encoder(self, x, class_embedding):
        """Run the encoder and extract entropy from the penultimate block.

        Returns:
            z_AR:        [B, seq_len, embed_dim] encoder output (with diffusion pos embed).
            entropy_map: [B, seq_len] per-position normalized entropy.
        """
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape
        x = torch.cat(
            [torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device, dtype=x.dtype), x],
            dim=1,
        )

        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(device=x.device, dtype=x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, : self.buffer_size] = class_embedding.unsqueeze(1)
        x[:, self.buffer_size :] = self.mask_token

        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # Run all blocks except the last two (penultimate + last)
        num_blocks = len(self.encoder_blocks)
        penult_idx = num_blocks - 2

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for i in range(penult_idx):
                x = checkpoint(self.encoder_blocks[i], x)
        else:
            for i in range(penult_idx):
                x = self.encoder_blocks[i](x)

        # Penultimate block with block causal mask for entropy (Eq. 4).
        # Buffer tokens attend bidirectionally to each other but not to content.
        # Content tokens attend to all buffer tokens + causal content positions.
        # This makes log(r) the correct normalization for content row r.
        block_mask = make_buffer_content_causal_mask(
            self.buffer_size, seq_len, device=x.device, dtype=x.dtype
        )
        x, attn_weights = self.encoder_blocks[penult_idx](x, mask=block_mask, return_attn=True)

        # Last block (runs with full bidirectional attention as before)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(self.encoder_blocks[-1], x)
        else:
            x = self.encoder_blocks[-1](x)

        x = self.encoder_norm(x)

        # Strip buffer tokens
        x = x[:, self.buffer_size:]

        # Compute per-position entropy from the causal-masked penultimate attention.
        # Strip buffer rows/cols so entropy is computed over content positions only.
        # Re-normalize rows to sum to 1 (attention mass on buffer tokens is removed).
        attn_no_buffer = attn_weights[:, :, self.buffer_size:, self.buffer_size:]
        row_sums = attn_no_buffer.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_no_buffer = attn_no_buffer / row_sums
        entropy_map = compute_per_position_entropy(attn_no_buffer)

        x = x + self.diffusion_pos_embed_learned
        return x, entropy_map

    def forward_generate(self, z, entropy_map, noise=None):
        """Single-step generation with entropy-parameterized prior (Eq. 7).

        x0 = z_AR_token + sigma(E) * eps
        x_hat = drift_decoder(x0, z_AR, entropy_map)

        where z_AR_token is z projected from encoder space to token space.
        """
        bsz = z.shape[0]
        seq_len = self.seq_len

        # Entropy-parameterized variance (Eq. 6)
        sigma = entropy_to_variance(
            entropy_map, sigma_max=self.sigma_max, tau_sigma=self.tau_sigma
        )  # [B, seq_len]

        if noise is None:
            noise = torch.randn(bsz, seq_len, self.token_embed_dim, device=z.device, dtype=z.dtype)

        # Eq. 7: x0 = z_AR_token + sigma(E) * eps
        # Project z_AR from encoder space to token space for the prior mean
        z_AR_token = self.enc_to_token_proj(z)  # [B, seq_len, token_embed_dim]
        x0 = z_AR_token + sigma.unsqueeze(-1) * noise

        x_hat = self.drift_decoder(x0, z, entropy_map=entropy_map)
        return x_hat

    def forward(self, imgs, labels, pos_features, neg_features, phase2=False):
        """Training forward pass with three losses.

        Args:
            imgs:         [B, C, H, W] VAE latents (ground truth).
            labels:       [B] class labels.
            pos_features: [B, n_pos, D] positive features from memory bank.
            neg_features: [B, n_neg, D] negative features from memory bank.
            phase2:       If True, detach z_AR and entropy_map to prevent
                          gradients from flowing into the frozen encoder.

        Returns:
            losses: dict with 'drift', 'reg', 'entropy' scalar losses.
            info:   dict with auxiliary metrics.
            gen_latents: [B, C, H, W] generated latent images.
            entropy_map: [B, seq_len] per-position entropy.
        """
        class_embedding = self.class_emb(labels)

        hw = self.img_size // 16  # VAE spatial resolution
        x = self.patchify(imgs, hw=hw)
        x_dummy = torch.zeros(
            (x.size(0), self.seq_len, self.token_embed_dim),
            device=x.device, dtype=x.dtype,
        )

        # Encoder: get z_AR and entropy
        z_AR, entropy_map = self.forward_encoder(x_dummy, class_embedding)

        # In Phase II the AR encoder is frozen; detach to avoid building
        # the computation graph through the encoder (saves memory).
        if phase2:
            z_AR = z_AR.detach()
            entropy_map = entropy_map.detach()

        # --- L_entropy (Eq. 5) ---
        loss_entropy = compute_entropy_loss(entropy_map)

        # --- L_reg (Eq. 3) from draft model ---
        draft_pred = self.draft_model(z_AR.detach())
        loss_reg = compute_reg_loss(draft_pred, z_AR.detach())

        # --- Generate via entropy-parameterized prior + drift decoder ---
        gen_tokens = self.forward_generate(z_AR, entropy_map)
        gen_latents = self.unpatchify(gen_tokens, hw=hw)

        # --- L_drift (Eq. 8) via phi features ---
        loss_drift = torch.tensor(0.0, device=imgs.device)
        drift_info = {}

        if self.feature_encoder is not None and pos_features is not None:
            # Project decoder output (token space) to encoder space, then through phi
            gen_enc_space = self.token_to_enc_proj(gen_tokens)  # [B, seq_len, encoder_dim]
            phi_gen = self.feature_encoder.forward_with_grad(gen_enc_space)  # [B, seq_len, D]

            gen_feat = phi_gen  # [B, seq_len, D]

            loss_drift, drift_info = drift_loss(
                gen=gen_feat,
                fixed_pos=pos_features,
                fixed_neg=neg_features,
                R_list=self.R_list,
            )

        losses = {
            'drift': loss_drift,
            'reg': loss_reg,
            'entropy': loss_entropy,
        }

        return losses, drift_info, gen_latents, entropy_map

    @torch.no_grad()
    def samples(self, labels, cfg=1.0):
        """Single-step generation for evaluation with entropy-parameterized prior."""
        bsz = labels.size(0)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=labels.device)
        class_embedding = self.class_emb(labels)

        if cfg != 1.0:
            tokens = torch.cat([tokens, tokens], dim=0)
            class_embedding = torch.cat(
                [class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0
            )

        z_AR, entropy_map = self.forward_encoder(tokens, class_embedding)

        noise = randn_tensor(
            (bsz, self.seq_len, self.token_embed_dim),
            generator=None, device=z_AR.device, dtype=z_AR.dtype,
        )

        if cfg != 1.0:
            noise_double = torch.cat([noise, noise], dim=0)
            gen_tokens = self.forward_generate(z_AR, entropy_map, noise=noise_double)
            gen_cond, gen_uncond = gen_tokens.chunk(2, dim=0)
            gen_tokens = gen_uncond + cfg * (gen_cond - gen_uncond)
        else:
            gen_tokens = self.forward_generate(z_AR, entropy_map, noise=noise)

        hw = self.img_size // 16
        tokens_out = self.unpatchify(gen_tokens, hw=hw)
        return tokens_out


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------

def drift_ar_base(**kwargs):
    model = DriftAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model


def drift_ar_large(**kwargs):
    model = DriftAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model


def drift_ar_huge(**kwargs):
    model = DriftAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model
