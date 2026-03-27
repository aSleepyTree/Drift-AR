# Drift-AR: Single-Step Visual Autoregressive Generation via Anti-Symmetric Drifting

This is a PyTorch implementation of the paper *Drift-AR: Single-Step Visual Autoregressive Generation via Anti-Symmetric Drifting*, built upon the [TransDiff](https://github.com/TransDiff/TransDiff) codebase.

> **Note:** This repository contains core model components and training infrastructure. Pretrained checkpoints, evaluation scripts, and additional features will be released in a future update.

Drift-AR unifies autoregressive (AR) visual generation and visual decoding acceleration through **per-position prediction entropy**. It uses a **single-step drifting decoder**, achieving one-step (1-NFE) generation while retaining the AR encoder's compositional reasoning.

## Key Components

| Component | Description | Reference |
|-----------|-------------|-----------|
| Entropy from attention maps | Per-position normalized entropy from encoder self-attention | Eq. 4 |
| Entropy loss (L_entropy) | Maximizes attention entropy for uniform information spread | Eq. 5 |
| Entropy-to-variance mapping | Maps entropy to prior variance σ(E) | Eq. 6 |
| Entropy-parameterized prior | x₀ = z_AR + σ(E) · ε | Eq. 7 |
| Anti-symmetric drift loss (L_drift) | Feature-space drifting with Laplace kernels | Eq. 8 |
| Two-phase annealed training | α(t) decays linearly; Phase II freezes encoder | Eq. 9-10 |
| Draft AR model (L_reg) | Lightweight 6-block Transformer for speculative decoding | Eq. 3 |
| Entropy-AdaLN | Global + spatial entropy injection into the drift decoder | Sec. 4.2 |
| Speculative decoding | Draft-verify loop with entropy-based early stopping | Sec. 4.3 |

## Project Structure

```
models/
├── drift_ar.py              # Main Drift-AR model (encoder + drift decoder + draft model)
├── drift_decoder.py         # Single-step drift decoder with Entropy-AdaLN
├── drift_loss.py            # Anti-symmetric drift loss (Laplace kernel)
├── entropy.py               # Entropy computation, loss, and variance mapping (Eq. 4-6)
├── draft_model.py           # Lightweight draft AR model for speculative decoding
├── feature_encoder.py       # Penultimate-layer feature encoder (φ)
├── speculative_decoding.py  # Speculative decoding with entropy-based early stopping
├── memory_bank.py           # Class-wise ring buffer for drift training
├── crossattention.py        # Attention blocks with optional attention weight export
└── vae.py                   # VAE encoder/decoder

engine_drift.py              # Training engine (two-phase annealing, memory banks)
main_drift.py                # Entry point for Drift-AR training/evaluation
test_drift_ar_integration.py # Integration test suite (17 tests)
```

## Preparation

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### VAE Model
We adopt the VAE model from [MAR](https://github.com/LTH14/mar).

### Installation

A suitable [conda](https://conda.io/) environment can be created and activated with:

```bash
conda env create -f environment.yaml
conda activate drift-ar
```

## Usage

### Training

Train Drift-AR-L (two-phase annealed training, 800 epochs, ImageNet 256x256):

```bash
torchrun --nproc_per_node=8 --nnodes=8 --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_drift.py \
--img_size 256 --vae_path ckpt/vae/kl16.ckpt --vae_embed_dim 16 --patch_size 1 \
--model drift_ar_large \
--epochs 800 --warmup_epochs 100 --blr 1.0e-4 --batch_size 32 \
--sigma_max 0.5 --tau_sigma 2.0 \
--alpha_0 0.95 --T_freeze_frac 0.8 \
--draft_depth 6 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH}
```

**Key hyperparameters (from the paper):**

| Argument | Default | Description |
|----------|---------|-------------|
| `--sigma_max` | 0.5 | Maximum variance for entropy-parameterized prior (Eq. 6) |
| `--tau_sigma` | 2.0 | Temperature for entropy-to-variance mapping (Eq. 6) |
| `--alpha_0` | 0.95 | Initial weight for AR losses in Phase I (Eq. 9) |
| `--T_freeze_frac` | 0.8 | Fraction of total epochs for Phase I (Eq. 10) |
| `--draft_depth` | 6 | Number of Transformer blocks in the draft AR model |

**Notes:**
- The paper evaluates only at **256x256**. Higher resolutions remain future work.
- Add `--online_eval` to evaluate FID during training (every 50 epochs).
- Add `--use_cached --cached_path ${CACHED_PATH}` to train with cached VAE latents.
- Add `--bf16` if NaN loss occurs with mixed precision.
- Use `--gradient_accumulation_steps n` if needed.

### Evaluation

Evaluate with standard single-step generation:

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_drift.py \
--img_size 256 --vae_path ckpt/vae/kl16.ckpt --vae_embed_dim 16 --patch_size 1 \
--model drift_ar_large \
--output_dir ${OUTPUT_DIR} --resume ${CKPT_DIR} \
--evaluate --eval_bsz 256 --num_images 50000 --cfg 1.3
```

Evaluate with speculative decoding (Sec 4.3):

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_drift.py \
--img_size 256 --vae_path ckpt/vae/kl16.ckpt --vae_embed_dim 16 --patch_size 1 \
--model drift_ar_large \
--output_dir ${OUTPUT_DIR} --resume ${CKPT_DIR} \
--evaluate --eval_bsz 256 --num_images 50000 --cfg 1.3 \
--use_speculative_decoding --spec_gamma 0.8
```

### Integration Test

Run the test suite to verify all components:

```bash
python test_drift_ar_integration.py
```

This runs 17 tests covering entropy computation, AdaLN modulation, prior correctness, two-phase training, memory banks, and speculative decoding.

## Acknowledgements

We thank [TransDiff](https://github.com/TransDiff/TransDiff) (*Marrying Autoregressive Transformer and Diffusion with Multi-Reference Autoregression*, Zhen et al., 2025) for providing the AR encoder architecture, VAE integration, and training infrastructure that this work builds upon.

We also thank the following projects:
- [Drifting](https://github.com/lambertae/drifting) — the official JAX implementation of *Generative Modeling via Drifting*, from which the drift loss and memory bank are ported.
- [MAR](https://github.com/LTH14/mar) — for the VAE model.
- [diffusers](https://github.com/huggingface/diffusers) and [timm](https://github.com/huggingface/pytorch-image-models) — for foundational building blocks.

## Contact

If you have any questions, feel free to open an issue.
