"""
Training engine for Drift-AR with two-phase annealed training (Eq. 9-10).

Phase I:  L_total = alpha * (L_reg + L_entropy) + (1 - alpha) * L_drift
          alpha decays linearly from alpha_0 to 0 over T_freeze epochs.
Phase II: alpha = 0, encoder frozen, only L_drift active.
"""

import math
import sys
from typing import Iterable

import torch
import numpy as np
import os
import copy
import time
import random
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution
from models.memory_bank import ArrayMemoryBank
import torch_fidelity
import shutil


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def compute_alpha(epoch, total_epochs, alpha_0=0.95, T_freeze_frac=0.8):
    """Annealing schedule for the two-phase training (Eq. 9-10).

    Returns alpha in [0, alpha_0]. When epoch > T_freeze, alpha = 0
    and the encoder is frozen (Phase II).
    """
    T_freeze = T_freeze_frac * total_epochs
    if epoch <= T_freeze:
        return alpha_0 * (1.0 - epoch / T_freeze)
    return 0.0


def is_phase_two(epoch, total_epochs, T_freeze_frac=0.8):
    """Check if we are in Phase II (encoder frozen)."""
    return epoch > T_freeze_frac * total_epochs


def freeze_encoder(model_without_ddp):
    """Freeze all encoder parameters and enc_to_token_proj for Phase II.

    enc_to_token_proj is also frozen because it maps z_AR to token space
    for the prior mean — updating it would change the source distribution q,
    violating the stationary-source assumption required by the drifting field.
    """
    for name, param in model_without_ddp.named_parameters():
        if 'encoder_blocks' in name or 'encoder_norm' in name or \
           'z_proj' in name or 'z_proj_ln' in name or \
           'class_emb' in name or 'encoder_pos_embed_learned' in name or \
           'fake_latent' in name or 'mask_token' in name or \
           'enc_to_token_proj' in name or \
           'diffusion_pos_embed_learned' in name:
            param.requires_grad = False
    for param in model_without_ddp.draft_model.parameters():
        param.requires_grad = False


def init_feature_banks(feature_dim, args):
    """Initialize a single pair of memory banks for phi-space features.

    The PenultimateLayerEncoder outputs [B, seq_len, dim] features.
    We store flattened [seq_len * dim] vectors per sample.

    Returns:
        bank_pos: ArrayMemoryBank (per-class positive)
        bank_neg: ArrayMemoryBank (global negative, num_classes=1)
        feature_dim: int (the flattened feature dimension)
    """
    bank_pos = ArrayMemoryBank(
        num_classes=args.class_num, max_size=args.pos_bank_size
    )
    bank_neg = ArrayMemoryBank(
        num_classes=1, max_size=args.neg_bank_size
    )
    return bank_pos, bank_neg


def extract_and_push_to_banks(phi_features, labels, bank_pos):
    """Push phi-space features of real data to the positive memory bank only.

    The negative bank should only contain generated (pushforward) samples,
    not real data — mixing them would corrupt the generated distribution q.

    Args:
        phi_features: [B, seq_len, D] features from PenultimateLayerEncoder.
        labels:       [B] class labels.
        bank_pos:     per-class positive bank.
    """
    with torch.no_grad():
        B = phi_features.shape[0]
        feat_flat = phi_features.reshape(B, -1)  # [B, seq_len*D]
        bank_pos.add(feat_flat, labels)


def sample_from_banks(labels, bank_pos, bank_neg, pos_per_sample, neg_per_sample,
                      seq_len, feat_dim, device):
    """Sample features from memory banks and reshape for drift_loss.

    Returns:
        pos_features: [B, pos_per_sample * seq_len, D] or None
        neg_features: [B, neg_per_sample * seq_len, D] or None
    """
    pos_features = None
    neg_features = None

    if bank_pos.total_count() > 0:
        pos_flat = bank_pos.sample(labels, pos_per_sample, device=device)
        # pos_flat: [B, pos_per_sample, seq_len*D] -> [B, pos_per_sample*seq_len, D]
        B = pos_flat.shape[0]
        pos_features = pos_flat.reshape(B, -1, feat_dim)

    if bank_neg.total_count() > 0:
        neg_flat = bank_neg.sample(labels * 0, neg_per_sample, device=device)
        B = neg_flat.shape[0]
        neg_features = neg_flat.reshape(B, -1, feat_dim)

    return pos_features, neg_features


def train_one_epoch(model, vae,
                    bank_pos, bank_neg,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)

    model_without_ddp = model.module if hasattr(model, 'module') else model

    # Two-phase annealing
    alpha = compute_alpha(epoch, args.epochs, args.alpha_0, args.T_freeze_frac)
    phase2 = is_phase_two(epoch, args.epochs, args.T_freeze_frac)

    if phase2:
        freeze_encoder(model_without_ddp)
        print(f"Phase II: encoder frozen, alpha=0, only L_drift active")
    else:
        print(f"Phase I: alpha={alpha:.4f}")

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('drift_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('reg_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('entropy_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('alpha', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    min_bank_samples = max(args.pos_per_sample, 2)
    seq_len = model_without_ddp.seq_len
    feat_dim = model_without_ddp.encoder_embed_dim

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # VAE encode
        with torch.no_grad():
            if args.use_cached:
                moments = samples
                posterior = DiagonalGaussianDistribution(moments)
            else:
                posterior = vae.encode(samples.to(torch.bfloat16) if args.bf16 else samples)
            x = posterior.sample().mul_(0.2325)

        # Extract phi features from real data and push to positive bank.
        # Real data path: patchify -> token_to_enc_proj -> phi (frozen penultimate layer)
        if model_without_ddp.feature_encoder is not None:
            with torch.no_grad():
                hw = args.img_size // 16
                x_tokens = model_without_ddp.patchify(x, hw=hw)
                x_enc = model_without_ddp.token_to_enc_proj(x_tokens)  # [B, seq_len, enc_dim]
                phi_real = model_without_ddp.feature_encoder(x_enc)
                extract_and_push_to_banks(phi_real, labels, bank_pos)

        # Check if banks have enough data
        if bank_pos.total_count() < min_bank_samples:
            metric_logger.update(loss=0.0, drift_loss=0.0, reg_loss=0.0, entropy_loss=0.0, alpha=alpha)
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            continue

        # Sample from memory banks
        pos_features, neg_features = sample_from_banks(
            labels, bank_pos, bank_neg,
            args.pos_per_sample, args.neg_per_sample,
            seq_len, feat_dim, device,
        )

        # Forward pass
        processer = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else torch.cuda.amp.autocast()
        with processer:
            losses, info, gen_latents, entropy_map = model_without_ddp(
                x, labels,
                pos_features=pos_features,
                neg_features=neg_features,
                phase2=phase2,
            )

            loss_drift = losses['drift']
            loss_reg = losses['reg']
            loss_entropy = losses['entropy']

            if phase2:
                loss = loss_drift
            else:
                loss = alpha * (loss_reg + loss_entropy) + (1.0 - alpha) * loss_drift

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, skipping step".format(loss_value))
            optimizer.zero_grad()
            metric_logger.update(loss=0.0, drift_loss=0.0, reg_loss=0.0, entropy_loss=0.0, alpha=alpha)
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            continue

        if loss_scaler is not None:
            if (data_iter_step + 1) % args.gradient_accumulation_steps == 0 or (data_iter_step + 1) == len(data_loader):
                loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
                optimizer.zero_grad()
                torch.cuda.synchronize()
                update_ema(ema_params, model_params, rate=args.ema_rate)
            else:
                loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=False)
        else:
            loss.backward()
            if (data_iter_step + 1) % args.gradient_accumulation_steps == 0 or (data_iter_step + 1) == len(data_loader):
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                update_ema(ema_params, model_params, rate=args.ema_rate)

        # Push generated features to negative bank.
        # Gen data path: patchify -> token_to_enc_proj -> phi (same as real data path)
        if model_without_ddp.feature_encoder is not None:
            with torch.no_grad():
                gen_tokens = model_without_ddp.patchify(
                    gen_latents.detach(),
                    hw=hw,
                )
                gen_enc = model_without_ddp.token_to_enc_proj(gen_tokens)  # [B, seq_len, enc_dim]
                phi_gen = model_without_ddp.feature_encoder(gen_enc)
                B = phi_gen.shape[0]
                feat_flat = phi_gen.reshape(B, -1)
                bank_neg.add(feat_flat, labels.cpu() * 0)

        metric_logger.update(loss=loss_value)
        metric_logger.update(drift_loss=loss_drift.item() if torch.is_tensor(loss_drift) else loss_drift)
        metric_logger.update(reg_loss=loss_reg.item() if torch.is_tensor(loss_reg) else loss_reg)
        metric_logger.update(entropy_loss=loss_entropy.item() if torch.is_tensor(loss_entropy) else loss_entropy)
        metric_logger.update(alpha=alpha)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('drift_loss', losses['drift'].item() if torch.is_tensor(losses['drift']) else 0.0, epoch_1000x)
            log_writer.add_scalar('reg_loss', losses['reg'].item() if torch.is_tensor(losses['reg']) else 0.0, epoch_1000x)
            log_writer.add_scalar('entropy_loss', losses['entropy'].item() if torch.is_tensor(losses['entropy']) else 0.0, epoch_1000x)
            log_writer.add_scalar('alpha', alpha, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16,
             log_writer=None, cfg=1.0, use_ema=True):
    """Generate images and compute FID/IS.

    Supports both standard single-step generation and speculative decoding
    (when ``args.use_speculative_decoding`` is True).
    """
    model_without_ddp.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(
        args.output_dir,
        "epoch{}-drift-cfg{}-image{}".format(epoch, cfg, args.num_images),
    )
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    class_num = args.class_num
    assert args.num_images % class_num == 0
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[
            world_size * batch_size * i + local_rank * batch_size :
            world_size * batch_size * i + (local_rank + 1) * batch_size
        ]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        torch.cuda.synchronize()
        start_time = time.time()

        torch.manual_seed(random.randint(0, 1024))
        with torch.no_grad():
            processer = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else torch.cuda.amp.autocast()
            with processer:
                if getattr(args, 'use_speculative_decoding', False):
                    from models.speculative_decoding import speculative_decode
                    sampled_tokens, _info = speculative_decode(
                        target_model=model_without_ddp,
                        draft_model=model_without_ddp.draft_model,
                        labels=labels_gen,
                        gamma=getattr(args, 'spec_gamma', 0.8),
                        cfg=cfg,
                        acceptance_threshold=getattr(args, 'spec_acceptance_threshold', 0.1),
                    )
                else:
                    sampled_tokens = model_without_ddp.samples(cfg=cfg, labels=labels_gen)
                sampled_images = vae.decode(sampled_tokens / 0.2325)

        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(
                gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2

        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].float().numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()
    time.sleep(10)

    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    if log_writer is not None:
        if args.img_size == 256:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        elif args.img_size == 512:
            input2 = None
            fid_statistics_file = 'fid_stats/VIRTUAL_imagenet512.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = ""
        if use_ema:
            postfix = postfix + "_ema"
        if cfg != 1.0:
            postfix = postfix + "_cfg{}".format(cfg)
        postfix = postfix + "_drift"
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        shutil.rmtree(save_folder)

    torch.distributed.barrier()
    time.sleep(10)
