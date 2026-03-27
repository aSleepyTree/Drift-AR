"""
Entry point for Drift-AR training and evaluation.

Implements the full Drift-AR paper with:
  - Two-phase annealed training (Eq. 9-10)
  - Entropy-parameterized prior (Eq. 7)
  - Draft AR model for speculative decoding
  - PenultimateLayerEncoder for drift loss features
"""

import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr, rand_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder

from models.vae import AutoencoderKL
from models import drift_ar
import copy


def get_args_parser():
    parser = argparse.ArgumentParser('Drift-AR training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=400, type=int)

    # Model parameters
    parser.add_argument('--model', default='drift_ar_large', type=str, metavar='MODEL',
                        help='Name of model to train')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str)
    parser.add_argument('--vae_embed_dim', default=16, type=int)
    parser.add_argument('--patch_size', default=1, type=int)

    # Generation parameters
    parser.add_argument('--num_images', default=50000, type=int)
    parser.add_argument('--cfg', default=1.0, type=float, help="classifier-free guidance for evaluation")
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--save_last_freq', type=int, default=5)
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR')
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N')
    parser.add_argument('--ema_rate', default=0.9999, type=float)
    parser.add_argument('--grad_clip', type=float, default=2.0)

    # Dropout
    parser.add_argument('--attn_dropout', type=float, default=0.1)
    parser.add_argument('--proj_dropout', type=float, default=0.1)
    parser.add_argument('--buffer_size', type=int, default=64)

    # Drift-specific parameters
    parser.add_argument('--R_list', nargs='+', type=float, default=[0.02, 0.05, 0.2],
                        help='Multi-scale kernel bandwidth values for drift loss')
    parser.add_argument('--pos_per_sample', type=int, default=64,
                        help='Number of positive samples from memory bank per label')
    parser.add_argument('--neg_per_sample', type=int, default=16,
                        help='Number of negative samples from memory bank per label')
    parser.add_argument('--pos_bank_size', type=int, default=128,
                        help='Per-class positive memory bank size')
    parser.add_argument('--neg_bank_size', type=int, default=1000,
                        help='Global negative memory bank size')

    # Entropy parameters (Eq. 6)
    parser.add_argument('--sigma_max', type=float, default=0.5,
                        help='Maximum variance for entropy-parameterized prior (Eq. 6)')
    parser.add_argument('--tau_sigma', type=float, default=2.0,
                        help='Temperature for entropy-to-variance mapping (Eq. 6)')

    # Two-phase annealing (Eq. 9-10)
    parser.add_argument('--alpha_0', type=float, default=0.95,
                        help='Initial alpha for two-phase annealing (Eq. 9)')
    parser.add_argument('--T_freeze_frac', type=float, default=0.8,
                        help='Fraction of total epochs for Phase I (Eq. 10)')

    # Draft model
    parser.add_argument('--draft_depth', type=int, default=6,
                        help='Number of transformer blocks in draft AR model')

    # Speculative decoding (inference only)
    parser.add_argument('--use_speculative_decoding', action='store_true',
                        help='Use speculative decoding during evaluation (Sec 4.3)')
    parser.set_defaults(use_speculative_decoding=False)
    parser.add_argument('--spec_acceptance_threshold', type=float, default=0.1,
                        help='L2 distance threshold for accepting draft tokens')
    parser.add_argument('--spec_gamma', type=float, default=0.8,
                        help='Multiplier for context-aware entropy threshold')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str)
    parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # Caching latents
    parser.add_argument('--use_cached', action='store_true', dest='use_cached')
    parser.set_defaults(use_cached=False)
    parser.add_argument('--cached_path', default='')

    # bf16
    parser.add_argument('--bf16', action='store_true')
    parser.set_defaults(bf16=False)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_wo_ema', action='store_true')
    parser.set_defaults(eval_wo_ema=False)
    parser.add_argument('--load_step', default='last', type=str)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.bf16:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cudnn.allow_bf16_reduced_precision_reduction = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Data augmentation
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: rand_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if not args.evaluate:
        if args.use_cached:
            dataset_train = CachedFolder(args.cached_path)
        else:
            dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        print(dataset_train)

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    # VAE
    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
    if args.bf16:
        vae = vae.to(dtype=torch.bfloat16)
    for param in vae.parameters():
        param.requires_grad = False

    # Model
    model = drift_ar.__dict__[args.model](
        img_size=args.img_size,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        grad_checkpointing=args.grad_checkpointing,
        R_list=tuple(args.R_list),
        sigma_max=args.sigma_max,
        tau_sigma=args.tau_sigma,
        draft_depth=args.draft_depth,
    )

    # Initialize PenultimateLayerEncoder (phi) from target encoder
    model.init_feature_encoder()
    print("PenultimateLayerEncoder (phi) initialized from target encoder penultimate block")

    # Initialize draft model from target encoder's first blocks
    model.init_draft_from_encoder()
    print(f"DraftAR ({args.draft_depth} blocks) initialized from target encoder")

    from engine_drift import train_one_epoch, evaluate, init_feature_banks

    print("Model = %s" % str(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    if args.bf16:
        model = model.to(dtype=torch.bfloat16)
    for name, param in model.named_parameters():
        if 'feature_encoder' not in name:
            param.requires_grad = True
    model_without_ddp = model

    eff_batch_size = args.batch_size * misc.get_world_size() * args.gradient_accumulation_steps

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    if args.bf16:
        loss_scaler = None
    else:
        loss_scaler = NativeScaler()

    # Memory Banks (single pair for phi-space features)
    feat_dim = model_without_ddp.encoder_embed_dim
    bank_pos, bank_neg = init_feature_banks(feat_dim, args)
    print(f"Memory banks initialized: pos_bank_size={args.pos_bank_size}, neg_bank_size={args.neg_bank_size}")

    # Resume
    if args.resume and os.path.exists(os.path.join(args.resume, f"checkpoint-{args.load_step}.pth")):
        checkpoint = torch.load(os.path.join(args.resume, f"checkpoint-{args.load_step}.pth"), map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        model_params = list(model_without_ddp.parameters())
        ema_state_dict = checkpoint['model_ema'] if checkpoint.get('model_ema') else checkpoint['model']
        ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        checkpoint_epoch = checkpoint.get('epoch', 0)
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and (not args.evaluate):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint and loss_scaler is not None:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        del checkpoint

        # Re-init phi after loading weights
        model_without_ddp.init_feature_encoder()
    else:
        model_params = list(model_without_ddp.parameters())
        ema_params = copy.deepcopy(model_params)
        print("Training from scratch")

    # Evaluate
    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, vae, ema_params, args, checkpoint_epoch,
                 batch_size=args.eval_bsz, log_writer=log_writer,
                 cfg=args.cfg, use_ema=not args.eval_wo_ema)
        return

    # Training
    print(f"Start training for {args.epochs} epochs")
    print(f"Two-phase schedule: alpha_0={args.alpha_0}, T_freeze_frac={args.T_freeze_frac}")
    print(f"Entropy prior: sigma_max={args.sigma_max}, tau_sigma={args.tau_sigma}")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        print(f"Start training the {epoch}th epoch")

        train_one_epoch(
            model, vae,
            bank_pos, bank_neg,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        # Save checkpoint
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                            ema_params=ema_params, epoch_name="last")

        # Online evaluation
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, vae, ema_params, args, epoch,
                     batch_size=args.eval_bsz, log_writer=log_writer,
                     cfg=1.0, use_ema=True)
            if args.cfg != 1.0:
                evaluate(model_without_ddp, vae, ema_params, args, epoch,
                         batch_size=args.eval_bsz // 2, log_writer=log_writer,
                         cfg=args.cfg, use_ema=True)
            torch.cuda.empty_cache()

        # Save periodic checkpoint
        if epoch > 0 and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                            ema_params=ema_params, epoch_name=str(epoch))

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
