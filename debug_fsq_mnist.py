"""
Debug FSQ codebook utilization on MNIST using the exact VQModel code from this repo,
running under accelerate + DeepSpeed to test whether DeepSpeed causes FSQ collapse.

Single GPU (no DeepSpeed):
    python debug_fsq_mnist.py

Multi-GPU with DeepSpeed ZeRO-2 bf16 (same as real training):
    accelerate launch --config_file accelerate_config_bf16.yaml \
        --num_processes 2 --gpu_ids "0,1" \
        debug_fsq_mnist.py

Multi-GPU with DeepSpeed ZeRO-2 fp16:
    accelerate launch --config_file accelerate_config.yaml \
        --num_processes 2 --gpu_ids "0,1" \
        debug_fsq_mnist.py

Interpretation:
  - Collapse under DeepSpeed but not plain Python → DeepSpeed is causing the issue
  - Collapse under both                           → bug in FSQ/VQModel code
  - No collapse under either                      → implementation is fine;
                                                    collapse in real training is due to
                                                    weak VQ gradient from diffusion loss
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.condition_encoder import VQModel


# --------------------------------------------------------------------------- #
# args                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--batch_size",  type=int,   default=256,
                   help="Per-process batch size")
    p.add_argument("--fsq_levels",  type=int,   nargs="+", default=[5, 5, 5, 5])
    p.add_argument("--ch_mult",     type=int,   nargs="+", default=[1, 2, 4])
    p.add_argument("--skip_vq",     action="store_true",
                   help="Bypass FSQ — train as continuous AE for comparison")
    p.add_argument("--output_dir",  default="debug_fsq_mnist")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# analysis (runs on CPU, gathered from all processes)                          #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def count_codes_local(raw_model, loader, n_e, device, dtype):
    """Count FSQ code usage — runs only on rank 0, uses a non-distributed loader.
    Does NOT touch model train/eval mode to avoid interfering with DeepSpeed state."""
    counts = torch.zeros(n_e, dtype=torch.long)
    for x, _ in loader:
        x = x.to(device=device, dtype=dtype)
        indices = raw_model.encode(x)          # (B, H'*W')
        flat = indices.reshape(-1).cpu().clamp(0, n_e - 1)
        counts += torch.bincount(flat, minlength=n_e)
    return counts


def print_analysis(counts, n_e, tag=""):
    counts_np = counts.numpy()
    total = counts_np.sum()
    used  = (counts_np > 0).sum()
    print(f"\n{'='*50}  {tag}")
    print(f"  Codes used  : {used} / {n_e}  ({used/n_e*100:.1f}%)")
    print(f"  Total tokens: {total:,}")
    if total > 0:
        print(f"  Max count   : {counts_np.max():,}  (index {counts_np.argmax()})")
        print(f"  Entropy     : {-(counts_np/total * np.log(counts_np/total + 1e-10)).sum():.3f}"
              f" / {np.log(n_e):.3f} (max)")
    return int(used)


@torch.no_grad()
def save_reconstructions(raw_model, loader, device, dtype, path, n=8):
    """Saves input/reconstruction pairs — runs only on rank 0."""
    x, _ = next(iter(loader))
    x = x[:n].to(device=device, dtype=dtype)
    recon, _, _ = raw_model(x)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].float().cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0].float().cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Input",  fontsize=8)
    axes[1, 0].set_ylabel("Recon",  fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_code_distribution(counts, n_e, path):
    counts_np = counts.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].bar(np.arange(n_e), counts_np, width=1, color="steelblue", alpha=0.8)
    axes[0].axhline(counts_np.sum() / max(n_e, 1), color="red", linestyle="--",
                    linewidth=1, label="uniform")
    axes[0].set_title("Code distribution")
    axes[0].legend()
    axes[1].bar(np.arange(n_e), np.sort(counts_np)[::-1],
                width=1, color="darkorange", alpha=0.8)
    axes[1].set_title("Sorted by frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    args   = parse_args()
    accel  = Accelerator()
    device = accel.device
    n_e    = int(np.prod(args.fsq_levels))

    # match the dtype that DeepSpeed will cast model weights to
    mp = accel.mixed_precision
    weight_dtype = torch.bfloat16 if mp == "bf16" else (torch.float16 if mp == "fp16" else torch.float32)

    if accel.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"DeepSpeed  : {accel.state.deepspeed_plugin is not None}")
        print(f"Mixed prec : {accel.mixed_precision}")
        print(f"Num procs  : {accel.num_processes}")
        print(f"FSQ levels : {args.fsq_levels}  →  {n_e} codes")
        print(f"skip_vq    : {args.skip_vq}")

    # ------------------------------------------------------------------ #
    # data                                                                #
    # ------------------------------------------------------------------ #
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_ds = datasets.MNIST("~/.cache/mnist", train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    # Separate non-distributed loader for rank-0-only analysis.
    # Never passed to accel.prepare() so it doesn't participate in collective ops.
    analysis_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------ #
    # model                                                               #
    # ------------------------------------------------------------------ #
    model = VQModel(
        vq_type="FSQ",
        n_e=n_e,
        in_channels=1,
        quantizer_channels=4,
        ch_mult=tuple(args.ch_mult),
        input_size=(28, 28),
        skip_vq=args.skip_vq,
        fsq_levels=args.fsq_levels,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=1e-4)

    model, optimizer, train_loader, val_loader = accel.prepare(
        model, optimizer, train_loader, val_loader
    )
    raw_model = accel.unwrap_model(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    # ------------------------------------------------------------------ #
    # training loop                                                       #
    # ------------------------------------------------------------------ #
    history = {"loss": [], "codes_used": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{args.epochs}",
                    disable=not accel.is_main_process)

        for x, _ in pbar:
            x = x.to(dtype=weight_dtype)
            with accel.accumulate(model):
                recon, perplexity, emb_loss = model(x)
                loss = F.mse_loss(recon.float(), x.float())
                accel.backward(loss)
                if accel.sync_gradients:
                    accel.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.detach().float().item()
            if accel.is_main_process:
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 perplexity=f"{perplexity.item():.1f}")

        avg_loss = epoch_loss / len(train_loader)
        history["loss"].append(avg_loss)

        # Analysis runs only on rank 0 using the non-distributed analysis_loader.
        # No collective ops here — avoids cross-process deadlocks with DeepSpeed.
        if accel.is_main_process and (epoch % 5 == 0 or epoch == args.epochs or epoch == 1):
            if not args.skip_vq:
                counts = count_codes_local(raw_model, analysis_loader, n_e, device, weight_dtype)
                used = print_analysis(counts, n_e,
                                      tag=f"Epoch {epoch}  loss={avg_loss:.4f}")
                history["codes_used"].append((epoch, used))
                plot_code_distribution(
                    counts, n_e,
                    os.path.join(args.output_dir, f"codes_epoch{epoch:02d}.png")
                )
            save_reconstructions(
                raw_model, analysis_loader, device, weight_dtype,
                os.path.join(args.output_dir, f"recon_epoch{epoch:02d}.png")
            )

    # ------------------------------------------------------------------ #
    # final analysis & plots (main process only)                         #
    # ------------------------------------------------------------------ #
    if accel.is_main_process:
        if not args.skip_vq:
            counts = count_codes_local(raw_model, analysis_loader, n_e, device, weight_dtype)
            print_analysis(counts, n_e, tag="FINAL (val)")
            plot_code_distribution(
                counts, n_e,
                os.path.join(args.output_dir, "codes_final.png")
            )
            np.save(os.path.join(args.output_dir, "codes_final.npy"), counts.numpy())

        plt.figure(figsize=(8, 3))
        plt.plot(history["loss"])
        plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "loss.png"), dpi=120)
        plt.close()

        if history["codes_used"]:
            ep, cu = zip(*history["codes_used"])
            plt.figure(figsize=(8, 3))
            plt.plot(ep, cu, marker="o")
            plt.axhline(n_e, color="red", linestyle="--", label=f"max ({n_e})")
            plt.xlabel("Epoch"); plt.ylabel("Codes used")
            plt.title("FSQ codebook utilization")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "utilization.png"), dpi=120)
            plt.close()

        print(f"\nOutputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
