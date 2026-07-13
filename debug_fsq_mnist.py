"""
Debug FSQ codebook utilization on MNIST using the exact VQModel code from this repo.

Purpose: verify whether FSQ collapse is a fundamental issue with the implementation
or specific to the sign language training setup (diffusion loss, appearance encoder, etc.)

If FSQ uses all 625 codes on MNIST with simple MSE loss → the implementation is fine
and collapse in sign language is due to weak/absent VQ gradient in the diffusion pipeline.
If FSQ collapses on MNIST too → there is a bug in the FSQ/VQModel code itself.

Usage:
    python debug_fsq_mnist.py
    python debug_fsq_mnist.py --epochs 20 --lr 1e-3 --skip_vq   # continuous (no FSQ)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    p.add_argument("--batch_size",  type=int,   default=256)
    p.add_argument("--ch",          type=int,   default=32,
                   help="Base channels for Encoder/Decoder (default 32; sign-lang uses 128)")
    p.add_argument("--ch_mult",     type=int,   nargs="+", default=[1, 2, 4],
                   help="Channel multipliers → spatial downsampling steps")
    p.add_argument("--fsq_levels",  type=int,   nargs="+", default=[5, 5, 5, 5],
                   help="FSQ levels per dimension (default [5,5,5,5] → 625 codes)")
    p.add_argument("--skip_vq",     action="store_true",
                   help="Bypass FSQ (train as continuous autoencoder, for comparison)")
    p.add_argument("--output_dir",  default="debug_fsq_mnist")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# analysis                                                                     #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def count_codes(model, loader, n_e, device):
    """Run full dataset through encoder and count distinct FSQ codes used."""
    counts = torch.zeros(n_e, dtype=torch.long)
    model.eval()
    for x, _ in loader:
        x = x.to(device)
        # use VQModel.encode() which returns indices
        indices = model.encode(x)          # (B, H'*W')
        indices = indices.reshape(-1).cpu()
        counts += torch.bincount(indices.clamp(0, n_e - 1), minlength=n_e)
    model.train()
    return counts


def print_analysis(counts, n_e, fsq_levels, tag=""):
    counts_np = counts.numpy()
    total = counts_np.sum()
    used  = (counts_np > 0).sum()
    print(f"\n{'='*50}  {tag}")
    print(f"  Codes used : {used} / {n_e}  ({used/n_e*100:.1f}%)")
    print(f"  Total tokens: {total:,}")
    if total > 0:
        print(f"  Max count  : {counts_np.max():,}  (index {counts_np.argmax()})")
        print(f"  Entropy    : {-(counts_np/total * np.log(counts_np/total + 1e-10)).sum():.3f}"
              f" / {np.log(n_e):.3f} (max)")
    return used


def save_reconstructions(model, loader, device, path, n=8):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    with torch.no_grad():
        recon, _, _ = model(x)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=8)
    axes[1, 0].set_ylabel("Recon", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    model.train()


def plot_code_distribution(counts, n_e, path):
    counts_np = counts.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].bar(np.arange(n_e), counts_np, width=1, color="steelblue", alpha=0.8)
    axes[0].axhline(counts_np.sum() / n_e, color="red", linestyle="--", linewidth=1,
                    label="uniform")
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
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    n_e = int(np.prod(args.fsq_levels))
    input_size = (28, 28)

    print(f"FSQ levels : {args.fsq_levels}  →  {n_e} codes")
    print(f"ch={args.ch}, ch_mult={args.ch_mult}")
    print(f"skip_vq={args.skip_vq}  (continuous mode — no FSQ)" if args.skip_vq else
          f"skip_vq=False  (FSQ active)")

    # ------------------------------------------------------------------ #
    # data                                                                #
    # ------------------------------------------------------------------ #
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
    ])
    train_ds = datasets.MNIST("~/.cache/mnist", train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------ #
    # model — exact VQModel from the repo                                 #
    # ------------------------------------------------------------------ #
    model = VQModel(
        vq_type="FSQ",
        n_e=n_e,
        in_channels=1,           # grayscale MNIST
        quantizer_channels=4,
        ch_mult=tuple(args.ch_mult),
        input_size=input_size,
        skip_vq=args.skip_vq,
        fsq_levels=args.fsq_levels,
    ).to(device)

    # patch Encoder ch (default 128 → use args.ch for MNIST)
    # rebuild with smaller ch by monkey-patching is complex; instead just use default ch=128
    # (28×28 with ch=128 is fine for debugging)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")

    # spatial size after encoder
    from models.vq.basic_vae import Encoder as _Enc
    dummy_enc = _Enc(z_channels=4, in_channels=1, ch_mult=tuple(args.ch_mult),
                     input_size=input_size)
    with torch.no_grad():
        dummy_out = dummy_enc(torch.zeros(1, 1, *input_size))
    h_out, w_out = dummy_out.shape[2], dummy_out.shape[3]
    print(f"Encoder output spatial: {h_out}×{w_out} = {h_out*w_out} tokens/image")

    # ------------------------------------------------------------------ #
    # training                                                            #
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    history = {"loss": [], "codes_used": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{args.epochs}")

        for x, _ in pbar:
            x = x.to(device)
            recon, perplexity, emb_loss = model(x)
            loss = F.mse_loss(recon, x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             perplexity=f"{perplexity.item():.1f}")

        avg_loss = epoch_loss / len(train_loader)
        history["loss"].append(avg_loss)

        # code analysis every 5 epochs and at the end
        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            if not args.skip_vq:
                counts = count_codes(model, val_loader, n_e, device)
                used = print_analysis(counts, n_e, args.fsq_levels,
                                      tag=f"Epoch {epoch}  loss={avg_loss:.4f}")
                history["codes_used"].append((epoch, used))
                plot_code_distribution(
                    counts, n_e,
                    os.path.join(args.output_dir, f"codes_epoch{epoch:02d}.png")
                )
            save_reconstructions(
                model, val_loader, device,
                os.path.join(args.output_dir, f"recon_epoch{epoch:02d}.png")
            )

    # ------------------------------------------------------------------ #
    # final analysis                                                       #
    # ------------------------------------------------------------------ #
    if not args.skip_vq:
        counts = count_codes(model, val_loader, n_e, device)
        print_analysis(counts, n_e, args.fsq_levels, tag="FINAL (val)")
        counts_train = count_codes(model, train_loader, n_e, device)
        print_analysis(counts_train, n_e, args.fsq_levels, tag="FINAL (train)")
        plot_code_distribution(
            counts, n_e,
            os.path.join(args.output_dir, "codes_final.png")
        )
        np.save(os.path.join(args.output_dir, "codes_final.npy"), counts.numpy())

    # loss curve
    plt.figure(figsize=(8, 3))
    plt.plot(history["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss.png"), dpi=120)
    plt.close()

    if history["codes_used"]:
        epochs_logged, codes_logged = zip(*history["codes_used"])
        plt.figure(figsize=(8, 3))
        plt.plot(epochs_logged, codes_logged, marker="o")
        plt.axhline(n_e, color="red", linestyle="--", label=f"max ({n_e})")
        plt.xlabel("Epoch")
        plt.ylabel("Codes used")
        plt.title("FSQ codebook utilization over training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "utilization.png"), dpi=120)
        plt.close()

    print(f"\nOutputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
