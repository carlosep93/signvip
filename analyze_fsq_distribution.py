"""
Runs FSQ VAE inference over a dataset split and plots the distribution
of discrete codebook indices.
"""

import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.condition_encoder import VQConditionEncoder
from signdatasets import SignLangVideoDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="VQ config YAML (e.g. configs/vq/vq_multicond_RWTH_compress.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved vq model.bin (e.g. workspace/.../best/condition_encoder/model.bin)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"],
                        help="Dataset split to analyze")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Cap number of batches (useful for quick runs)")
    parser.add_argument("--output_dir", type=str, default="fsq_analysis",
                        help="Directory to save plots")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_encoder(cfg, checkpoint_path, device):
    vq_kwargs = OmegaConf.to_container(cfg.modules.condition_encoder_kwargs.vq_kwargs)
    encoder = VQConditionEncoder(
        conditioning_channels=3,
        image_finetune=True,   # no motion modules needed — we process frame-by-frame
        num_conds=2,
        use_vq=True,
        vq_kwargs=vq_kwargs,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    # strip any "module." prefix from DDP/DeepSpeed saves
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    encoder = encoder.to(device).eval()
    return encoder


def build_dataset(cfg, split):
    dataset_cfg = cfg.dataset
    if split == "val":
        meta_paths = OmegaConf.to_container(cfg.validation_data.meta_paths)
    else:
        meta_paths = OmegaConf.to_container(dataset_cfg.meta_paths)

    return SignLangVideoDataset(
        frame_size=OmegaConf.to_container(dataset_cfg.frame_size),
        frame_scale=OmegaConf.to_container(dataset_cfg.frame_scale),
        frame_ratio=OmegaConf.to_container(dataset_cfg.frame_ratio),
        roots=OmegaConf.to_container(dataset_cfg.roots),
        sk_roots=OmegaConf.to_container(dataset_cfg.sk_roots),
        hamer_roots=OmegaConf.to_container(dataset_cfg.hamer_roots),
        meta_paths=meta_paths,
        sample_rate=dataset_cfg.sample_rate,
        num_frames=dataset_cfg.num_frames,
        ref_margin=dataset_cfg.ref_margin,
        uncond_ratio=0,
        mask_ratio=0,
        mask_thershold=0,
        skip_ratio=0,
        sk_mask_ratio=0,
        hamer_mask_ratio=0,
        both_mask_ratio=0,
    )


@torch.no_grad()
def collect_indices(encoder, dataloader, device, codebook_size, max_batches=None):
    counts = torch.zeros(codebook_size, dtype=torch.long)
    n_frames = 0

    for i, batch in enumerate(tqdm(dataloader, desc="Encoding")):
        if max_batches is not None and i >= max_batches:
            break

        sk = batch["tgt_sk_frames"].to(device)    # (B, C, F, H, W)
        hamer = batch["tgt_hamer_frames"].to(device)

        B, C, F, H, W = sk.shape
        # flatten batch and frames for frame-by-frame encoding
        sk_flat = sk.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        hamer_flat = hamer.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)

        # encode → indices of shape (B*F, H'*W')
        indices = encoder.encode(sk_flat, hamer_flat, return_indices=True)
        indices = indices.reshape(-1).cpu()

        counts += torch.bincount(indices, minlength=codebook_size)
        n_frames += B * F

    return counts, n_frames


def plot_distribution(counts, codebook_size, fsq_levels, output_dir, split):
    os.makedirs(output_dir, exist_ok=True)
    counts_np = counts.numpy()
    total = counts_np.sum()
    used = (counts_np > 0).sum()
    utilization = used / codebook_size * 100

    print(f"\n=== FSQ Codebook Analysis ({split} split) ===")
    print(f"  Codebook size : {codebook_size}")
    print(f"  Entries used  : {used} / {codebook_size}  ({utilization:.1f}%)")
    print(f"  Total tokens  : {total:,}")
    print(f"  Max count     : {counts_np.max():,}  (index {counts_np.argmax()})")
    print(f"  Min count (used): {counts_np[counts_np > 0].min():,}")
    print(f"  Entropy       : {-(counts_np/total * np.log(counts_np/total + 1e-10)).sum():.3f} "
          f"/ {np.log(codebook_size):.3f} (max)")

    # --- 1. Full index distribution bar chart ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(codebook_size), counts_np, width=1.0, color="steelblue", alpha=0.8)
    ax.set_xlabel("Codebook index")
    ax.set_ylabel("Count")
    ax.set_title(f"FSQ codebook usage — {split} split  "
                 f"({used}/{codebook_size} entries used, {utilization:.1f}%)")
    ax.axhline(total / codebook_size, color="red", linestyle="--",
               linewidth=1, label="uniform baseline")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"fsq_distribution_{split}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- 2. Sorted distribution (most → least used) ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(codebook_size), np.sort(counts_np)[::-1],
           width=1.0, color="darkorange", alpha=0.8)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Count")
    ax.set_title("FSQ codebook usage — sorted by frequency")
    ax.axhline(total / codebook_size, color="red", linestyle="--",
               linewidth=1, label="uniform baseline")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"fsq_distribution_sorted_{split}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- 3. Per-level marginal distributions ---
    levels = fsq_levels
    n_levels = len(levels)
    fig, axes = plt.subplots(1, n_levels, figsize=(4 * n_levels, 4))
    indices_all = np.arange(codebook_size)
    # decode each codebook index back to per-level values
    level_indices = np.zeros((codebook_size, n_levels), dtype=np.int32)
    for l, lv in enumerate(levels):
        basis = int(np.prod(levels[:l]))
        level_indices[:, l] = (indices_all // basis) % lv

    for l, ax in enumerate(axes):
        lv = levels[l]
        level_counts = np.zeros(lv, dtype=np.float64)
        for code_idx in range(codebook_size):
            level_counts[level_indices[code_idx, l]] += counts_np[code_idx]
        ax.bar(np.arange(lv), level_counts, color="seagreen", alpha=0.8)
        ax.set_xlabel(f"Level {l} value")
        ax.set_ylabel("Count")
        ax.set_title(f"Dimension {l}  (L={lv})")
        ax.axhline(level_counts.sum() / lv, color="red", linestyle="--", linewidth=1)

    fig.suptitle("Per-dimension marginal distributions", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f"fsq_per_level_{split}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- 4. CDF — how many entries cover X% of tokens ---
    sorted_counts = np.sort(counts_np)[::-1]
    cdf = np.cumsum(sorted_counts) / total
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, codebook_size + 1), cdf * 100, color="steelblue")
    ax.axhline(90, color="red", linestyle="--", linewidth=1, label="90%")
    ax.axhline(99, color="orange", linestyle="--", linewidth=1, label="99%")
    for pct in [90, 99]:
        n_entries = int(np.searchsorted(cdf, pct / 100)) + 1
        ax.axvline(n_entries, color="gray", linestyle=":", linewidth=0.8)
        ax.text(n_entries + 2, pct - 3, f"{n_entries}", fontsize=8, color="gray")
    ax.set_xlabel("Number of top-k codebook entries (ranked)")
    ax.set_ylabel("% of total tokens covered")
    ax.set_title("Codebook coverage CDF")
    ax.legend()
    ax.set_xlim(0, codebook_size)
    ax.set_ylim(0, 101)
    plt.tight_layout()
    path = os.path.join(output_dir, f"fsq_coverage_cdf_{split}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # save raw counts for later use
    np.save(os.path.join(output_dir, f"fsq_counts_{split}.npy"), counts_np)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vq_kwargs = OmegaConf.to_container(cfg.modules.condition_encoder_kwargs.vq_kwargs)
    codebook_size = vq_kwargs["n_e"]
    fsq_levels = vq_kwargs["fsq_levels"]

    print(f"Loading encoder from {args.checkpoint} ...")
    encoder = load_encoder(cfg, args.checkpoint, device)

    print(f"Building {args.split} dataset ...")
    dataset = build_dataset(cfg, args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    counts, n_frames = collect_indices(
        encoder, dataloader, device, codebook_size, args.max_batches
    )
    print(f"\nProcessed {n_frames:,} frames.")

    plot_distribution(counts, codebook_size, fsq_levels, args.output_dir, args.split)


if __name__ == "__main__":
    main()
