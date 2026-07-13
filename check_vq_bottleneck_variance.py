"""
Diagnose why FSQ training collapses to ~12 codes.

Checks per-position variance of backbone features at the 4×3 VQ bottleneck.
If variance per position is near zero → distillation can't force code diversity
(one code per position is the optimal distillation solution).

Run:
    python check_vq_bottleneck_variance.py \
        --config configs/vq/vq_multicond_RWTH_compress.yaml
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.condition_encoder import VQConditionEncoder, VQModel
from signdatasets import SignLangVideoDataset
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n_batches", type=int, default=50,
                   help="Number of batches to sample (covers ~200-400 videos)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    vq_cfg = OmegaConf.to_container(cfg.modules.condition_encoder_kwargs.vq_kwargs)

    # ------------------------------------------------------------------ #
    # Frozen backbone (Stage 1 weights)
    # ------------------------------------------------------------------ #
    backbone = VQConditionEncoder(
        conditioning_channels=3,
        image_finetune=True,
        num_conds=2,
        use_vq=False,
    ).to(device).eval()
    backbone.requires_grad_(False)

    state = torch.load(cfg.modules.condition_encoder, map_location="cpu")
    miss, unexp = backbone.load_state_dict(state, strict=False)
    print(f"Backbone load: missing={len(miss)}, unexpected={len(unexp)}")

    # ------------------------------------------------------------------ #
    # VQ encoder (random init — we only need the spatial downsampling)
    # ------------------------------------------------------------------ #
    in_ch = 256 * 2  # gate_module output channels (backbone[-1] * num_conds)
    # Infer from backbone if possible
    with torch.no_grad():
        dummy = torch.zeros(1, 3, *cfg.dataset.frame_size, device=device)
        feat = backbone.encode(dummy, dummy)
        in_ch = feat.shape[1]
    print(f"Backbone output channels: {in_ch}")

    vq = VQModel(
        in_channels=in_ch,
        quantizer_channels=vq_cfg["quantizer_channels"],
        n_e=vq_cfg["n_e"],
        vq_type=vq_cfg["vq_type"],
        fsq_levels=vq_cfg["fsq_levels"],
        ch_mult=vq_cfg["ch_mult"],
        input_size=vq_cfg["input_size"],
    ).to(device).eval()

    # Find bottleneck spatial size
    with torch.no_grad():
        enc_out = vq.downsample_encoder(feat)
        bh, bw = enc_out.shape[2], enc_out.shape[3]
    print(f"FSQ bottleneck spatial size: {bh}×{bw} = {bh*bw} positions")
    print(f"FSQ codes: {vq_cfg['n_e']}")
    print()

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    dataset = SignLangVideoDataset(
        frame_size=OmegaConf.to_container(cfg.dataset.frame_size),
        frame_scale=OmegaConf.to_container(cfg.dataset.frame_scale),
        frame_ratio=OmegaConf.to_container(cfg.dataset.frame_ratio),
        roots=OmegaConf.to_container(cfg.dataset.roots),
        sk_roots=OmegaConf.to_container(cfg.dataset.sk_roots),
        hamer_roots=OmegaConf.to_container(cfg.dataset.hamer_roots),
        meta_paths=OmegaConf.to_container(cfg.dataset.meta_paths),
        sample_rate=cfg.dataset.sample_rate,
        num_frames=cfg.dataset.num_frames,
        ref_margin=cfg.dataset.ref_margin,
        uncond_ratio=0, mask_ratio=0, mask_thershold=0,
        skip_ratio=0, sk_mask_ratio=0, hamer_mask_ratio=0, both_mask_ratio=0,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True,
                        num_workers=4, drop_last=True)

    # ------------------------------------------------------------------ #
    # Collect encoder outputs across many samples
    # ------------------------------------------------------------------ #
    # shape: (n_samples, bh*bw, quantizer_channels)
    all_enc = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=args.n_batches,
                                       desc="Collecting features")):
            if i >= args.n_batches:
                break

            sk = batch["tgt_sk_frames"].to(device)
            hm = batch["tgt_hamer_frames"].to(device)
            B, C, F, H, W = sk.shape
            sk_flat = sk.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            hm_flat = hm.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)

            feat = backbone.encode(sk_flat, hm_flat)     # (BF, ch, 34, 28)
            enc_out = vq.downsample_encoder(feat)         # (BF, 4, bh, bw)

            # Rearrange to (BF, bh*bw, 4)
            enc_flat = enc_out.permute(0, 2, 3, 1).reshape(B * F, bh * bw, -1)
            all_enc.append(enc_flat.float().cpu())

    all_enc = torch.cat(all_enc, dim=0)   # (N, 12, 4)
    N = all_enc.shape[0]
    print(f"Collected {N} frame-samples")

    # ------------------------------------------------------------------ #
    # Per-position variance analysis
    # ------------------------------------------------------------------ #
    print("\n--- Per-position variance of VQ encoder output ---")
    print(f"{'Pos':>4}  {'var(across samples)':>22}  {'std':>8}  {'pct of total':>14}")
    total_var = all_enc.var(dim=0).mean().item()  # avg across positions & channels
    pos_vars = []
    for pos in range(bh * bw):
        # variance across N samples for this spatial position
        v = all_enc[:, pos, :].var(dim=0).mean().item()
        pos_vars.append(v)
        print(f"{pos:>4}  {v:>22.6f}  {v**0.5:>8.4f}  {v/total_var*100:>13.1f}%")

    pos_vars = np.array(pos_vars)
    print(f"\nOverall mean per-position variance : {pos_vars.mean():.6f}")
    print(f"Overall std  of position variances : {pos_vars.std():.6f}")

    # ------------------------------------------------------------------ #
    # How many UNIQUE codes does a random encoder produce (sanity check)?
    # ------------------------------------------------------------------ #
    print("\n--- Code usage with RANDOM VQ encoder (sanity) ---")
    with torch.no_grad():
        counts = torch.zeros(vq_cfg["n_e"], dtype=torch.long)
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            sk = batch["tgt_sk_frames"].to(device)
            hm = batch["tgt_hamer_frames"].to(device)
            B, C, F, H, W = sk.shape
            sk_flat = sk.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            hm_flat = hm.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            feat = backbone.encode(sk_flat, hm_flat)
            idx = vq.encode(feat)
            counts += torch.bincount(idx.reshape(-1).cpu(), minlength=vq_cfg["n_e"])

    used = (counts > 0).sum().item()
    total = counts.sum().item()
    print(f"Random encoder codes used: {used}/{vq_cfg['n_e']} "
          f"({used/vq_cfg['n_e']*100:.1f}%) over {total:,} tokens")

    # ------------------------------------------------------------------ #
    # Diagnosis
    # ------------------------------------------------------------------ #
    print("\n=== DIAGNOSIS ===")
    low_var_threshold = 0.05
    low_var_positions = (pos_vars < low_var_threshold).sum()
    print(f"Positions with very low variance (<{low_var_threshold}): "
          f"{low_var_positions}/{bh*bw}")
    if low_var_positions > bh * bw * 0.5:
        print("→ POSITION-SPECIFIC COLLAPSE: backbone features at the 4×3 bottleneck")
        print("  have low variance across samples. Distillation alone cannot force")
        print("  code diversity — one code per position minimises MSE loss.")
        print("  Solution: Phase 1 KDE repulsion + Phase 2a (decoder only).")
    else:
        print("→ Feature variance is sufficient. The collapse is a training issue,")
        print("  not a data issue. Check gradient flow to the VQ encoder.")
        print("  Possible causes: wrong LR, wrong checkpoint loaded, gradient vanishing.")


if __name__ == "__main__":
    main()
