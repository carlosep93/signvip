"""
Compare FSQ code counts for the Phase 1 encoder in fp32 vs bf16.

If fp32 gives ~523 and bf16 gives ~147, bf16 quantization noise is shifting
encoder outputs across FSQ boundaries and destroying code diversity.
In that case Phase 2b should train in fp32.

Usage:
    python check_phase1_fp32_vs_bf16.py \
        --phase1   workspace/vq_phase1_<date>/encoder.bin \
        --config   configs/vq/vq_multicond_RWTH_compress.yaml \
        --n_batches 10
"""
import argparse
import copy

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.condition_encoder import VQConditionEncoder, VQModel
from signdatasets import SignLangVideoDataset
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1",    required=True, help="Phase 1 encoder.bin path")
    p.add_argument("--config",    required=True, help="VQ config yaml")
    p.add_argument("--n_batches", type=int, default=10)
    p.add_argument("--device",    default="cuda")
    return p.parse_args()


def load_vq(cfg, phase1_path, device, dtype):
    vq_cfg = OmegaConf.to_container(cfg.modules.condition_encoder_kwargs.vq_kwargs)

    # Infer backbone output channels
    backbone = VQConditionEncoder(
        conditioning_channels=3,
        image_finetune=True,
        num_conds=2,
        use_vq=False,
    ).to(device).eval()
    backbone.requires_grad_(False)
    state = torch.load(cfg.modules.condition_encoder, map_location="cpu")
    backbone.load_state_dict(state, strict=False)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, *cfg.dataset.frame_size, device=device)
        feat = backbone.encode(dummy, dummy)
        in_ch = feat.shape[1]

    vq = VQModel(
        in_channels=in_ch,
        quantizer_channels=vq_cfg["quantizer_channels"],
        n_e=vq_cfg["n_e"],
        vq_type=vq_cfg["vq_type"],
        fsq_levels=vq_cfg["fsq_levels"],
        ch_mult=vq_cfg["ch_mult"],
        input_size=vq_cfg["input_size"],
    )

    # Load Phase 1 weights (keys are "vq.downsample_encoder.*")
    raw = torch.load(phase1_path, map_location="cpu")
    raw = {k[7:] if k.startswith("module.") else k: v for k, v in raw.items()}
    # Strip leading "vq." to match VQModel's own state dict
    raw = {(k[3:] if k.startswith("vq.") else k): v for k, v in raw.items()}
    missing, unexpected = vq.load_state_dict(raw, strict=False)
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:3]}")

    vq = vq.to(device=device, dtype=dtype).eval()
    return backbone.to(dtype=dtype), vq


@torch.no_grad()
def count_codes(backbone, vq, loader, n_e, device, dtype, n_batches, label):
    counts = torch.zeros(n_e, dtype=torch.long)
    for i, batch in enumerate(tqdm(loader, total=n_batches, desc=label)):
        if i >= n_batches:
            break
        sk = batch["tgt_sk_frames"].to(device=device, dtype=dtype)
        hm = batch["tgt_hamer_frames"].to(device=device, dtype=dtype)
        B, C, F, H, W = sk.shape
        sk_flat = sk.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        hm_flat = hm.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        feat = backbone.encode(sk_flat, hm_flat)
        # Also cast feat in case backbone output dtype differs
        idx = vq.encode(feat.to(dtype=dtype))
        counts += torch.bincount(idx.reshape(-1).cpu().clamp(0, n_e - 1), minlength=n_e)

    used  = (counts > 0).sum().item()
    total = counts.sum().item()
    import numpy as np
    p = counts.float() / total
    entropy = -(p[p > 0] * p[p > 0].log()).sum().item()
    max_entropy = torch.tensor(n_e).float().log().item()
    print(f"\n[{label}]")
    print(f"  Codes used : {used} / {n_e}  ({used/n_e*100:.1f}%)")
    print(f"  Total tokens: {total:,}")
    print(f"  Entropy    : {entropy:.3f} / {max_entropy:.3f} (max)")

    # Per-dim std of encoder output (to check if std≈1.0 is maintained)
    return used, counts


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    seed_everything(cfg.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    n_e = cfg.modules.condition_encoder_kwargs.vq_kwargs.n_e

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
    # Fixed shuffle seed so both runs see the same batches
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(dataset, batch_size=cfg.dataloader.batch_size,
                        shuffle=True, num_workers=4, drop_last=True, generator=g)

    print(f"\nPhase 1 checkpoint : {args.phase1}")
    print(f"Batches            : {args.n_batches}")
    print(f"Batch size         : {cfg.dataloader.batch_size}")
    print(f"Total tokens (est) : {args.n_batches * cfg.dataloader.batch_size * cfg.dataset.num_frames * 4 * 3:,}")

    # ── fp32 ──────────────────────────────────────────────────────────────
    backbone_fp32, vq_fp32 = load_vq(cfg, args.phase1, device, torch.float32)
    g.manual_seed(42)
    loader_fp32 = DataLoader(dataset, batch_size=cfg.dataloader.batch_size,
                             shuffle=True, num_workers=4, drop_last=True, generator=g)
    used_fp32, _ = count_codes(backbone_fp32, vq_fp32, loader_fp32, n_e,
                                device, torch.float32, args.n_batches, "fp32")

    # ── bf16 ──────────────────────────────────────────────────────────────
    backbone_bf16, vq_bf16 = load_vq(cfg, args.phase1, device, torch.bfloat16)
    g.manual_seed(42)
    loader_bf16 = DataLoader(dataset, batch_size=cfg.dataloader.batch_size,
                             shuffle=True, num_workers=4, drop_last=True, generator=g)
    used_bf16, _ = count_codes(backbone_bf16, vq_bf16, loader_bf16, n_e,
                                device, torch.bfloat16, args.n_batches, "bf16")

    # ── verdict ───────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"fp32 codes : {used_fp32} / {n_e}")
    print(f"bf16 codes : {used_bf16} / {n_e}")
    delta = used_fp32 - used_bf16
    if delta > 50:
        print(f"\n→ bf16 loses {delta} codes ({delta/used_fp32*100:.0f}% of fp32 total).")
        print("  Train Phase 2b with weight_dtype: fp32 (or fp16) instead of bf16.")
    elif delta > 10:
        print(f"\n→ Modest bf16 impact ({delta} codes). bf16 is probably still usable.")
    else:
        print(f"\n→ bf16 has negligible impact. The 523→147 drop has a different cause.")


if __name__ == "__main__":
    main()
