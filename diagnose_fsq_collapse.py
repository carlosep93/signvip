"""
Diagnose FSQ codebook collapse by checking code distribution at init
using REAL backbone features (not simulated noise).

If init coverage is low  → backbone features lack spatial diversity at the FSQ scale.
If init coverage is high → training dynamics are collapsing diversity.

Run:
    python diagnose_fsq_collapse.py \
        --config configs/vq/vq_multicond_RWTH_compress.yaml \
        --backbone_ckpt workspace/stage_1_multicond_RWTH/best/condition_encoder/model.bin \
        [--motion_ckpt workspace/stage_2_RWTH/.../best/condition_encoder/model.bin]
"""
import argparse
import math
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.condition_encoder import VQConditionEncoder, VQModel
from signdatasets import SignLangVideoDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--backbone_ckpt", required=True,
                   help="Stage 1 condition_encoder/model.bin")
    p.add_argument("--motion_ckpt", default=None,
                   help="Stage 2 motion module model.bin (optional)")
    p.add_argument("--n_batches", type=int, default=20,
                   help="Batches to evaluate (more = more reliable)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vq_cfg = OmegaConf.to_container(cfg.modules.condition_encoder_kwargs.vq_kwargs)

    # ------------------------------------------------------------------ #
    # 1. Load backbone (frozen, no VQ)
    # ------------------------------------------------------------------ #
    backbone = VQConditionEncoder(
        conditioning_channels=3,
        image_finetune=True,
        num_conds=2,
        use_vq=False,
    ).to(device).eval()

    state = torch.load(args.backbone_ckpt, map_location="cpu")
    if args.motion_ckpt:
        motion_state = torch.load(args.motion_ckpt, map_location="cpu")
        state.update(motion_state)
    miss, unexp = backbone.load_state_dict(state, strict=False)
    print(f"Backbone loaded — missing: {len(miss)}, unexpected: {len(unexp)}")

    # ------------------------------------------------------------------ #
    # 2. Fresh VQ model (random init) with correct in_channels
    # ------------------------------------------------------------------ #
    in_ch = 256 * 2  # backbone_channels[-1] * num_conds = 512
    vq = VQModel(
        in_channels=in_ch,
        quantizer_channels=vq_cfg["quantizer_channels"],
        n_e=vq_cfg["n_e"],
        vq_type=vq_cfg["vq_type"],
        fsq_levels=vq_cfg["fsq_levels"],
        ch_mult=vq_cfg["ch_mult"],
        input_size=vq_cfg["input_size"],
    ).to(device).eval()
    print(f"VQ model: in_channels={in_ch}, quantizer_channels={vq_cfg['quantizer_channels']}")

    # ------------------------------------------------------------------ #
    # 3. Real data loader
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
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)

    # ------------------------------------------------------------------ #
    # 4. Collect backbone features & FSQ indices
    # ------------------------------------------------------------------ #
    n_e = vq_cfg["n_e"]
    counts = torch.zeros(n_e, dtype=torch.long)
    all_enc_outputs = []   # pre-LayerNorm encoder output, for std analysis
    all_backbone_feats = []

    print(f"\nRunning {args.n_batches} batches on real data ...")
    for i, batch in enumerate(tqdm(loader)):
        if i >= args.n_batches:
            break

        sk = batch["tgt_sk_frames"].to(device)
        hamer = batch["tgt_hamer_frames"].to(device)
        B, C, F, H, W = sk.shape
        sk_flat = sk.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        hm_flat = hamer.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)

        # Backbone features (512-ch, 34×28)
        feat = backbone.encode(sk_flat, hm_flat)   # (BF, 512, 34, 28)
        all_backbone_feats.append(feat.cpu().float())

        # FSQ codes from fresh (random) VQ encoder
        idx = vq.encode(feat)   # (BF, h'*w')
        counts += torch.bincount(idx.reshape(-1).cpu(), minlength=n_e)

        # Encoder output BEFORE LayerNorm (to check magnitude)
        enc_raw = vq.downsample_encoder(feat)  # (BF, 4, h', w')
        all_enc_outputs.append(enc_raw.cpu().float())

    total_tokens = counts.sum().item()
    used = (counts > 0).sum().item()
    expected_uniform = n_e * (1 - ((n_e - 1) / n_e) ** total_tokens)
    pct_of_uniform = used / expected_uniform * 100
    entropy = -(counts.float() / total_tokens *
                torch.log(counts.float() / total_tokens + 1e-10)).sum().item()

    # Backbone feature stats
    bf = torch.cat(all_backbone_feats, dim=0)   # (N, 512, 34, 28)
    enc_out = torch.cat(all_enc_outputs, dim=0)  # (N, 4, h', w')

    print("\n" + "=" * 60)
    print("BACKBONE FEATURE STATISTICS (input to VQ encoder)")
    print(f"  Shape          : {tuple(bf.shape)}")
    print(f"  Global std     : {bf.std():.4f}")
    # Patch-level std: average each backbone token's 4×4 region → approximate FSQ granularity
    h_prime, w_prime = enc_out.shape[2], enc_out.shape[3]
    scale_h = bf.shape[2] // h_prime
    scale_w = bf.shape[3] // w_prime
    bf_patches = bf.unfold(2, scale_h, scale_h).unfold(3, scale_w, scale_w)
    # bf_patches: (N, 512, h', w', scale_h, scale_w)
    bf_patch_mean = bf_patches.mean(dim=(-1, -2))   # (N, 512, h', w')
    print(f"  Patch-avg std  : {bf_patch_mean.std():.4f}  "
          f"(backbone variance at FSQ spatial scale {h_prime}×{w_prime})")
    print(f"  Per-channel patch std (mean across channels): "
          f"{bf_patch_mean.std(dim=(0, 2, 3)).mean():.4f}")

    print("\nENCODER OUTPUT STATISTICS (pre-LayerNorm / pre-FSQ)")
    print(f"  Shape          : {tuple(enc_out.shape)}")
    print(f"  Global std     : {enc_out.std():.4f}")
    print(f"  Per-channel std: {enc_out.std(dim=(0, 2, 3)).tolist()}")
    print(f"  Max abs value  : {enc_out.abs().max():.4f}")

    print("\nFSQ CODE DISTRIBUTION (random init VQ, real backbone features)")
    print(f"  Total tokens   : {total_tokens:,}")
    print(f"  Codes used     : {used} / {n_e}  ({used/n_e*100:.1f}%)")
    print(f"  Expected (uniform, {total_tokens} samples): {expected_uniform:.0f}")
    print(f"  Coverage vs uniform: {pct_of_uniform:.1f}%")
    print(f"  Entropy        : {entropy:.3f} / {math.log(n_e):.3f}")

    print("\n" + "=" * 60)
    if used / expected_uniform < 0.15:
        print("DIAGNOSIS: LOW coverage even at random init.")
        print("  → The backbone features at the FSQ spatial scale lack diversity.")
        print("  → No amount of VQ training tricks will fix this.")
        print("  → Fix: (a) train Stage 1 longer / improve backbone quality,")
        print("             OR (b) use a coarser FSQ (fewer ch_mult levels)")
        print("             OR (c) reduce quantizer_channels to 2 or use fewer FSQ levels")
        print("             OR (d) train the VQ jointly with the full diffusion loss")
    elif used / expected_uniform < 0.50:
        print("DIAGNOSIS: MODERATE coverage at random init — backbone is somewhat diverse.")
        print("  → Training dynamics are collapsing codes further.")
        print("  → Fix: two-phase training (freeze decoder first, then joint).")
    else:
        print("DIAGNOSIS: GOOD coverage at random init — backbone is sufficiently diverse.")
        print("  → Pure training dynamics collapse. Fix: stronger entropy or decoder freeze.")


if __name__ == "__main__":
    main()
