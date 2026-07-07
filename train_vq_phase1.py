"""
Phase 1 VQ training: train only vq.downsample_encoder to produce
high-variance output (diverse FSQ codes), WITHOUT the upsample_decoder
interfering.

Loss: -var(encoder_output) over the batch — pushes features away from
the collapsed center, giving the full range of FSQ codes a chance to
be used before Phase 2 distillation training begins.

Run:
    python train_vq_phase1.py --config configs/vq/vq_multicond_RWTH_compress.yaml

Output: workspace/vq_phase1_<date>/encoder.bin  (only downsample_encoder weights)

Then in Phase 2, set:
    vq_model: workspace/vq_phase1_<date>/encoder.bin
in the VQ config and run the normal training script.
"""
import argparse
import datetime
import math
import os
import pathlib

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.condition_encoder import VQConditionEncoder, VQModel
from signdatasets import SignLangVideoDataset
from utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n_steps", type=int, default=2000,
                   help="Number of variance-maximisation steps")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate for spread phase")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override config batch size")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    seed_everything(cfg.seed)

    vq_cfg = OmegaConf.to_container(cfg.modules.condition_encoder_kwargs.vq_kwargs)
    batch_size = args.batch_size or cfg.dataloader.batch_size

    # ------------------------------------------------------------------ #
    # Backbone (frozen)
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
    print(f"Backbone: missing={len(miss)}, unexpected={len(unexp)}")

    # ------------------------------------------------------------------ #
    # VQ encoder (random init, only downsample_encoder is trained)
    # ------------------------------------------------------------------ #
    in_ch = 256 * 2  # backbone_channels[-1] * num_conds
    vq = VQModel(
        in_channels=in_ch,
        quantizer_channels=vq_cfg["quantizer_channels"],
        n_e=vq_cfg["n_e"],
        vq_type=vq_cfg["vq_type"],
        fsq_levels=vq_cfg["fsq_levels"],
        ch_mult=vq_cfg["ch_mult"],
        input_size=vq_cfg["input_size"],
    ).to(device)

    # Only train the downsample_encoder
    for name, p in vq.named_parameters():
        p.requires_grad_("downsample_encoder" in name)

    trainable = [p for p in vq.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}  "
          f"(downsample_encoder only)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps, eta_min=args.lr * 0.1
    )

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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=cfg.dataloader.num_workers, drop_last=True)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    out_dir = pathlib.Path("workspace") / f"vq_phase1_{datetime.datetime.now():%Y%m%d-%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    pbar = tqdm(total=args.n_steps, desc="Phase1")
    n_e = vq_cfg["n_e"]
    log_every = 100

    while step < args.n_steps:
        for batch in loader:
            if step >= args.n_steps:
                break

            sk = batch["tgt_sk_frames"].to(device)
            hamer = batch["tgt_hamer_frames"].to(device)
            B, C, F, H, W = sk.shape
            sk_flat = sk.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            hm_flat = hamer.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)

            with torch.no_grad():
                feat = backbone.encode(sk_flat, hm_flat)  # (BF, 512, h, w)

            vq.train()
            enc_out = vq.downsample_encoder(feat)  # (BF, 4, h', w')

            # Per-channel variance of FSQ-bounded values across all tokens.
            # Using global var() allowed the encoder to satisfy the objective via
            # constant per-channel biases (e.g. ch0→+20, ch1→-20 regardless of
            # input), which maximises cross-channel variance while all tokens
            # collapse to the same code. var(dim=0) instead measures variance
            # ACROSS different input tokens for each FSQ dimension, so the encoder
            # must produce different codes for different inputs.
            # bound() = tanh(z)*2.002 keeps values in (-2.002, 2.002), which
            # auto-attenuates gradients when outputs saturate (prevents runaway).
            enc_flat = enc_out.permute(0, 2, 3, 1).reshape(-1, 4).float()  # (N, 4)
            z_b = vq.quantizer.bound(enc_flat)        # (N, 4), in (-2.002, 2.002)
            spread_loss = -z_b.var(dim=0).mean()      # per-FSQ-dim variance

            optimizer.zero_grad()
            spread_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)

            if step % log_every == 0:
                with torch.no_grad():
                    vq.eval()
                    idx = vq.encode(feat)
                    codes_used = idx.reshape(-1).unique().numel()

                dim_stds = z_b.detach().std(dim=0).tolist()
                avg_zb_std = sum(dim_stds) / 4
                tqdm.write(
                    f"[step {step:5d}]  loss={spread_loss.item():.4f}"
                    f"  zb_std={avg_zb_std:.3f}"
                    f"  per_dim=[{', '.join(f'{s:.3f}' for s in dim_stds)}]"
                    f"  codes={codes_used}/{n_e}"
                )

    pbar.close()

    # ------------------------------------------------------------------ #
    # Save encoder weights (keyed as vq.downsample_encoder.*)
    # ------------------------------------------------------------------ #
    enc_state = {
        f"vq.{k}": v
        for k, v in vq.state_dict().items()
        if k.startswith("downsample_encoder")
    }
    save_path = out_dir / "encoder.bin"
    torch.save(enc_state, save_path)
    print(f"\nSaved Phase 1 encoder to: {save_path}")
    print(f"Set  vq_model: {save_path}  in the VQ config for Phase 2.")

    # Final code coverage check
    vq.eval()
    counts = torch.zeros(n_e, dtype=torch.long)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 10:
                break
            sk = batch["tgt_sk_frames"].to(device)
            hamer = batch["tgt_hamer_frames"].to(device)
            B, C, F, H, W = sk.shape
            sk_flat = sk.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            hm_flat = hamer.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            feat = backbone.encode(sk_flat, hm_flat)
            idx = vq.encode(feat)
            counts += torch.bincount(idx.reshape(-1).cpu(), minlength=n_e)

    used = (counts > 0).sum().item()
    total = counts.sum().item()
    expected = n_e * (1 - ((n_e - 1) / n_e) ** total)
    print(f"\nFinal coverage: {used}/{n_e} codes  "
          f"({used/expected*100:.1f}% of uniform-expected {expected:.0f})")


if __name__ == "__main__":
    main()
