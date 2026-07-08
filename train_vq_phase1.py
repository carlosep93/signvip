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
    p.add_argument("--lr", type=float, default=5e-4,
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

            # Target: raw (pre-tanh) encoder output should have std ≈ 1.0 per channel.
            #
            # Why std=1.0? The FSQ Voronoi boundaries in raw space are at ≈±0.256
            # and ≈±0.975. For a Gaussian(0, σ=1) input these five cells each
            # receive ≈20% of tokens — the closest to a uniform code distribution.
            #
            # Why raw (not bounded)? Maximising var(z_bounded) is solved by a
            # bimodal ±2 distribution (global max of bounded variance), which
            # saturates the tanh and leaves only 2⁴=16 codes. The gradient also
            # vanishes through tanh at saturation. Operating on enc_flat directly
            # avoids both problems: the gradient is always non-zero and actively
            # opposes saturation when std > 1 (pulling values back toward the
            # inner FSQ levels).
            enc_flat = enc_out.permute(0, 2, 3, 1).reshape(-1, 4).float()  # (N, 4)
            N_tok = enc_flat.shape[0]

            # Term 1 — variance: each FSQ channel should have std ≈ 1.
            # Prevents collapse-to-zero and prevents saturation.
            var_loss = (enc_flat.var(dim=0) - 1.0).pow(2).mean()

            # Term 2 — decorrelation: the four channels must carry independent
            # information. Without this, the encoder satisfies the variance target
            # by routing the same backbone signal to all four channels
            # (e.g. ch0=ch1=ch2=ch3=f), so the code is always (L,L,L,L) — at
            # most 5 codes on the diagonal instead of 625.
            # This is identical to why VICReg/Barlow-Twins add a covariance term.
            enc_c = enc_flat - enc_flat.mean(dim=0, keepdim=True)
            cov = (enc_c.T @ enc_c) / (N_tok - 1)        # (4,4)
            eye4 = torch.eye(4, device=enc_flat.device)
            decorr_loss = (cov * (1 - eye4)).pow(2).mean()

            spread_loss = var_loss + decorr_loss

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

                with torch.no_grad():
                    dim_stds = enc_flat.detach().std(dim=0).tolist()
                    avg_std = sum(dim_stds) / 4
                    enc_c_d = enc_flat.detach() - enc_flat.detach().mean(dim=0)
                    cov_d = (enc_c_d.T @ enc_c_d) / (enc_c_d.shape[0] - 1)
                    max_corr = (cov_d * (1 - eye4)).abs().max().item()
                tqdm.write(
                    f"[step {step:5d}]  loss={spread_loss.item():.4f}"
                    f"  (var={var_loss.item():.4f} decorr={decorr_loss.item():.4f})"
                    f"  raw_std={avg_std:.3f}  max_corr={max_corr:.3f}"
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
