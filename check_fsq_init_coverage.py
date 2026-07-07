"""
Quick sanity check: does the FSQ model (with LayerNorm) give good code coverage
at *random initialization*, before any training?

Run: python check_fsq_init_coverage.py --config configs/vq/vq_multicond_RWTH_compress.yaml
"""
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from models.condition_encoder import VQConditionEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_samples", type=int, default=2048,
                        help="Number of random feature vectors to test")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    vq_kwargs = OmegaConf.to_container(cfg.modules.condition_encoder_kwargs.vq_kwargs)
    n_e = vq_kwargs["n_e"]

    # Build a fresh VQModel (random init, no checkpoint)
    from models.condition_encoder import VQModel
    model = VQModel(
        in_channels=vq_kwargs.get("in_channels", 320),
        quantizer_channels=vq_kwargs["quantizer_channels"],
        n_e=n_e,
        vq_type=vq_kwargs["vq_type"],
        fsq_levels=vq_kwargs["fsq_levels"],
        ch_mult=vq_kwargs["ch_mult"],
        input_size=vq_kwargs["input_size"],
    ).eval()

    # Simulate backbone output: Gaussian with std=1.7 (measured from real data)
    # Shape of downsample_encoder input: (B, in_channels, H, W)
    # We don't have real data, so simulate plausible backbone features.
    in_ch = 320  # typical backbone output channels going into vq
    h, w = vq_kwargs["input_size"]
    B = 8

    print("=== FSQ code coverage at random init ===")
    print(f"  Simulated input: ({B}, {in_ch}, {h}, {w}) — N(0, 1.7)")

    all_indices = []
    with torch.no_grad():
        for _ in range(args.n_samples // (B * h * w) + 1):
            x = torch.randn(B, in_ch, h, w) * 1.7
            _, _, loss = model(x)
            # encode to indices
            idx = model.encode(x)  # (B, h'*w')
            all_indices.append(idx.reshape(-1))

    indices = torch.cat(all_indices).long()
    counts = torch.bincount(indices, minlength=n_e)
    used = (counts > 0).sum().item()
    total = counts.sum().item()
    entropy = -(counts.float() / total * torch.log(counts.float() / total + 1e-10)).sum().item()
    max_entropy = np.log(n_e)

    print(f"  Codebook size : {n_e}")
    print(f"  Codes used    : {used} / {n_e}  ({used/n_e*100:.1f}%)")
    print(f"  Entropy       : {entropy:.3f} / {max_entropy:.3f} (max)")
    print(f"  Top-3 codes   : {counts.topk(3).indices.tolist()} "
          f"({counts.topk(3).values.tolist()})")

    if used < 50:
        print("\n  [WARNING] Fewer than 50 codes used at init — LayerNorm may not be applied")
        print("  Check that VQModel.__init__ has self.pre_quant_norm and")
        print("  VQModel.forward/encode call self.pre_quant_norm(x.float()).to(x.dtype)")
    elif used > 200:
        print("\n  [OK] Good coverage at init — LayerNorm is working.")
        print("  Proceed with: accelerate launch ... train_compress_vq_multicond.py ...")
    else:
        print(f"\n  [MODERATE] {used} codes at init — acceptable, should improve during training.")


if __name__ == "__main__":
    main()
