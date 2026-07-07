"""
Quick sanity check: does the FSQ model (with LayerNorm) give good code coverage
at *random initialization*, before any training?

Run: python check_fsq_init_coverage.py --config configs/vq/vq_multicond_RWTH_compress.yaml
"""
import argparse
import math
import torch
import numpy as np
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n_batches", type=int, default=50,
                        help="Number of random batches to run")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    vq_kwargs = OmegaConf.to_container(cfg.modules.condition_encoder_kwargs.vq_kwargs)
    n_e = vq_kwargs["n_e"]
    in_ch = vq_kwargs.get("in_channels", 320)
    h_in, w_in = vq_kwargs["input_size"]  # e.g. [34, 28]

    from models.condition_encoder import VQModel
    model = VQModel(
        in_channels=in_ch,
        quantizer_channels=vq_kwargs["quantizer_channels"],
        n_e=n_e,
        vq_type=vq_kwargs["vq_type"],
        fsq_levels=vq_kwargs["fsq_levels"],
        ch_mult=vq_kwargs["ch_mult"],
        input_size=vq_kwargs["input_size"],
    ).eval()

    # Probe output spatial size with one forward pass
    with torch.no_grad():
        probe = torch.randn(1, in_ch, h_in, w_in)
        probe_idx = model.encode(probe)        # (1, h'*w')
        n_spatial = probe_idx.shape[1]
        print(f"  Encoder output: {n_spatial} tokens per sample "
              f"(input {h_in}×{w_in} → output {int(math.isqrt(n_spatial))+1}×{n_spatial // (int(math.isqrt(n_spatial))+1)} approx)")

    B = args.batch_size
    tokens_per_batch = B * n_spatial
    total_tokens = args.n_batches * tokens_per_batch

    print(f"=== FSQ code coverage at random init ===")
    print(f"  Running {args.n_batches} batches × {B} samples × {n_spatial} tokens = {total_tokens:,} tokens")
    print(f"  Codebook size: {n_e}")

    all_indices = []
    with torch.no_grad():
        for _ in range(args.n_batches):
            x = torch.randn(B, in_ch, h_in, w_in) * 1.7
            idx = model.encode(x)   # (B, n_spatial)
            all_indices.append(idx.reshape(-1))

    indices = torch.cat(all_indices).long()
    counts = torch.bincount(indices, minlength=n_e)
    total = counts.sum().item()
    used = (counts > 0).sum().item()

    # Expected unique codes if distribution were uniform
    expected_uniform = n_e * (1 - ((n_e - 1) / n_e) ** total)
    pct_of_expected = used / expected_uniform * 100

    entropy = -(counts.float() / total * torch.log(counts.float() / total + 1e-10)).sum().item()
    max_entropy = math.log(n_e)

    print(f"\n  Codes used    : {used} / {n_e}  ({used/n_e*100:.1f}%)")
    print(f"  Expected (uniform dist, {total} samples): {expected_uniform:.0f} codes")
    print(f"  Coverage vs uniform: {pct_of_expected:.1f}%")
    print(f"  Entropy       : {entropy:.3f} / {max_entropy:.3f} (max)")

    # Collapse diagnosis: compare to a maximally collapsed model (all same code)
    # A healthy model should use > 30% of what a uniform distribution would predict
    if pct_of_expected < 10:
        print("\n  [FAIL] Severe collapse at init — LayerNorm may not be active.")
        print("  Verify pre_quant_norm is in VQModel.__init__ and called in forward/encode.")
    elif pct_of_expected < 30:
        print("\n  [WARN] Moderate concentration — some clustering expected from tanh,")
        print("  but this seems low. Training may still fix it.")
    else:
        print(f"\n  [OK] Good spread at init ({pct_of_expected:.0f}% of uniform baseline).")
        print("  LayerNorm is working. Proceed with training from scratch:")
        print("    accelerate launch --config_file accelerate_config.yaml \\")
        print("      --num_processes 2 --gpu_ids '0,1' \\")
        print("      train_compress_vq_multicond.py \\")
        print("      --config configs/vq/vq_multicond_RWTH_compress.yaml")


if __name__ == "__main__":
    main()
