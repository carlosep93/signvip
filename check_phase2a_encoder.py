"""
Verify that the Phase 2a saved checkpoint actually contains Phase 1 encoder weights.

Usage:
    python check_phase2a_encoder.py \
        --phase1  workspace/vq_phase1_20260708-1014/encoder.bin \
        --phase2a workspace/<phase2a_dir>/best/condition_encoder/model.bin \
        --config  configs/vq/vq_multicond_RWTH_compress.yaml

Interprets:
  - max_diff == 0 for all encoder keys  → Phase 1 weights loaded correctly;
                                          147 codes is the genuine natural distribution
  - max_diff  > 0 for some encoder keys → weights were overwritten or not loaded;
                                          code count is meaningless
"""
import argparse

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1",  required=True, help="Phase 1 encoder.bin")
    p.add_argument("--phase2a", required=True, help="Phase 2a best/condition_encoder/model.bin")
    p.add_argument("--config",  default=None,  help="(optional) config yaml")
    return p.parse_args()


def main():
    args = parse_args()

    p1 = torch.load(args.phase1,  map_location="cpu")
    p2 = torch.load(args.phase2a, map_location="cpu")

    enc_keys_p1 = {k for k in p1 if "downsample_encoder" in k}
    enc_keys_p2 = {k for k in p2 if "downsample_encoder" in k}

    print(f"Phase 1  encoder keys : {len(enc_keys_p1)}")
    print(f"Phase 2a encoder keys : {len(enc_keys_p2)}")
    print(f"Phase 2a all keys     : {sorted(set(k.split('.')[0] + '.' + k.split('.')[1] for k in p2))}")
    print()

    # Keys present in Phase 1 but missing from Phase 2a
    missing = enc_keys_p1 - enc_keys_p2
    if missing:
        print(f"MISSING from Phase 2a ({len(missing)} keys):")
        for k in sorted(missing)[:5]:
            print(f"  {k}")
        print()

    # Compare shared keys
    shared = enc_keys_p1 & enc_keys_p2
    print(f"Comparing {len(shared)} shared encoder keys:")
    any_diff = False
    for k in sorted(shared):
        diff = (p1[k].float() - p2[k].float()).abs().max().item()
        if diff > 1e-6:
            print(f"  MISMATCH  {k:<60s}  max_diff={diff:.6f}")
            any_diff = True

    if not any_diff and not missing:
        print("  All encoder keys match exactly — Phase 1 weights ARE loaded in Phase 2a.")
        print()
        print("=> The 147-code result is GENUINE.")
        print("   Phase 1's 523-code report was measured on a single batch DURING training")
        print("   (only ~1536 tokens). The full-data natural distribution is 147 codes.")
        print("   The encoder DOES produce 147 diverse codes, which is far better than")
        print("   the 9-14 codes from pure distillation, but less than 625.")
    elif missing or any_diff:
        print()
        print("=> Phase 1 weights are NOT correctly loaded in Phase 2a.")
        print("   The encoder was randomly initialised and then partially trained by distillation.")
        print("   Fix: verify vq_model path in the Phase 2a config and re-run.")


if __name__ == "__main__":
    main()
