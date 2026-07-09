"""
Download the umnooob/signvip checkpoint from HuggingFace and print all
condition_encoder keys, highlighting any that don't exist in the current
VQConditionEncoder model.

Run on the cluster:
    python inspect_hf_checkpoint.py
"""
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from omegaconf import OmegaConf

from models.condition_encoder import VQConditionEncoder


def main():
    repo_id = "umnooob/signvip"

    # ------------------------------------------------------------------ #
    # 1. List all files in the HF repo
    # ------------------------------------------------------------------ #
    print(f"Files in {repo_id}:")
    files = list(list_repo_files(repo_id))
    for f in sorted(files):
        print(f"  {f}")

    # ------------------------------------------------------------------ #
    # 2. Download condition_encoder checkpoint(s)
    # ------------------------------------------------------------------ #
    ce_files = [f for f in files if "condition_encoder" in f and f.endswith(".bin")]
    if not ce_files:
        print("\nNo condition_encoder .bin files found — checking for safetensors...")
        ce_files = [f for f in files if "condition_encoder" in f]

    print(f"\nCondition encoder files: {ce_files}")

    for fname in ce_files:
        print(f"\n{'='*60}")
        print(f"Inspecting: {fname}")
        local = hf_hub_download(repo_id=repo_id, filename=fname)
        state = torch.load(local, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        keys = sorted(state.keys())
        print(f"  Total keys: {len(keys)}")

        # Group by top-level module
        from collections import Counter
        prefixes = Counter(k.split(".")[0] for k in keys)
        print("\n  Keys by top-level module:")
        for prefix, count in sorted(prefixes.items()):
            print(f"    {prefix}: {count} params")

        print("\n  All keys:")
        for k in keys:
            print(f"    {k}")

    # ------------------------------------------------------------------ #
    # 3. Build current model and compare keys
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("Building current VQConditionEncoder (with VQ, no motion)...")
    cfg = OmegaConf.load("configs/vq/vq_multicond_RWTH_compress.yaml")
    model = VQConditionEncoder(
        conditioning_channels=3,
        image_finetune=True,
        num_conds=2,
        **cfg.modules.condition_encoder_kwargs,
    )
    current_keys = set(model.state_dict().keys())

    # Collect all keys from HF checkpoints
    hf_keys = set()
    for fname in ce_files:
        local = hf_hub_download(repo_id=repo_id, filename=fname)
        state = torch.load(local, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        hf_keys.update(state.keys())

    # Keys in HF but not in current model — these are the missing layers
    missing_in_current = hf_keys - current_keys
    extra_in_current = current_keys - hf_keys

    print(f"\nKeys in HF checkpoint but NOT in current model ({len(missing_in_current)}):")
    for k in sorted(missing_in_current):
        print(f"  [MISSING] {k}")

    print(f"\nKeys in current model but NOT in HF checkpoint ({len(extra_in_current)}):")
    for k in sorted(extra_in_current)[:30]:
        print(f"  [EXTRA]   {k}")
    if len(extra_in_current) > 30:
        print(f"  ... and {len(extra_in_current)-30} more")


if __name__ == "__main__":
    main()
