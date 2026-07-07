import torch
from omegaconf import OmegaConf
from models.condition_encoder import VQConditionEncoder
from signdatasets import SignLangVideoDataset
from torch.utils.data import DataLoader
from einops import rearrange

cfg = OmegaConf.load("configs/vq/vq_multicond_RWTH_compress.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = VQConditionEncoder(
    conditioning_channels=3,
    image_finetune=True,   # no motion modules â frame-by-frame is enough
    num_conds=2,
    use_vq=False,          # backbone only, no quantizer
)

# load Stage 1 backbone weights
model='workspace/vq_multicond_RWTH_compress/20260707-0936/best/condition_encoder/model.bin'
state = torch.load(model, map_location="cpu")
missing, unexpected = enc.load_state_dict(state, strict=False)
print(f"missing: {len(missing)}  unexpected: {len(unexpected)}")
enc = enc.to(device).eval()

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
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

all_feats = []
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i >= 20:
            break
        sk = batch["tgt_sk_frames"].to(device)
        hamer = batch["tgt_hamer_frames"].to(device)
        B, C, F, H, W = sk.shape
        sk = sk.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        hamer = hamer.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
        feats = enc.encode(sk, hamer)   # (BF, 512, H', W')
        all_feats.append(feats.cpu().float())

all_feats = torch.cat(all_feats, dim=0)   # (N, 512, H', W')
print(f"Feature shape : {all_feats.shape}")
print(f"Mean          : {all_feats.mean():.4f}")
print(f"Std (global)  : {all_feats.std():.4f}")
print(f"Std per channel (min/mean/max): "
      f"{all_feats.std(dim=(0,2,3)).min():.4f} / "
      f"{all_feats.std(dim=(0,2,3)).mean():.4f} / "
      f"{all_feats.std(dim=(0,2,3)).max():.4f}")
