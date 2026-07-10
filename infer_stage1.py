"""
Stage 1 inference: generate a single conditioned frame using SignViPStaticPipeline.

Usage:
    python infer_stage1.py \
        --config  configs/stage1/stage_1_multicond_RWTH.yaml \
        --apperance_encoder  workspace/stage_1_multicond_RWTH/best/appearance_encoder \
        --condition_encoder  workspace/stage_1_multicond_RWTH/best/condition_encoder/model.bin \
        --index 0 \
        --output_dir infer_stage1_out
"""

import argparse
import json
import os
import pickle

import cv2
import numpy as np
import torch
from diffusers import DDIMScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, PILToTensor

from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import ConditionEncoder
from models.unet import UNet3DConditionModel
from diffusers import AutoencoderKL
from pipelines.pipeline_static import SignViPStaticPipeline
from scripts.sk.dwpose.util import draw_pose


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

class ZeroOneNormalize:
    def __call__(self, img):
        return img.float().div(255)


def build_transform(frame_size):
    h, w = frame_size
    return Compose([
        PILToTensor(),
        ZeroOneNormalize(),
        RandomResizedCrop(
            frame_size,
            scale=(1.0, 1.0),
            ratio=(w / h, w / h),
            antialias=True,
        ),
        Normalize(mean=[0.5], std=[0.5]),
    ])


def read_video_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx} from {path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def load_sk_frame(sk_path, idx, vid_h, vid_w):
    with open(sk_path, "rb") as f:
        sk_data = pickle.load(f)
    return draw_pose(sk_data[idx], vid_h, vid_w)


def save_image(tensor, path):
    """tensor: (C, H, W) in [-1, 1]"""
    img = ((tensor.cpu().float() + 1) / 2 * 255).clamp(0, 255).byte()
    img = img.permute(1, 2, 0).numpy()
    Image.fromarray(img).save(path)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",              required=True,  help="Stage 1 config yaml")
    p.add_argument("--apperance_encoder",   required=True,  help="Appearance encoder dir")
    p.add_argument("--condition_encoder",   required=True,  help="Condition encoder model.bin")
    p.add_argument("--unet",                default=None,   help="Fine-tuned UNet model.bin (optional)")
    p.add_argument("--index",               type=int, default=0,
                   help="Index of video in the meta_paths JSON to run inference on")
    p.add_argument("--frame",               type=int, default=None,
                   help="Target frame index inside the video (default: mid-point)")
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--guidance_scale",      type=float, default=None)
    p.add_argument("--output_dir",          default="infer_stage1_out")
    p.add_argument("--device",              default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    device = torch.device(args.device)
    dtype  = torch.float16 if cfg.weight_dtype == "fp16" else torch.bfloat16

    num_inference_steps = args.num_inference_steps or cfg.validation_data.num_inference_steps
    guidance_scale      = args.guidance_scale      or cfg.validation_data.guidance_scale
    frame_size          = tuple(cfg.dataset.frame_size)   # (H, W)
    transform           = build_transform(frame_size)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # load metadata and pick video                                        #
    # ------------------------------------------------------------------ #
    meta_path  = cfg.dataset.meta_paths[0]
    root       = cfg.dataset.roots[0]
    sk_root    = cfg.dataset.sk_roots[0]
    hamer_root = cfg.dataset.hamer_roots[0]

    entries = json.load(open(meta_path))
    entry   = entries[args.index]
    rel     = entry["video"]
    base    = os.path.basename(rel)

    vid_path   = os.path.join(root,       rel)
    sk_path    = os.path.join(sk_root,    base.replace(".mp4", ".pkl"))
    hamer_path = os.path.join(hamer_root, base)

    cap = cv2.VideoCapture(vid_path)
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    ref_idx = 0
    tgt_idx = args.frame if args.frame is not None else vid_len // 2

    print(f"Video : {rel}  ({vid_len} frames)")
    print(f"Text  : {entry.get('text', '')}")
    print(f"Ref frame: {ref_idx}   Target frame: {tgt_idx}")

    # ------------------------------------------------------------------ #
    # load inputs                                                         #
    # ------------------------------------------------------------------ #
    ref_pil   = read_video_frame(vid_path,   ref_idx)
    tgt_pil   = read_video_frame(vid_path,   tgt_idx)
    sk_pil    = Image.fromarray(load_sk_frame(sk_path, tgt_idx, vid_h, vid_w))
    hamer_pil = read_video_frame(hamer_path, tgt_idx)

    state = torch.get_rng_state()
    ref_tensor   = transform(ref_pil).unsqueeze(0)             # (1, C, H, W)
    torch.set_rng_state(state)
    sk_tensor    = transform(sk_pil).unsqueeze(0).unsqueeze(2) # (1, C, 1, H, W)
    torch.set_rng_state(state)
    hamer_tensor = transform(hamer_pil).unsqueeze(0).unsqueeze(2)

    # ------------------------------------------------------------------ #
    # build models                                                        #
    # ------------------------------------------------------------------ #
    print("Loading models …")
    vae = AutoencoderKL.from_pretrained(cfg.modules.vae).to(dtype=dtype, device=device)

    unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.modules.unet_2d,
        unet_additional_kwargs=OmegaConf.to_container(
            cfg.modules.unet_additional_kwargs, resolve=True
        ),
    ).to(dtype=dtype, device=device)

    if args.unet:
        state = torch.load(args.unet, map_location="cpu")
        unet.load_state_dict(state, strict=False)
        print(f"  Loaded fine-tuned UNet from {args.unet}")

    appearance_encoder = AppearanceEncoderModel.from_pretrained(
        args.apperance_encoder
    ).to(dtype=dtype, device=device)

    condition_encoder = ConditionEncoder(
        conditioning_channels=3,
        image_finetune=True,
        num_conds=2,
    ).to(dtype=dtype, device=device)
    state = torch.load(args.condition_encoder, map_location="cpu")
    condition_encoder.load_state_dict(state, strict=False)

    scheduler = DDIMScheduler.from_pretrained(cfg.modules.scheduler)
    empty_text_emb = torch.load(cfg.modules.empty_text_emb, map_location="cpu")

    pipeline = SignViPStaticPipeline(
        vae=vae,
        denoising_unet=unet,
        scheduler=scheduler,
        empty_text_emb=empty_text_emb,
        appearance_encoder=appearance_encoder,
        condition_encoder=condition_encoder,
    ).to(dtype=dtype, device=device)

    # ------------------------------------------------------------------ #
    # inference                                                           #
    # ------------------------------------------------------------------ #
    print("Running inference …")
    with torch.no_grad():
        generated = pipeline(
            prompt=cfg.validation_data.prompt,
            reference_image=ref_tensor.to(dtype=dtype, device=device),
            sk_image=sk_tensor.to(dtype=dtype, device=device),
            hamer_image=hamer_tensor.to(dtype=dtype, device=device),
            width=frame_size[1],
            height=frame_size[0],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            batch_size=1,
        )

    # generated: (B, C, F, H, W) or (B, C, H, W)
    if generated.ndim == 5:
        generated = generated[:, :, 0]   # take first (only) frame

    stem = os.path.splitext(base)[0]
    save_image(generated[0], os.path.join(args.output_dir, f"{stem}_ref.png"))
    save_image(ref_tensor[0], os.path.join(args.output_dir, f"{stem}_ref_input.png"))
    save_image(transform(tgt_pil), os.path.join(args.output_dir, f"{stem}_gt.png"))
    sk_tensor_vis  = sk_tensor[0, :, 0]     # (C,H,W) in [-1,1]
    hamer_tensor_vis = hamer_tensor[0, :, 0]
    save_image(sk_tensor_vis,    os.path.join(args.output_dir, f"{stem}_sk.png"))
    save_image(hamer_tensor_vis, os.path.join(args.output_dir, f"{stem}_hamer.png"))

    print(f"Saved to {args.output_dir}/")
    print(f"  {stem}_ref_input.png  — reference frame fed to appearance encoder")
    print(f"  {stem}_sk.png         — DWPose skeleton condition")
    print(f"  {stem}_hamer.png      — HAMER hand condition")
    print(f"  {stem}_gt.png         — ground-truth target frame")
    print(f"  {stem}_ref.png        — generated frame")


if __name__ == "__main__":
    main()
