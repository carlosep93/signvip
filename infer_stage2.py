"""
Stage 2 inference: generate a video clip using SignViPPipeline.
Works with both the regular Stage 2 checkpoint and the Stage 2 compress checkpoint (with VQ).

Usage — regular Stage 2:
    python infer_stage2.py \
        --config configs/stage2/stage_2_RWTH.yaml \
        --apperance_encoder  workspace/stage_1_multicond_RWTH/best/appearance_encoder \
        --condition_encoder  workspace/stage_2_RWTH/best/condition_encoder/model.bin \
        --unet               workspace/stage_2_RWTH/best/unet/model.bin \
        --index 0 \
        --output_dir infer_stage2_out

Usage — Stage 2 compress (with VQ):
    python infer_stage2.py \
        --config configs/stage2/stage_2_RWTH_compress.yaml \
        --apperance_encoder  workspace/stage_1_multicond_RWTH/best/appearance_encoder \
        --condition_encoder  workspace/stage_2_RWTH_compress/best/condition_encoder/model.bin \
        --unet               workspace/stage_2_RWTH_compress/best/unet/model.bin \
        --index 0 \
        --output_dir infer_stage2_compress_out
"""

import argparse
import json
import os
import pickle
from collections import OrderedDict

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, PILToTensor

from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import VQConditionEncoder
from models.unet import UNet3DConditionModel
from pipelines.pipeline_multicond import SignViPPipeline
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


def load_sk_frame(sk_data, idx, vid_h, vid_w):
    return Image.fromarray(draw_pose(sk_data[idx], vid_h, vid_w))


def tensor_to_uint8(t):
    """(C, H, W) in [-1,1] or [0,1] → (H, W, C) uint8"""
    if t.min() < 0:
        t = (t + 1) / 2
    return (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


def save_video_mp4(frames_chw, path, fps=8):
    """frames_chw: list of (C,H,W) tensors in [-1,1]. Uses ffmpeg for H.264 output."""
    import subprocess, tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        for i, f in enumerate(frames_chw):
            img = tensor_to_uint8(f)
            Image.fromarray(img).save(os.path.join(tmp, f"{i:05d}.png"))
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmp, "%05d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        shutil.rmtree(tmp)


def save_comparison_grid(gt_frames, gen_frames, sk_frames, hamer_frames, path):
    """Save a side-by-side comparison image (all frames as a horizontal strip)."""
    rows = []
    for gt, gen, sk, hm in zip(gt_frames, gen_frames, sk_frames, hamer_frames):
        gt_img  = tensor_to_uint8(gt)
        gen_img = tensor_to_uint8(gen)
        sk_img  = tensor_to_uint8(sk)
        hm_img  = tensor_to_uint8(hm)
        row = np.concatenate([gt_img, gen_img, sk_img, hm_img], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(path)


def load_mm_state_dict(path):
    """Load a motion module checkpoint, handling module. prefix and state_dict key."""
    sd = torch.load(path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    cleaned = OrderedDict()
    for k, v in sd.items():
        key = k[len("module."):] if k.startswith("module.") else k
        cleaned[key] = v
    return cleaned


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",            required=True, help="Stage 2 config yaml")
    p.add_argument("--apperance_encoder", required=True, help="Appearance encoder dir")
    p.add_argument("--condition_encoder", required=True,
                   help="Condition encoder model.bin (Stage 2 full checkpoint)")
    p.add_argument("--unet",              default=None,
                   help="Fine-tuned UNet model.bin (optional; omit to use SD v1.5 + mm only)")
    p.add_argument("--vq_model",          default=None,
                   help="vq_model.bin for Stage 2 compress (optional)")
    p.add_argument("--index",             type=int, default=0,
                   help="Index of video in the meta JSON")
    p.add_argument("--num_frames",        type=int, default=None,
                   help="Number of frames to generate (default: full video)")
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--guidance_scale",    type=float, default=None)
    p.add_argument("--fps",               type=int, default=8)
    p.add_argument("--output_dir",        default="infer_stage2_out")
    p.add_argument("--device",            default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    device = torch.device(args.device)
    dtype  = torch.bfloat16 if cfg.weight_dtype == "bf16" else torch.float16

    num_inference_steps = args.num_inference_steps or cfg.validation_data.num_inference_steps
    guidance_scale      = args.guidance_scale      or cfg.validation_data.guidance_scale
    frame_size          = tuple(cfg.dataset.frame_size)   # (H, W)
    context_frames      = cfg.dataset.num_frames
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

    num_frames = min(args.num_frames or vid_len, vid_len)

    print(f"Video : {rel}  ({vid_len} frames)")
    print(f"Text  : {entry.get('text', '')}")
    print(f"Generating {num_frames} frames")

    with open(sk_path, "rb") as f:
        sk_data = pickle.load(f)

    # ------------------------------------------------------------------ #
    # load all frames                                                     #
    # ------------------------------------------------------------------ #
    frame_ids = np.linspace(0, vid_len - 1, num_frames, dtype=int).tolist()

    state = torch.get_rng_state()

    ref_pil = read_video_frame(vid_path, 0)
    ref_tensor = transform(ref_pil).unsqueeze(0).to(dtype=dtype, device=device)  # (1,C,H,W)

    tgt_tensors   = []
    sk_tensors    = []
    hamer_tensors = []

    for idx in frame_ids:
        torch.set_rng_state(state)
        tgt_tensors.append(transform(read_video_frame(vid_path, idx)))
        torch.set_rng_state(state)
        sk_tensors.append(transform(load_sk_frame(sk_data, idx, vid_h, vid_w)))
        torch.set_rng_state(state)
        hamer_tensors.append(transform(read_video_frame(hamer_path, idx)))

    # stack → (1, C, F, H, W)
    tgt_bcfhw   = torch.stack(tgt_tensors,   dim=0).unsqueeze(0).permute(0,2,1,3,4)
    sk_bcfhw    = torch.stack(sk_tensors,    dim=0).unsqueeze(0).permute(0,2,1,3,4)
    hamer_bcfhw = torch.stack(hamer_tensors, dim=0).unsqueeze(0).permute(0,2,1,3,4)

    sk_bcfhw    = sk_bcfhw.to(dtype=dtype,    device=device)
    hamer_bcfhw = hamer_bcfhw.to(dtype=dtype, device=device)

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

    # load AnimateDiff mm then fine-tuned unet
    if cfg.modules.get("mm"):
        mm_sd = load_mm_state_dict(cfg.modules.mm)
        unet.load_state_dict(mm_sd, strict=False)
        print(f"  Loaded motion modules from {cfg.modules.mm}")
    if args.unet:
        unet_sd = torch.load(args.unet, map_location="cpu")
        unet.load_state_dict(unet_sd, strict=False)
        print(f"  Loaded fine-tuned UNet from {args.unet}")

    appearance_encoder = AppearanceEncoderModel.from_pretrained(
        args.apperance_encoder
    ).to(dtype=dtype, device=device)

    ce_kwargs = OmegaConf.to_container(
        cfg.modules.condition_encoder_kwargs, resolve=True
    ) if cfg.modules.get("condition_encoder_kwargs") else {}

    condition_encoder = VQConditionEncoder(
        conditioning_channels=3,
        image_finetune=False,
        num_conds=2,
        motion_module_type=cfg.modules.unet_additional_kwargs.motion_module_type,
        motion_module_kwargs=OmegaConf.to_container(
            cfg.modules.unet_additional_kwargs.motion_module_kwargs, resolve=True
        ),
        **ce_kwargs,
    )
    ce_sd = torch.load(args.condition_encoder, map_location="cpu")
    condition_encoder.load_state_dict(ce_sd, strict=False)
    if args.vq_model:
        vq_sd = torch.load(args.vq_model, map_location="cpu")
        condition_encoder.load_state_dict(vq_sd, strict=False)
        print(f"  Loaded VQ weights from {args.vq_model}")
    condition_encoder = condition_encoder.to(dtype=dtype, device=device)

    scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
    empty_text_emb = torch.load(cfg.modules.empty_text_emb, map_location="cpu").to(
        dtype=dtype, device=device
    )

    pipeline = SignViPPipeline(
        vae=vae,
        denoising_unet=unet,
        scheduler=scheduler,
        appearance_encoder=appearance_encoder,
        condition_encoder=condition_encoder,
        empty_text_emb=empty_text_emb,
    ).to(device=device)

    # ------------------------------------------------------------------ #
    # inference                                                           #
    # ------------------------------------------------------------------ #
    print("Running inference …")
    with torch.no_grad():
        video = pipeline(
            ref_image=ref_tensor,
            sk_images=sk_bcfhw,
            hamer_images=hamer_bcfhw,
            width=frame_size[1],
            height=frame_size[0],
            video_length=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            context_frames=context_frames,
        )
    # video: (1, C, T, H, W) in [0,1]
    video = video[0]  # (C, T, H, W)

    # ------------------------------------------------------------------ #
    # save outputs                                                        #
    # ------------------------------------------------------------------ #
    stem = os.path.splitext(base)[0]

    # generated video
    gen_frames = [video[:, t] * 2 - 1 for t in range(video.shape[1])]   # back to [-1,1] for helper
    save_video_mp4(gen_frames, os.path.join(args.output_dir, f"{stem}_generated.mp4"), fps=args.fps)

    # ground-truth video
    save_video_mp4(tgt_tensors, os.path.join(args.output_dir, f"{stem}_gt.mp4"), fps=args.fps)

    # side-by-side comparison grid (gt | generated | skeleton | hamer)
    save_comparison_grid(
        tgt_tensors, gen_frames, sk_tensors, hamer_tensors,
        os.path.join(args.output_dir, f"{stem}_comparison.png"),
    )

    print(f"\nSaved to {args.output_dir}/")
    print(f"  {stem}_gt.mp4          — ground-truth video")
    print(f"  {stem}_generated.mp4   — generated video")
    print(f"  {stem}_comparison.png  — frame-by-frame grid: gt | generated | skeleton | hamer")


if __name__ == "__main__":
    main()
