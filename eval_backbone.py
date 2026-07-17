"""
Evaluate SignViP pipeline performance across all training stages.

Runs the full video generation pipeline on a dev/test split and reports
per-video SSIM vs ground-truth.  Supports all four configurations:

  pretrained  — original SD v1.5 + AnimateDiff weights, no fine-tuning
  stage1      — trained AppearanceEncoder + ConditionEncoder (static, no VQ)
  stage2      — stage1 + trained motion modules (temporal)
  vq          — stage2 + trained VQ autoencoder (quantised conditioning)

Stage is detected automatically from the config, but can be forced with --stage.
For each stage the right pipeline (static vs motion) and conditioning path
(raw pose maps vs on-the-fly VQ encoding) is selected automatically.

Usage:
    # evaluate stage2 checkpoint on dev split
    python eval_backbone.py \\
        --config  configs/stage2/stage_2_RWTH.yaml \\
        --stage   stage2 \\
        --meta_path  /path/to/dev_processed_videos.json \\
        --root       /path/to/PHOENIX-2014-T \\
        --sk_root    /path/to/dev_processed_videos/sk \\
        --hamer_root /path/to/dev_processed_videos/hamer_rendered \\
        --gt_dir     /path/to/dev_processed_videos \\
        --output_dir outputs/stage2_dev

    # evaluate VQ checkpoint
    python eval_backbone.py \\
        --config  configs/vq/vq_multicond_RWTH_compress.yaml \\
        --stage   vq \\
        ...

    # compare all stages (run once per config, then diff the SSIM outputs)
"""
import argparse
import json
import os
import pickle
import sys
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import (
    Compose, Normalize, PILToTensor, RandomResizedCrop,
)
from diffusers import AutoencoderKL, DDIMScheduler

sys.path.insert(0, os.path.dirname(__file__))

from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import VQConditionEncoder
from models.unet import UNet3DConditionModel
from pipelines.pipeline_multicond import SignViPPipeline
from pipelines.pipeline_static import SignViPStaticPipeline
from utils import save_video

warnings.filterwarnings("ignore")


# ── stage detection ──────────────────────────────────────────────────────────

def detect_stage(cfg, forced=None):
    """Return one of: 'pretrained', 'stage1', 'stage2', 'vq'."""
    if forced:
        return forced
    ce_kwargs = cfg.modules.condition_encoder_kwargs
    use_vq = ce_kwargs.get("use_vq", False)
    skip_vq = ce_kwargs.get("vq_kwargs", {}).get("skip_vq", False)
    has_mm = cfg.modules.get("condition_encoder_motion") or cfg.modules.get("mm")
    has_vq_weights = bool(cfg.modules.get("vq_model"))
    if use_vq and not skip_vq and has_vq_weights:
        return "vq"
    if has_mm and cfg.modules.get("condition_encoder_motion"):
        return "stage2"
    if cfg.modules.get("condition_encoder"):
        return "stage1"
    return "pretrained"


# ── model loading ────────────────────────────────────────────────────────────

def load_condition_encoder(cfg, stage, device, dtype):
    image_finetune = (stage == "stage1" or stage == "pretrained")
    use_vq = (stage == "vq") and cfg.modules.condition_encoder_kwargs.get("use_vq", False)
    skip_vq_in_cfg = cfg.modules.condition_encoder_kwargs.get("vq_kwargs", OmegaConf.create()).get("skip_vq", False)

    ce_kwargs = OmegaConf.to_container(cfg.modules.condition_encoder_kwargs)
    # force skip_vq for non-VQ stages even if the config has VQ fields
    if "vq_kwargs" in ce_kwargs and stage != "vq":
        ce_kwargs["vq_kwargs"]["skip_vq"] = True

    encoder = VQConditionEncoder(
        conditioning_channels=3,
        image_finetune=image_finetune,
        num_conds=2,
        motion_module_type=cfg.modules.unet_additional_kwargs.motion_module_type,
        motion_module_kwargs=OmegaConf.to_container(
            cfg.modules.unet_additional_kwargs.motion_module_kwargs
        ),
        **ce_kwargs,
    )

    merged = {}

    if cfg.modules.get("condition_encoder"):
        sd = torch.load(cfg.modules.condition_encoder, map_location="cpu")
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        merged.update(sd)

    if not image_finetune and cfg.modules.get("condition_encoder_motion"):
        sd = torch.load(cfg.modules.condition_encoder_motion, map_location="cpu")
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        merged.update(sd)

    if stage == "vq" and cfg.modules.get("vq_model"):
        sd = torch.load(cfg.modules.vq_model, map_location="cpu")
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        merged.update(sd)

    if merged:
        missing, unexpected = encoder.load_state_dict(merged, strict=False)
        print(f"  ConditionEncoder  missing={len(missing)}  unexpected={len(unexpected)}")

    return encoder.to(device, dtype).eval()


def load_models(cfg, stage, device, dtype):
    mods = cfg.modules

    vae = AutoencoderKL.from_pretrained(mods.vae).to(device, dtype)

    unet = UNet3DConditionModel.from_pretrained_2d(
        mods.unet_2d,
        unet_additional_kwargs=OmegaConf.to_container(mods.unet_additional_kwargs),
    )

    # load UNet motion module weights
    if stage in ("stage2", "vq") and mods.get("mm"):
        mm_sd = torch.load(mods.mm, map_location="cpu")
        mm_sd = mm_sd.get("state_dict", mm_sd)
        mm_sd = {k[7:] if k.startswith("module.") else k: v for k, v in mm_sd.items()}
        unet.load_state_dict(mm_sd, strict=False)
        print(f"  Loaded UNet motion module from {mods.mm}")

    if mods.get("unet") and os.path.exists(str(mods.unet)):
        unet.load_state_dict(torch.load(mods.unet, map_location="cpu"), strict=False)
        print(f"  Loaded full UNet from {mods.unet}")

    unet.to(device, dtype)

    appearance_encoder = AppearanceEncoderModel.from_pretrained(
        mods.apperance_encoder
    ).to(device, dtype)

    condition_encoder = load_condition_encoder(cfg, stage, device, dtype)

    empty_text_emb = torch.load(mods.empty_text_emb).to(device, dtype)
    scheduler = DDIMScheduler.from_pretrained(mods.scheduler)

    return vae, unet, appearance_encoder, condition_encoder, empty_text_emb, scheduler


# ── data helpers ─────────────────────────────────────────────────────────────

class _ZeroOne:
    def __call__(self, img):
        return img.float().div(255)


def make_transform(frame_size, frame_scale=(1.0, 1.0)):
    return Compose([
        PILToTensor(),
        _ZeroOne(),
        RandomResizedCrop(
            frame_size,
            scale=frame_scale,
            ratio=(frame_size[1] / frame_size[0], frame_size[1] / frame_size[0]),
            antialias=True,
        ),
        Normalize(mean=[0.5], std=[0.5]),
    ])


def load_video_data(vid_path, sk_root, hamer_root, sample_rate, transform, frame_size):
    """Returns (ref_frame_pil, sk_frames_tensor, hamer_frames_tensor, original_fps) or None."""
    sk_pkl = os.path.join(sk_root, os.path.basename(vid_path).replace(".mp4", ".pkl"))
    hamer_mp4 = os.path.join(hamer_root, os.path.basename(vid_path))

    if not os.path.exists(sk_pkl):
        return None
    if not os.path.exists(hamer_mp4):
        return None

    try:
        with open(sk_pkl, "rb") as f:
            sk_data = pickle.load(f)
    except Exception as e:
        print(f"  skip {sk_pkl}: {e}")
        return None

    orig_cap = cv2.VideoCapture(vid_path)
    original_fps = orig_cap.get(cv2.CAP_PROP_FPS) or 25.0

    # reference frame = first frame of original video
    ret, frame0 = orig_cap.read()
    if not ret:
        orig_cap.release()
        return None
    ref_pil = Image.fromarray(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)).resize(
        (frame_size[1], frame_size[0])
    )
    orig_cap.release()

    hamer_cap = cv2.VideoCapture(hamer_mp4)
    n_frames = len(sk_data)
    frame_ids = list(range(0, n_frames, sample_rate))

    from signdatasets.sign_cond import draw_pose

    sk_frames, hamer_frames = [], []
    rng_state = torch.get_rng_state()

    for fid in frame_ids:
        pose = sk_data[fid]
        img_sk = draw_pose(pose, frame_size[0], frame_size[1])
        img_sk = Image.fromarray(img_sk)

        hamer_cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = hamer_cap.read()
        if not ret:
            break
        img_hamer = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        torch.set_rng_state(rng_state)
        sk_t = transform(img_sk)
        torch.set_rng_state(rng_state)
        hamer_t = transform(img_hamer)

        sk_frames.append(sk_t)
        hamer_frames.append(hamer_t)

    hamer_cap.release()

    if not sk_frames:
        return None

    return ref_pil, torch.stack(sk_frames), torch.stack(hamer_frames), original_fps


# ── SSIM ─────────────────────────────────────────────────────────────────────

def _ssim_channel(a, b):
    C1, C2 = 0.01**2, 0.03**2
    a, b = a.astype(np.float64), b.astype(np.float64)
    k = cv2.getGaussianKernel(11, 1.5)
    w = np.outer(k, k.T)
    mu1 = cv2.filter2D(a, -1, w)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, w)[5:-5, 5:-5]
    s1  = cv2.filter2D(a**2,   -1, w)[5:-5, 5:-5] - mu1**2
    s2  = cv2.filter2D(b**2,   -1, w)[5:-5, 5:-5] - mu2**2
    s12 = cv2.filter2D(a*b,    -1, w)[5:-5, 5:-5] - mu1*mu2
    return (((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean()


def frame_ssim(f1, f2):
    f1 = cv2.resize(f1, (f2.shape[1], f2.shape[0])).astype(np.float64) / 255.
    f2 = f2.astype(np.float64) / 255.
    return np.mean([_ssim_channel(f1[:,:,c], f2[:,:,c]) for c in range(3)])


def video_ssim(pred_path, gt_path, max_frames=None):
    cap_p = cv2.VideoCapture(pred_path)
    cap_g = cv2.VideoCapture(gt_path)
    scores = []
    n = 0
    while True:
        ret_p, fp = cap_p.read()
        ret_g, fg = cap_g.read()
        if not ret_p or not ret_g:
            break
        fp = cv2.cvtColor(fp, cv2.COLOR_BGR2RGB)
        fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
        scores.append(frame_ssim(fp, fg))
        n += 1
        if max_frames and n >= max_frames:
            break
    cap_p.release()
    cap_g.release()
    return float(np.mean(scores)) if scores else None


# ── inference ─────────────────────────────────────────────────────────────────

def infer_static(pipeline, cfg, args, condition_encoder, ref_pil, sk_frames, hamer_frames):
    """Stage 1: generate one frame at a time, return list of PIL images."""
    out_frames = []
    for i in range(sk_frames.shape[0]):
        sk_i  = sk_frames[i:i+1].unsqueeze(0)   # (1, C, 1, H, W) — not needed for static
        hm_i  = hamer_frames[i:i+1].unsqueeze(0)
        video = pipeline(
            reference_image=ref_pil,
            sk_image=sk_frames[i],
            hamer_image=hamer_frames[i],
            width=cfg.dataset.frame_size[1],
            height=cfg.dataset.frame_size[0],
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        # video: (1, C, 1, H, W) tensor in [0,1]
        frame = (video[0, :, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out_frames.append(frame)
    return out_frames


def infer_motion(pipeline, cfg, args, condition_encoder, ref_pil, sk_frames, hamer_frames):
    """Stage 2 / VQ: generate full video with motion pipeline."""
    # sk_frames: (F, C, H, W)  →  pipeline wants (1, C, F, H, W)
    sk_t  = sk_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)   # wait, already (F,C,H,W)
    # Actually pipeline takes list of PIL or tensors per frame
    # Pass as tensor (B, C, F, H, W)
    sk_t  = sk_frames.permute(1, 0, 2, 3).unsqueeze(0)   # (1, C, F, H, W)
    hm_t  = hamer_frames.permute(1, 0, 2, 3).unsqueeze(0)

    video = pipeline(
        condition_encoder=condition_encoder,
        sk_images=sk_t,
        hamer_images=hm_t,
        pose_latent=None,
        ref_image=ref_pil,
        width=cfg.dataset.frame_size[1],
        height=cfg.dataset.frame_size[0],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        context_batch_size=1,
        context_frames=24,
    )
    return video  # tensor (1, C, F, H, W) or similar — handled by save_video


def infer_vq(pipeline, cfg, args, condition_encoder, ref_pil, sk_frames, hamer_frames, device, dtype):
    """VQ stage: encode pose maps to tokens on-the-fly, then run motion pipeline."""
    sk_t = sk_frames.to(device, dtype)
    hm_t = hamer_frames.to(device, dtype)
    with torch.no_grad():
        indices = condition_encoder.encode(sk_t, hm_t, return_indices=True)  # (F, h*w)
        pose_latent = indices  # (F, pose_size)

    video = pipeline(
        condition_encoder=condition_encoder,
        sk_images=None,
        hamer_images=None,
        pose_latent=pose_latent,
        ref_image=ref_pil,
        width=cfg.dataset.frame_size[1],
        height=cfg.dataset.frame_size[0],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        context_batch_size=1,
        context_frames=24,
    )
    return video


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True)
    p.add_argument("--meta_path", required=True, help="JSON metadata for dev/test split")
    p.add_argument("--root",      required=True, help="Root dir of original videos")
    p.add_argument("--sk_root",   required=True, help="Root dir of skeleton .pkl files")
    p.add_argument("--hamer_root",required=True, help="Root dir of HAMER .mp4 files")
    p.add_argument("--output_dir",required=True, help="Where to write generated .mp4 files")
    p.add_argument("--gt_dir",    default=None,  help="Ground truth video dir for SSIM (optional)")
    p.add_argument("--stage",     default=None,
                   choices=["pretrained", "stage1", "stage2", "vq"],
                   help="Force a stage; auto-detected from config if omitted")
    p.add_argument("--sample_rate",         type=int,   default=1)
    p.add_argument("--num_inference_steps", type=int,   default=20)
    p.add_argument("--guidance_scale",      type=float, default=3.5)
    p.add_argument("--max_videos",          type=int,   default=None)
    p.add_argument("--device",              default="cuda")
    p.add_argument("--skip_existing",       action="store_true", default=True)
    args = p.parse_args()

    cfg   = OmegaConf.load(args.config)
    stage = detect_stage(cfg, args.stage)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map.get(cfg.weight_dtype, torch.float16)

    print(f"\nStage detected : {stage}")
    print(f"dtype          : {cfg.weight_dtype}")
    print(f"device         : {device}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    vae, unet, appearance_encoder, condition_encoder, empty_text_emb, scheduler = \
        load_models(cfg, stage, device, dtype)

    frame_size  = tuple(cfg.dataset.frame_size)
    frame_scale = tuple(cfg.dataset.frame_scale)
    transform   = make_transform(frame_size, frame_scale)

    if stage == "stage1":
        pipeline = SignViPStaticPipeline(
            vae=vae,
            denoising_unet=unet,
            scheduler=scheduler,
            appearance_encoder=appearance_encoder,
            condition_encoder=condition_encoder,
        ).to(dtype=dtype, device=device)
    else:
        pipeline = SignViPPipeline(
            vae=vae,
            denoising_unet=unet,
            scheduler=scheduler,
            empty_text_emb=empty_text_emb,
            appearance_encoder=appearance_encoder,
        ).to(dtype=dtype, device=device)

    with open(args.meta_path) as f:
        meta = json.load(f)
    if args.max_videos:
        meta = meta[:args.max_videos]

    ssim_scores = []
    skipped = 0

    for entry in tqdm(meta, desc=f"[{stage}] generating"):
        vid_name    = os.path.basename(entry["video"])
        vid_path    = os.path.join(args.root, entry["video"])
        output_path = os.path.join(args.output_dir, vid_name)

        if args.skip_existing and os.path.exists(output_path):
            # still compute SSIM if GT available
            if args.gt_dir:
                gt_path = os.path.join(args.gt_dir, vid_name)
                if os.path.exists(gt_path):
                    s = video_ssim(output_path, gt_path)
                    if s is not None:
                        ssim_scores.append(s)
            continue

        data = load_video_data(
            vid_path, args.sk_root, args.hamer_root,
            args.sample_rate, transform, frame_size
        )
        if data is None:
            skipped += 1
            continue

        ref_pil, sk_frames, hamer_frames, original_fps = data

        with torch.no_grad():
            if stage == "stage1":
                frames = infer_static(
                    pipeline, cfg, args, condition_encoder,
                    ref_pil, sk_frames, hamer_frames
                )
                # write with OpenCV
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    original_fps,
                    (w, h),
                )
                for fr in frames:
                    writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                writer.release()

            elif stage == "stage2":
                video = infer_motion(
                    pipeline, cfg, args, condition_encoder,
                    ref_pil, sk_frames, hamer_frames
                )
                save_video(video, output_path, device=device, fps=original_fps)

            else:  # vq or pretrained
                video = infer_vq(
                    pipeline, cfg, args, condition_encoder,
                    ref_pil, sk_frames, hamer_frames, device, dtype
                )
                save_video(video, output_path, device=device, fps=original_fps)

        if args.gt_dir:
            gt_path = os.path.join(args.gt_dir, vid_name)
            if os.path.exists(gt_path):
                s = video_ssim(output_path, gt_path)
                if s is not None:
                    ssim_scores.append(s)

    print(f"\n{'='*50}")
    print(f"  Stage          : {stage}")
    print(f"  Videos done    : {len(meta) - skipped}  (skipped {skipped})")
    print(f"  Output dir     : {args.output_dir}")
    if ssim_scores:
        print(f"{'='*50}")
        print(f"  SSIM vs GT     : {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
        print(f"  SSIM min/max   : {np.min(ssim_scores):.4f} / {np.max(ssim_scores):.4f}")
    else:
        print("  (no GT dir supplied — SSIM not computed)")
    print(f"{'='*50}\n")

    if ssim_scores:
        out_csv = os.path.join(args.output_dir, "ssim_scores.txt")
        with open(out_csv, "w") as f:
            for s in ssim_scores:
                f.write(f"{s:.6f}\n")
        print(f"  Per-video SSIM written to {out_csv}")


if __name__ == "__main__":
    main()
