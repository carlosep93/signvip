"""
Compute SSIM and FVD between original and VQ-reconstructed videos.

Usage:
    python eval_vq_metrics.py \
        --gt_dir   /path/to/dev_processed_videos \
        --pred_dir dev_vq_outputs \
        [--max_videos 100] \
        [--max_frames 64] \
        [--no_fvd]

Both directories are searched for matching .mp4 files by filename.
Only files present in both directories are evaluated.

FVD needs the I3D torchscript model at:
    metrics/fvd/styleganv/i3d_torchscript.pt
It will be downloaded automatically if missing (requires wget + internet).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))


# ── SSIM ─────────────────────────────────────────────────────────────────────

def _ssim_single_channel(img1, img2):
    C1, C2 = 0.01**2, 0.03**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    s1 = cv2.filter2D(img1**2,    -1, window)[5:-5, 5:-5] - mu1_sq
    s2 = cv2.filter2D(img2**2,    -1, window)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(img1*img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    return (((2*mu1_mu2+C1)*(2*s12+C2)) /
            ((mu1_sq+mu2_sq+C1)*(s1+s2+C2))).mean()


def frame_ssim(f1, f2):
    """f1, f2: HxWx3 uint8 numpy arrays."""
    f1 = cv2.resize(f1, (f2.shape[1], f2.shape[0])).astype(np.float64) / 255.0
    f2 = f2.astype(np.float64) / 255.0
    return np.mean([_ssim_single_channel(f1[:,:,c], f2[:,:,c]) for c in range(3)])


# ── video I/O ─────────────────────────────────────────────────────────────────

def read_video(path, max_frames=None):
    """Returns list of HxWx3 uint8 RGB frames."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def frames_to_tensor(frames, target_frames, size=224):
    """
    Convert list of HxWxC uint8 frames → float32 tensor CTHW in [0,1].
    Pads with last frame if shorter than target_frames.
    Resizes spatially to size×size.
    """
    if not frames:
        return None
    while len(frames) < target_frames:
        frames.append(frames[-1])
    frames = frames[:target_frames]
    # HxWxC → float [0,1]
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0   # T H W C
    t = torch.from_numpy(arr).permute(3, 0, 1, 2)               # C T H W
    # spatial resize
    t = F.interpolate(
        t.unsqueeze(0), size=(t.shape[1], size, size),
        mode='trilinear', align_corners=False
    ).squeeze(0)
    return t  # C T H W in [0,1]


# ── FVD helpers ───────────────────────────────────────────────────────────────

def load_i3d(device):
    from metrics.fvd.styleganv.fvd import load_i3d_pretrained
    return load_i3d_pretrained(device=device)


def extract_fvd_feats(video_tensors, i3d, device, batch_size=8):
    """
    video_tensors: list of C T H W tensors in [0,1].
    Returns numpy array of shape (N, 400).
    """
    from metrics.fvd.styleganv.fvd import get_fvd_feats, preprocess_single
    all_feats = []
    for start in range(0, len(video_tensors), batch_size):
        batch = video_tensors[start:start + batch_size]
        # preprocess: CTHW [0,1] → CTHW [-1,1] at 224×224
        batch_pp = torch.stack([preprocess_single(v) for v in batch]).to(device)  # B C T H W
        feats = get_fvd_feats(batch_pp, i3d=i3d, device=device)   # (B, 400)
        all_feats.append(feats)
    return np.vstack(all_feats)


def compute_fvd(feats_gt, feats_pred):
    from metrics.fvd.styleganv.fvd import frechet_distance
    return frechet_distance(feats_pred, feats_gt)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir",    required=True)
    p.add_argument("--pred_dir",  required=True)
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=64,
                   help="Max frames per video (pad shorter, truncate longer)")
    p.add_argument("--no_fvd",   action="store_true",
                   help="Skip FVD (faster, no I3D model needed)")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--fvd_size", type=int, default=224,
                   help="Spatial size for FVD I3D input")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    gt_files   = {f for f in os.listdir(args.gt_dir)   if f.endswith(".mp4")}
    pred_files = {f for f in os.listdir(args.pred_dir) if f.endswith(".mp4")}
    common = sorted(gt_files & pred_files)

    if not common:
        print("No matching .mp4 files found in both directories.")
        print(f"  gt_dir   ({args.gt_dir})  : {len(gt_files)} .mp4 files")
        print(f"  pred_dir ({args.pred_dir}): {len(pred_files)} .mp4 files")
        return

    if args.max_videos:
        common = common[:args.max_videos]
    print(f"Evaluating {len(common)} video pairs  (max_frames={args.max_frames})")

    # Load I3D once if needed
    i3d = None
    if not args.no_fvd:
        print("Loading I3D model for FVD...")
        i3d = load_i3d(device)

    ssim_scores   = []
    gt_tensors    = []
    pred_tensors  = []

    for fname in tqdm(common, desc="Reading videos"):
        gt_frames   = read_video(os.path.join(args.gt_dir,   fname), args.max_frames)
        pred_frames = read_video(os.path.join(args.pred_dir, fname), args.max_frames)

        if not gt_frames or not pred_frames:
            continue

        # ── SSIM ──
        n = min(len(gt_frames), len(pred_frames))
        scores = []
        for i in range(n):
            scores.append(frame_ssim(gt_frames[i], pred_frames[i]))
        ssim_scores.append(np.mean(scores))

        # ── FVD tensors ──
        if i3d is not None:
            t_gt   = frames_to_tensor(gt_frames,   args.max_frames, args.fvd_size)
            t_pred = frames_to_tensor(pred_frames, args.max_frames, args.fvd_size)
            if t_gt is not None and t_pred is not None:
                gt_tensors.append(t_gt)
                pred_tensors.append(t_pred)

    # ── Print SSIM ──
    print(f"\n=== SSIM ({len(ssim_scores)} videos) ===")
    if ssim_scores:
        print(f"  Mean : {np.mean(ssim_scores):.4f}")
        print(f"  Std  : {np.std(ssim_scores):.4f}")
        print(f"  Min  : {np.min(ssim_scores):.4f}")
        print(f"  Max  : {np.max(ssim_scores):.4f}")
    else:
        print("  No valid pairs.")

    # ── Print FVD ──
    if i3d is not None:
        print(f"\n=== FVD ({len(gt_tensors)} videos, max_frames={args.max_frames}) ===")
        if len(gt_tensors) < 2:
            print("  Need at least 2 videos for FVD — skipped.")
        else:
            print("  Extracting I3D features for GT videos...")
            feats_gt   = extract_fvd_feats(gt_tensors,   i3d, device)
            print("  Extracting I3D features for pred videos...")
            feats_pred = extract_fvd_feats(pred_tensors, i3d, device)
            fvd = compute_fvd(feats_gt, feats_pred)
            print(f"  FVD  : {fvd:.2f}")
            print("  (lower is better; 0 = identical distributions)")


if __name__ == "__main__":
    main()
