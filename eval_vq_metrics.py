"""
Compute SSIM between original and VQ-reconstructed videos.

Usage:
    python eval_vq_metrics.py \
        --gt_dir  /path/to/dev_processed_videos \
        --pred_dir dev_vq_outputs

Both directories are searched for matching .mp4 files by filename.
Only files present in both directories are evaluated.
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def ssim_frame(img1, img2):
    C1, C2 = 0.01**2, 0.03**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12  = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    return (((2*mu1_mu2+C1)*(2*sigma12+C2)) /
            ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))).mean()


def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def video_ssim(gt_path, pred_path):
    gt_frames   = read_video_frames(gt_path)
    pred_frames = read_video_frames(pred_path)
    n = min(len(gt_frames), len(pred_frames))
    if n == 0:
        return None
    scores = []
    for i in range(n):
        gt   = cv2.resize(gt_frames[i],   (pred_frames[i].shape[1], pred_frames[i].shape[0]))
        pred = pred_frames[i]
        # per-channel SSIM, averaged
        ch_scores = [ssim_frame(gt[:,:,c].astype(np.float64)/255.0,
                                pred[:,:,c].astype(np.float64)/255.0)
                     for c in range(3)]
        scores.append(np.mean(ch_scores))
    return np.mean(scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir",   required=True, help="Directory with original .mp4 files")
    p.add_argument("--pred_dir", required=True, help="Directory with reconstructed .mp4 files")
    p.add_argument("--max_videos", type=int, default=None,
                   help="Cap number of videos (for quick runs)")
    args = p.parse_args()

    gt_files   = {f for f in os.listdir(args.gt_dir)   if f.endswith(".mp4")}
    pred_files = {f for f in os.listdir(args.pred_dir) if f.endswith(".mp4")}
    common = sorted(gt_files & pred_files)

    if not common:
        print("No matching .mp4 files found in both directories.")
        print(f"  gt_dir   : {len(gt_files)} files")
        print(f"  pred_dir : {len(pred_files)} files")
        return

    if args.max_videos:
        common = common[:args.max_videos]

    print(f"Evaluating {len(common)} video pairs...")
    ssim_scores = []
    failed = 0
    for fname in tqdm(common):
        score = video_ssim(
            os.path.join(args.gt_dir,   fname),
            os.path.join(args.pred_dir, fname),
        )
        if score is not None:
            ssim_scores.append(score)
        else:
            failed += 1

    if ssim_scores:
        print(f"\n=== SSIM Results ({len(ssim_scores)} videos) ===")
        print(f"  Mean  : {np.mean(ssim_scores):.4f}")
        print(f"  Std   : {np.std(ssim_scores):.4f}")
        print(f"  Min   : {np.min(ssim_scores):.4f}")
        print(f"  Max   : {np.max(ssim_scores):.4f}")
        if failed:
            print(f"  Failed: {failed} videos (empty or unreadable)")
    else:
        print("No valid video pairs could be evaluated.")


if __name__ == "__main__":
    main()
