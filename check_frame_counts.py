"""
Check that every video in the metadata has matching frame counts across:
  - original video (.mp4)
  - DWPose skeleton (.pkl)
  - HAMER rendered video (.mp4)

Usage:
    python check_frame_counts.py --config configs/vq/vq_multicond_RWTH_compress.yaml
    python check_frame_counts.py --config configs/vq/vq_multicond_RWTH_compress.yaml --split val
"""

import argparse
import json
import os
import pickle

import cv2
from omegaconf import OmegaConf
from tqdm import tqdm


def frame_count_mp4(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n if n > 0 else None


def frame_count_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return len(data)


def check(meta_path, root, sk_root, hamer_root):
    entries = json.load(open(meta_path))
    ok = bad = missing = 0
    issues = []

    for entry in tqdm(entries, desc=os.path.basename(meta_path)):
        rel = entry["video"]
        base = os.path.basename(rel)

        vid_path   = os.path.join(root,       rel)
        sk_path    = os.path.join(sk_root,    base.replace(".mp4", ".pkl"))
        hamer_path = os.path.join(hamer_root, base)

        # existence
        absent = [p for p in (vid_path, sk_path, hamer_path) if not os.path.exists(p)]
        if absent:
            missing += 1
            issues.append(("MISSING", base, str(absent)))
            continue

        vid_n   = frame_count_mp4(vid_path)
        sk_n    = frame_count_pkl(sk_path)
        hamer_n = frame_count_mp4(hamer_path)

        if vid_n is None or hamer_n is None:
            issues.append(("UNREADABLE", base, f"vid={vid_n} hamer={hamer_n}"))
            bad += 1
            continue

        if vid_n == sk_n == hamer_n:
            ok += 1
        else:
            bad += 1
            issues.append(("MISMATCH", base,
                           f"vid={vid_n}  sk={sk_n}  hamer={hamer_n}"))

    print(f"\n=== {os.path.basename(meta_path)} ===")
    print(f"  OK      : {ok}")
    print(f"  Missing : {missing}")
    print(f"  Mismatch: {bad}")
    print(f"  Total   : {len(entries)}")

    if issues:
        print(f"\n  First 20 issues:")
        for tag, name, detail in issues[:20]:
            print(f"    [{tag}] {name}  —  {detail}")
        if len(issues) > 20:
            print(f"    ... and {len(issues) - 20} more")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.split == "val":
        meta_paths = list(cfg.validation_data.meta_paths)
    else:
        meta_paths = list(cfg.dataset.meta_paths)

    roots       = list(cfg.dataset.roots)
    sk_roots    = list(cfg.dataset.sk_roots)
    hamer_roots = list(cfg.dataset.hamer_roots)

    all_issues = []
    for meta, root, sk_root, hamer_root in zip(meta_paths, roots, sk_roots, hamer_roots):
        issues = check(meta, root, sk_root, hamer_root)
        all_issues.extend(issues)

    print(f"\nTotal issues across all splits: {len(all_issues)}")


if __name__ == "__main__":
    main()
