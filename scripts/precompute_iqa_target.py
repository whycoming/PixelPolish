"""Precompute target IQA distribution (μ, σ) per head over a reference set.

This anchors the GRPO terminal reward at "natural-typical IQA" rather than
"more is more". Run once on a high-quality reference image set, write JSON,
load at training time.

Usage:
    python -m scripts.precompute_iqa_target \
        --input-dir /path/to/reference/images \
        --heads clipiqa musiq \
        --image-size 256 \
        --output configs/iqa_target.json \
        [--brightness-min 0.3 --brightness-max 0.85] \
        [--max-images 500]
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.rewards.iqa import build_head


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_image(path: Path, size: int) -> Tensor:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True,
                    help="folder of reference (high-quality) images, recursive")
    ap.add_argument("--heads", nargs="+", default=["clipiqa", "musiq"])
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--output", required=True, help="JSON path to write (μ, σ) per head")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-images", type=int, default=500,
                    help="cap to avoid hours of pyiqa inference")
    ap.add_argument("--brightness-min", type=float, default=0.0,
                    help="reject images with mean<this in [0,1] (filter underexposed)")
    ap.add_argument("--brightness-max", type=float, default=1.0,
                    help="reject images with mean>this in [0,1] (filter overexposed)")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    paths: List[Path] = sorted(
        p for p in Path(args.input_dir).rglob("*")
        if p.suffix.lower() in IMG_EXTS and p.is_file()
    )
    if not paths:
        raise SystemExit(f"no images found under {args.input_dir}")
    print(f"[precompute] found {len(paths)} candidate images")

    heads = {}
    for n in args.heads:
        h = build_head(n, device=args.device)
        if h is None:
            print(f"[precompute] skipping unavailable head: {n}")
            continue
        heads[n] = h._compute
    if not heads:
        raise SystemExit("no IQA heads loaded")

    scores = {n: [] for n in heads}
    kept_paths: List[str] = []
    rejected = {"too_dark": 0, "too_bright": 0}
    for i, p in enumerate(paths):
        if len(kept_paths) >= args.max_images:
            break
        try:
            x = _load_image(p, args.image_size)
        except Exception as e:
            print(f"[precompute] skip {p.name}: {e}")
            continue
        m = float(x.mean())
        if m < args.brightness_min:
            rejected["too_dark"] += 1
            continue
        if m > args.brightness_max:
            rejected["too_bright"] += 1
            continue
        x = x.to(args.device)
        for n, fn in heads.items():
            try:
                s = float(fn(x).view(-1)[0])
                scores[n].append(s)
            except Exception as e:
                print(f"[precompute] {n} failed on {p.name}: {e}")
        kept_paths.append(str(p))
        if (i + 1) % 25 == 0:
            print(f"  ... processed {i+1} (kept {len(kept_paths)})")

    if not kept_paths:
        raise SystemExit("all images rejected by brightness filter")

    out = {
        "n_images": len(kept_paths),
        "image_size": int(args.image_size),
        "brightness_filter": [args.brightness_min, args.brightness_max],
        "heads": {},
    }
    print(f"\n[precompute] computed over {len(kept_paths)} images "
          f"(rejected dark={rejected['too_dark']}, bright={rejected['too_bright']})")
    for n in heads:
        arr = np.asarray(scores[n], dtype=np.float64)
        if arr.size == 0:
            print(f"  {n}: NO SCORES")
            continue
        mu, sd = float(arr.mean()), float(arr.std())
        if sd < 1e-6:
            print(f"  WARN {n}: σ ~ 0, training would divide by 0; clamping to 1e-3")
            sd = 1e-3
        out["heads"][n] = {"mu": mu, "sigma": sd, "n": int(arr.size)}
        print(f"  {n:10s}  μ={mu:+9.4f}  σ={sd:9.4f}  (n={arr.size})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[precompute] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
