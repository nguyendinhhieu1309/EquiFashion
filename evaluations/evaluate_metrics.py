"""Compute FID, Coverage (PRDC), and CLIP-S between two folders."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch


def _compute_fid(real_dir: str, fake_dir: str, num_workers: int, batch_size: int) -> float:
    from cleanfid import fid

    return float(
        fid.compute_fid(
            real_dir,
            fake_dir,
            mode="clean",
            num_workers=num_workers,
            batch_size=batch_size,
            model_name="inception_v3",
        )
    )


def _compute_prdc_coverage(real_dir: str, fake_dir: str, num_workers: int, batch_size: int) -> float:
    from cleanfid import prdc

    scores = prdc.prdc_scores(
        real_dir,
        fake_dir,
        mode="clean",
        model_name="inception_v3",
        num_workers=num_workers,
        batch_size=batch_size,
        verbose=True,
    )
    return float(scores["coverage"])


def _compute_clip_s(real_dir: str, fake_dir: str, device: str, batch_size: int, num_workers: int) -> Optional[float]:
    try:
        import clip  # type: ignore
    except Exception:
        return None

    from PIL import Image
    from torch.utils.data import Dataset, DataLoader

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    class PairedImageTextDataset(Dataset):
        def __init__(self, images_dir: str, texts_dir: str, preprocess):
            self.images_dir = images_dir
            self.texts_dir = texts_dir
            self.preprocess = preprocess
            self.items = []

            for fn in sorted(os.listdir(images_dir)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in IMAGE_EXTS:
                    continue
                txt_path = os.path.join(texts_dir, f"{stem}.txt")
                if os.path.exists(txt_path):
                    self.items.append((os.path.join(images_dir, fn), txt_path))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            img_path, txt_path = self.items[idx]
            img = Image.open(img_path).convert("RGB")
            with open(txt_path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            return self.preprocess(img), txt

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    texts_dir = fake_dir
    ds = PairedImageTextDataset(fake_dir, texts_dir, preprocess)
    if len(ds) == 0:
        return None

    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    sims = []
    with torch.no_grad():
        for imgs, texts in dl:
            imgs = imgs.to(device)
            tokens = clip.tokenize(list(texts)).to(device)
            img_feat = model.encode_image(imgs)
            txt_feat = model.encode_text(tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat * txt_feat).sum(dim=-1)
            sims.append(sim.detach().cpu())

    sims_t = torch.cat(sims, dim=0)
    return float(sims_t.mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True, help="Folder containing real images")
    parser.add_argument("--fake_dir", type=str, required=True, help="Folder containing generated images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    fid_score = _compute_fid(args.real_dir, args.fake_dir, args.num_workers, args.batch_size)
    coverage = _compute_prdc_coverage(args.real_dir, args.fake_dir, args.num_workers, args.batch_size)
    clip_s = _compute_clip_s(args.real_dir, args.fake_dir, args.device, args.batch_size, args.num_workers)

    print("=== Metrics ===")
    print(f"FID: {fid_score:.4f}")
    print(f"Coverage: {coverage:.4f}")
    if clip_s is None:
        print("CLIP-S: N/A (missing paired captions or 'clip' package not available)")
    else:
        print(f"CLIP-S: {clip_s:.4f}")


if __name__ == "__main__":
    main()

