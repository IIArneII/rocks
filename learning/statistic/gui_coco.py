"""
--------------------------------------------------------------------
Interactive rock-size analysis using *ground-truth* COCO masks.

Requirements
------------
pip install pycocotools opencv-python numpy matplotlib pandas

Usage
-----
python gui_coco.py --ann data/coco/val.json --image data/raw_frames/141.png
--------------------------------------------------------------------
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from common import ScaleSelector, plot_size_distribution, overlay_results


# --------------------------- Analysis -------------------------------- #
def analyse_masks(masks, mm_per_px):
    """
    Compute fragment metrics.

    Parameters
    ----------
    masks : list[np.ndarray{bool}]  – one binary mask per fragment (H×W)
    """
    n = len(masks)
    colours = (np.random.rand(n, 3) * 255).astype(np.uint8)
    H, W = masks[0].shape
    colour_mask = np.zeros((H, W, 3), np.uint8)

    rows = []
    for idx, m in enumerate(masks):
        area_px = int(m.sum())
        area_mm2 = area_px * mm_per_px ** 2
        equiv_diam = np.sqrt(4 * area_mm2 / np.pi)

        # bounding box
        ys, xs = np.where(m)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        length_px = max(x1 - x0, y1 - y0)
        width_px = min(x1 - x0, y1 - y0)

        rows.append(dict(
            id=idx,
            area_px=area_px,
            area_mm2=area_mm2,
            equiv_diam_mm=equiv_diam,
            max_length_mm=length_px * mm_per_px,
            min_length_mm=width_px * mm_per_px
        ))
        colour_mask[m] = colours[idx]

    return pd.DataFrame(rows), colour_mask


# --------------------------- Main ------------------------------------ #
def main():
    ap = argparse.ArgumentParser("Rock size distribution using COCO annotations")
    ap.add_argument('--ann', required=True, help='COCO JSON with instance masks')
    ap.add_argument('--image', required=True, help='Path to the image to analyse')
    ap.add_argument('--images-dir', default='', help='Prefix to prepend to file_name from JSON')
    args = ap.parse_args()

    coco = COCO(args.ann)

    # 1. load image
    img_path = Path(args.image)
    if not img_path.is_file():
        sys.exit("Image not found")
    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]

    # 2. find matching image record in JSON
    name = img_path.name
    img_entry = next((img for img in coco.dataset['images']
                      if img['file_name'] == name), None)
    if img_entry is None:
        sys.exit(f"Image {name} not present in annotation file")
    img_id = img_entry['id']

    # 3. decode masks
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    masks = []
    for a in anns:
        m = coco.annToMask(a).astype(bool)
        if m.sum() == 0:
            continue
        masks.append(m)
    if not masks:
        sys.exit("No masks for that image")

    # 4. reference scale
    selector = ScaleSelector()
    mm_per_px, scale_pts = selector.get_scale(img_bgr.copy())

    # 5. analysis
    df, colour_mask = analyse_masks(masks, mm_per_px)
    print(df.head())

    out_dir = img_path.parent
    base = img_path.stem

    # 6. graphics
    overlay_results(img_bgr, colour_mask, scale_pts, mm_per_px,
            str(out_dir / f"{base}_gt_overlay.png"), "Masks (GT)")
    plot_size_distribution(df, str(out_dir / f"{base}_gt_size_curve.png"))

    # 7. csv
    csv_fn = out_dir / f"{base}_gt_sizes.csv"
    df.to_csv(csv_fn, index=False)
    print("Saved CSV to", csv_fn)


if __name__ == '__main__':
    main()