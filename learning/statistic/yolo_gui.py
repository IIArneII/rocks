"""
--------------------------------------------------------------------
Interactive rock-size analysis based on a trained YOLO-Seg model.

Requirements
------------
pip install ultralytics opencv-python numpy matplotlib pandas

Example
-------
python statistic/yolo_gui.py --weights outputs/yolo/best.pt --image data/raw_frames/141.png
--------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from common import ScaleSelector, plot_size_distribution, overlay_results

# --------------------------- Analysis part --------------------------- #
def analyse_masks_yolo(result, mm_per_px: float, pred_score_thr: float = 0.3):
    """
    Convert YOLO segmentation result to a DataFrame with size metrics in mm.
    Returns (df, coloured_mask)
    """
    if result.masks is None or result.masks.data is None:
        sys.exit("Model did not return masks. Make sure you are using a *seg* variant of YOLO.")

    masks = result.masks.data.cpu().numpy()           # (N, H, W)  bool
    scores = result.boxes.conf.cpu().numpy()          # (N,)
    bboxes = result.boxes.xyxy.cpu().numpy()          # (N, 4)

    keep = scores >= pred_score_thr
    masks, scores, bboxes = masks[keep], scores[keep], bboxes[keep]

    n = masks.shape[0]
    if n == 0:
        sys.exit('No fragments above score threshold.')

    colors = (np.random.rand(n, 3) * 255).astype(np.uint8)
    coloured_mask = np.zeros((*masks[0].shape, 3), dtype=np.uint8)

    measurements = []
    for idx in range(n):
        mask = masks[idx].astype(np.uint8)
        area_px = mask.sum()
        area_mm2 = area_px * (mm_per_px ** 2)
        equiv_diam_mm = np.sqrt(4 * area_mm2 / np.pi)

        x_min, y_min, x_max, y_max = bboxes[idx]
        length_px = max(x_max - x_min, y_max - y_min)
        width_px = min(x_max - x_min, y_max - y_min)

        measurements.append(
            dict(
                id=idx,
                score=scores[idx],
                area_px=int(area_px),
                area_mm2=area_mm2,
                equiv_diam_mm=equiv_diam_mm,
                max_length_mm=length_px * mm_per_px,
                min_length_mm=width_px * mm_per_px,
            )
        )
        coloured_mask[mask.astype(bool)] = colors[idx]

    df = pd.DataFrame(measurements)
    return df, coloured_mask


# --------------------------- Main ------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description='Rock size distribution via YOLO instance segmentation.')
    parser.add_argument('--weights', required=True, help='Path to *.pt weights or model name (e.g. yolov8n-seg.pt)')
    parser.add_argument('--image', required=True, help='Input image')
    parser.add_argument('--thr', type=float, default=0.3, help='Score threshold (confidence)')
    parser.add_argument('--device', default='', help='cuda:0, 0, 1 … or cpu (uses torch defaults if omitted)')
    args = parser.parse_args()

    # 1. Load model
    model = YOLO(args.weights)
    if args.device:
        model.to(args.device)

    # 2. Load image
    img_path = Path(args.image)
    if not img_path.is_file():
        sys.exit(f"Image not found: {img_path}")
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit('Failed to read image.')

    # 3. Ask user for reference scale
    scale_selector = ScaleSelector()
    mm_per_px, scale_pts = scale_selector.get_scale(img.copy())

    # 4. Run inference
    results = model(img, conf=args.thr, verbose=False)
    result = results[0]  # first (and only) image

    # 5. Analyse masks
    df, coloured_mask = analyse_masks_yolo(result, mm_per_px, args.thr)

    print('\n===== Rock size table (first 10 rows) =====')
    print(df.head(10).to_string(index=False, formatters={'area_mm2': '{:.1f}'.format,
                                                         'equiv_diam_mm': '{:.2f}'.format}))
    print('Total fragments:', len(df))

    # Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(df['equiv_diam_mm'], bins='auto', color='slateblue', alpha=0.7)
    plt.xlabel('Equivalent diameter [mm]')
    plt.ylabel('Count')
    plt.title('Rock-size distribution')
    plt.tight_layout()
    hist_png = img_path.with_name(img_path.stem + '_hist.png')
    plt.savefig(hist_png, dpi=150)
    print('Histogram saved to', hist_png)

    size_curve_png = img_path.with_name(img_path.stem + '_size_curve.png')
    plot_size_distribution(df, size_curve_png)

    # 6. Overlay + save
    out_img = img_path.with_name(img_path.stem + '_segmented.png')
    # Resize coloured_mask if needed to match image dimensions
    if img.shape[:2] != coloured_mask.shape[:2]:
        coloured_mask = cv2.resize(coloured_mask, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
    overlay_results(img, coloured_mask, scale_pts, mm_per_px, str(out_img))
    print('Overlay saved to', out_img)

    # 7. Save CSV
    csv_path = img_path.with_name(img_path.stem + '_sizes.csv')
    df.to_csv(csv_path, index=False)
    print('Measurement table saved to', csv_path)


if __name__ == '__main__':
    main()