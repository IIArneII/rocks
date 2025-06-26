"""
--------------------------------------------------------------------
Interactive rock-size analysis based on a trained Cascade Mask R-CNN.

Requirements
------------
pip install mmcv-full mmdet mmengine opencv-python numpy matplotlib pandas

Usage
-----
python statistic/gui.py --config configs/resnetx_mask_rcnn.py --checkpoint work_dirs/resnetx_mask_rcnn/epoch_48.pth --image data/raw_frames/112.png
--------------------------------------------------------------------
"""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mmengine import Config
from mmdet.apis import init_detector, inference_detector

from common import ScaleSelector, plot_size_distribution, overlay_results


# --------------------------- Analysis part --------------------------- #
def analyse_masks(result, mm_per_px, pred_score_thr=0.3):
    """
    Convert segmentation result to a DataFrame with size metrics in mm.
    Returns: df, coloured_mask (RGB)
    """
    pred_instances = result.pred_instances.cpu()
    scores = pred_instances.scores
    keep = scores >= pred_score_thr
    masks = pred_instances.masks[keep].numpy()
    bboxes = pred_instances.bboxes[keep].numpy()
    scores = scores[keep].numpy()

    n = masks.shape[0]
    colors = (np.random.rand(n, 3) * 255).astype(np.uint8)

    measurements = []
    coloured_mask = np.zeros((*masks[0].shape, 3), dtype=np.uint8)

    for idx in range(n):
        mask = masks[idx].astype(np.uint8)
        area_px = mask.sum()
        area_mm2 = area_px * (mm_per_px ** 2)
        equiv_diam_mm = np.sqrt(4 * area_mm2 / np.pi)   # 2D equivalent circle Ø

        # Bounding box based sizes
        x_min, y_min, x_max, y_max = bboxes[idx]
        length_px = max(x_max - x_min, y_max - y_min)
        width_px  = min(x_max - x_min, y_max - y_min)

        measurements.append(dict(
            id=idx,
            score=scores[idx],
            area_px=int(area_px),
            area_mm2=area_mm2,
            equiv_diam_mm=equiv_diam_mm,
            max_length_mm=length_px * mm_per_px,
            min_length_mm=width_px * mm_per_px
        ))

        coloured_mask[mask.astype(bool)] = colors[idx]

    df = pd.DataFrame(measurements)
    return df, coloured_mask

# --------------------------- Main ------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description='Rock size distribution via instance segmentation.')
    parser.add_argument('--config', required=True, help='MMDetection config file')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth weights')
    parser.add_argument('--image', required=True, help='Input image')
    parser.add_argument('--thr', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--min-size-mm', type=float, default=0, help='Min equivalent diameter in mm for analysis')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    args = parser.parse_args()

    # 1. Load model
    cfg = Config.fromfile(args.config)
    cfg.test_pipeline = [dict(type='LoadImageFromFile'),
                         dict(type='Resize', scale=(600, 600), keep_ratio=True),
                         dict(type='PackDetInputs')]
    model = init_detector(cfg, args.checkpoint, device=args.device)

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
    result = inference_detector(model, img)

    # 5. Analyse masks
    df, coloured_mask = analyse_masks(result, mm_per_px, args.thr)
    if args.min_size_mm > 0:
        orig_count = len(df)
        df = df[df['equiv_diam_mm'] >= args.min_size_mm].copy()
        print(f"Filtered out {orig_count - len(df)} fragments smaller than {args.min_size_mm} mm.")

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

    hist_png = img_path.with_name(img_path.stem + '_size_curve.png')
    plot_size_distribution(df, hist_png)

    # 6. Overlay + save
    out_img = img_path.with_name(img_path.stem + '_segmented.png')
    overlay_results(img, coloured_mask, scale_pts, mm_per_px, str(out_img))
    print('Overlay saved to', out_img)

    # 7. Save CSV
    csv_path = img_path.with_name(img_path.stem + '_sizes.csv')
    df.to_csv(csv_path, index=False)
    print('Measurement table saved to', csv_path)


if __name__ == '__main__':
    main()