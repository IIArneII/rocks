import json
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def masks2rle(mask: np.ndarray) -> dict:
    """
    Encode a binary mask to COCO RLE.
    """
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

device = "cuda"

SCRIPT      = Path(__file__).parent
SAM_CHECKPT = SCRIPT / "checkpoints" / "sam_vit_h.pth"
MODEL_TYPE  = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=str(SAM_CHECKPT))
sam.to(device=device)

amg = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side        = 32,
    pred_iou_thresh        = 0.84,
    stability_score_thresh = 0.8,
    min_mask_region_area   = 150,
)

RAW_DIR  = Path("regions_test")
OUT_JSON = Path("sam1_masks") / "sam1_labels.json"
OUT_JSON.parent.mkdir(exist_ok=True, parents=True)

records = []

for img_id, img_path in enumerate(tqdm(sorted(RAW_DIR.glob("*.png")))):
    bgr = cv2.imread(str(img_path))
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    with torch.inference_mode(), torch.autocast(device):
        masks = amg.generate(img)

    for ann_id, m in enumerate(masks):
        records.append({
            "image_id"     : img_id,
            "ann_id"       : ann_id,
            "file_name"    : img_path.name,
            "height"       : H,
            "width"        : W,
            "segmentation" : masks2rle(m["segmentation"]),
            "area"         : int(m["area"]),
            "bbox"         : [float(x) for x in m["bbox"]],
            "iou"          : float(m.get("predicted_iou", 0.0)),
            "stab"         : float(m.get("stability_score", 0.0)),
        })

with open(OUT_JSON, "w") as f:
    json.dump(records, f)

print(f"Wrote {len(records)} masks to {OUT_JSON}")