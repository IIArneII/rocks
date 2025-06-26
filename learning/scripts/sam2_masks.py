import json, zlib, base64, cv2, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from pycocotools import mask as mask_utils

def masks2rle(m):
    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

device = "cuda"
SCRIPT = Path(__file__).parent
CKPT   = str(SCRIPT/"checkpoints/sam2.1_hiera_large.pt")
CFG    = str(SCRIPT/"checkpoints/sam2.1_hiera_l.yaml")
model = build_sam2(CFG, CKPT).to(device)
amg   = SAM2AutomaticMaskGenerator(model,
                                   points_per_side        = 24,
                                   pred_iou_thresh        = 0.84,
                                   stability_score_thresh = 0.2,
                                   min_mask_region_area   = 150)

RAW      = Path("regions_test")
OUT_JSON = Path("sam2_masks/sam2_labels.json")
records  = []

for img_id, png in enumerate(tqdm(sorted(RAW.glob("*.png")))):
    img = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    with torch.inference_mode(), torch.autocast(device):
        masks = amg.generate(img)

    for ann_id, m in enumerate(masks):
        records.append(dict(
            image_id  = img_id,
            ann_id    = ann_id,
            file_name = png.name,
            height    = H,
            width     = W,
            segmentation = masks2rle(m["segmentation"]),
            area         = int(m["area"]),
            bbox         = [float(x) for x in m["bbox"]],
            iou          = float(m["predicted_iou"]),
            stab         = float(m["stability_score"]),
        ))

json.dump(records, open(OUT_JSON, "w"))
print("Wrote", len(records), "masks")