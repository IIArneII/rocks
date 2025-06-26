from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2, torch
from pathlib import Path
import numpy as np

device = "cuda"
SCRIPT = Path(__file__).parent
CKPT   = str(SCRIPT/"checkpoints/sam2.1_hiera_large.pt")
CFG    = str(SCRIPT/"checkpoints/sam2.1_hiera_l.yaml")

model = build_sam2(CFG, CKPT).to(device)

amg = SAM2AutomaticMaskGenerator(
        model,
        points_per_side        = 24,
        pred_iou_thresh        = 0.7,
        stability_score_thresh = 0.7,
        box_nms_thresh         = 0.7,
        min_mask_region_area   = 250,
    )

save_dir = Path("sam2_masks"); save_dir.mkdir(exist_ok=True)

for img_path in Path("regions_test").glob("*.png"):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    with torch.inference_mode(), torch.autocast(device):
        masks = amg.generate(img)

    for k, m in enumerate(masks):
        out = (m["segmentation"].astype(np.uint8) * 255)
        cv2.imwrite(str(save_dir / f"{img_path.stem}_{k:03}.png"), out)