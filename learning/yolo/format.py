import json
from pathlib import Path
from pycocotools import mask as maskUtils
import cv2
import numpy as np
from ultralytics.data.converter import convert_coco

json_path = Path("data/coco/instances.json")
data = json.loads(json_path.read_text())
for ann in data["annotations"]:
    seg = ann.get("segmentation")
    if isinstance(seg, dict):
        rle = maskUtils.frPyObjects(seg, seg["size"][0], seg["size"][1])
        if isinstance(rle, list):
            rle = rle[0]
        mask = maskUtils.decode(rle)

        contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in contours:
            pts = c.reshape(-1, 2)
            if len(pts) >= 3:
                polys.append(pts.flatten().tolist())
        ann["segmentation"] = polys

json_path.write_text(json.dumps(data))

convert_coco(
    labels_dir="data/coco",
    save_dir="data/yolo",
    use_segments=True
)