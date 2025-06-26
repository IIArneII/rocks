import torch, cv2
from PIL import Image
from pathlib import Path
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)
import matplotlib.pyplot as plt
import numpy as np

ckpt_dir = Path("output/checkpoints/best_ep495")
device   = torch.device("cuda:0")

model      = (Mask2FormerForUniversalSegmentation
              .from_pretrained(ckpt_dir).to(device).eval())
processor   = Mask2FormerImageProcessor.from_pretrained(ckpt_dir)

img_path = "data/raw_frames/250.png"
image    = Image.open(img_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    outputs = model(**inputs) 

processed = processor.post_process_instance_segmentation(
    outputs=outputs,
    threshold      = 0.5,
    mask_threshold = 0.5,
    target_sizes   = [image.size[::-1]],
)[0]


seg_map = processed["segmentation"]
segments_info = processed["segments_info"]

scores = [seg["score"] for seg in segments_info]
labels = [seg["label_id"] for seg in segments_info]

palette = np.random.RandomState(42).rand(256,3) * 255
overlay = np.array(image).copy()

for inst_id in np.unique(seg_map):
    if inst_id == 0:
        continue
    color_idx = int(inst_id % len(palette))
    color     = palette[color_idx]
    mask_bool = seg_map == inst_id
    overlay[mask_bool] = (0.4*overlay[mask_bool] + 0.6*color).astype(np.uint8)

output_path = Path(img_path).with_name(f"{Path(img_path).stem}_overlay.png")
Image.fromarray(overlay).save(output_path)
print(f"Saved result to: {output_path}")