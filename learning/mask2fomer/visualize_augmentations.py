import json
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
import random
from rock_mask2former_train import (
    CFG, build_transforms, coco_poly_to_mask
)

def visualize_augmentations(output_dir, num_samples=20):
    """Applies augmentation pipeline and saves sample images for inspection"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(CFG.coco_json) as f:
        coco_data = json.load(f)
    
    image_ann_map = {img["id"]: img for img in coco_data["images"]}
    anns_by_image = {}
    for ann in coco_data["annotations"]:
        if not ann.get("iscrowd", 0):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
    
    transforms = build_transforms()
    
    for i, (img_id, anns) in enumerate(anns_by_image.items()):
        if i >= num_samples:
            break
            
        img_info = image_ann_map[img_id]
        img_path = CFG.frames_dir / img_info["file_name"]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        
        masks = []
        for ann in anns:
            mask = coco_poly_to_mask(
                ann["segmentation"], 
                img_info["height"], 
                img_info["width"]
            )
            masks.append(mask)
        
        augmented = transforms(image=img, masks=masks)
        aug_img = augmented["image"]
        aug_masks = augmented["masks"]
        
        vis_img = aug_img.copy()
        for mask in aug_masks:
            color = [random.randint(0, 255) for _ in range(3)]
            vis_img = np.where(mask[..., None] > 0, 
                              (vis_img * 0.7 + np.array(color) * 0.3).astype(np.uint8),
                              vis_img)
        
        orig_path = output_dir / f"{img_id}_original.jpg"
        aug_path = output_dir / f"{img_id}_augmented.jpg"
        vis_path = output_dir / f"{img_id}_visualization.jpg"
        
        cv2.imwrite(str(orig_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(aug_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(vis_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        print(f"Saved visualization for {img_id} to {output_dir}")

if __name__ == "__main__":
    visualize_augmentations("output/augmentation_visualization")