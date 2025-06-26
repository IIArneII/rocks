import json, random, numpy as np, tqdm
from pycocotools import mask as coco_mask

def coco2pairs(coco_path, out_path, n_points=1):
    coco = json.load(open(coco_path))
    id2img = {i["id"]: i for i in coco["images"]}
    out = []
    for ann in tqdm.tqdm(coco["annotations"]):
        if isinstance(ann["segmentation"], list):
            rle = coco_mask.frPyObjects(ann["segmentation"], id2img[ann["image_id"]]["height"], id2img[ann["image_id"]]["width"])
            m = coco_mask.decode(rle)
        else:
            seg = ann["segmentation"]
            if isinstance(seg, dict) and isinstance(seg.get("counts"), list):
                seg = coco_mask.frPyObjects(seg, seg['size'][0], seg['size'][1])
            m = coco_mask.decode(seg)
        ys, xs = np.where(m)
        if len(xs)==0: continue
        for _ in range(n_points):
            i = random.randint(0, len(xs)-1)
            px, py = xs[i], ys[i]
            out.append({
                "image": id2img[ann["image_id"]]["file_name"],
                "point": [int(px), int(py)],
                "label": ann["category_id"],
                "mask_rle": {
                    'counts': coco_mask.encode(np.asfortranarray(m.astype(np.uint8)))['counts'].decode('utf-8'),
                    'size': m.shape
                }
            })
    json.dump(out, open(out_path, "w"))

if __name__ == "__main__":
    coco2pairs("data/coco/auto.json", "data/coco/pairs.json")