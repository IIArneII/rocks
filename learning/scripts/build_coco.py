import json, pathlib

IN_JSON  = "sam2_masks/sam2_labels.json"
OUT_JSON = "sam2_masks/coco/auto_trainval.json"
IMG_ROOT = "regions_test/"

records = json.load(open(IN_JSON))
records.sort(key=lambda r: r["image_id"])

coco = dict(images=[], annotations=[], categories=[{"id":1,"name":"rock"}])
img_seen = set()
ann_id = 1

for r in records:
    if r["image_id"] not in img_seen:
        coco["images"].append(dict(
            id        = r["image_id"],
            file_name = r["file_name"],
            width     = r["width"],
            height    = r["height"]))
        img_seen.add(r["image_id"])

    coco["annotations"].append(dict(
        id            = ann_id,
        image_id      = r["image_id"],
        category_id   = 1,
        segmentation  = r["segmentation"],
        area          = r["area"],
        bbox          = r["bbox"],
        iscrowd       = 0))
    ann_id += 1

pathlib.Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
json.dump(coco, open(OUT_JSON, "w"))
print("COCO written:", OUT_JSON)