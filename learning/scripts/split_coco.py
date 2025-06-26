import json
import random
import argparse
from pathlib import Path

def split_coco(
    ann_file: str,
    train_file: str,
    val_file: str,
    val_ratio: float = 0.2,
    seed: int = 42
):
    """
    Split a COCO JSON annotations file into train and val subsets.
    """
    random.seed(seed)

    coco = json.load(open(ann_file, 'r'))
    images = coco['images']
    annotations = coco['annotations']

    # shuffle images
    random.shuffle(images)

    # split index
    n_val = int(len(images) * val_ratio)
    val_images = images[:n_val]
    train_images = images[n_val:]

    val_image_ids = {img['id'] for img in val_images}
    train_image_ids = {img['id'] for img in train_images}

    # filter annotations
    val_anns = [ann for ann in annotations if ann['image_id'] in val_image_ids]
    train_anns = [ann for ann in annotations if ann['image_id'] in train_image_ids]

    # prepare output dicts (keep categories, info, licenses if present)
    base = {k: coco[k] for k in coco if k not in ('images','annotations')}
    train_coco = {
        **base,
        'images': train_images,
        'annotations': train_anns
    }
    val_coco = {
        **base,
        'images': val_images,
        'annotations': val_anns
    }

    # write out
    Path(train_file).write_text(json.dumps(train_coco, indent=2))
    Path(val_file).write_text(json.dumps(val_coco, indent=2))
    print(f"Written {len(train_images)} images + {len(train_anns)} anns to {train_file}")
    print(f"Written {len(val_images)} images + {len(val_anns)} anns to {val_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Split a COCO JSON into train/val JSONs"
    )
    p.add_argument('--ann-file',      required=True, help="input COCO .json")
    p.add_argument('--train-file',    default='train.json', help="output train .json")
    p.add_argument('--val-file',      default='val.json',   help="output val .json")
    p.add_argument('--val-ratio',     type=float, default=0.2, help="fraction for val (0–1)")
    p.add_argument('--seed',          type=int,   default=42,  help="random seed")
    args = p.parse_args()
    split_coco(
        ann_file   = args.ann_file,
        train_file = args.train_file,
        val_file   = args.val_file,
        val_ratio  = args.val_ratio,
        seed       = args.seed
    )

# python test2.py --ann-file data/coco/auto.json --train-file data/coco/train.json --val-file   data/coco/val.json --val-ratio  0.3 --seed 123