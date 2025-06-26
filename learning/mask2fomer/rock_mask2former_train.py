from pathlib import Path
import json, random, argparse
import albumentations as A
import cv2, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    get_linear_schedule_with_warmup,
)
from pycocotools import mask as coco_mask
from tensorboardX import SummaryWriter
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None,
                    help="Path to a folder previously saved with accelerator.save_state()")
args = parser.parse_args()


def set_seed(seed=42):
    random.seed(seed);  np.random.seed(seed)
    torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)
set_seed(1)


class CFG:
    coco_json   = Path("data/coco/instances.json")
    frames_dir  = Path("data/raw_frames")
    logdir      = Path("output/rock_mask2former")
    output_dir  = Path("output/rock_mask2former/checkpoints")

    model_ckpt  = "facebook/mask2former-swin-base-coco-instance"
    num_labels  = 1

    num_epochs   = 800
    lr           = 1e-4
    weight_decay = 1e-5
    warmup_steps = 200
    batch_size   = 4
    gradient_accum = 6
    freeze_layers  = 1
    eval_interval  = 5

    img_size   = 400
    max_copy_paste = 6
    minority_classes = ["ladle", "ladle_handle"]
    copy_paste_prob = 0.7

CLASSES = ["rock"]
CONTIG_ID = {name: idx for idx, name in enumerate(CLASSES)}

def coco_poly_to_mask(segmentation, h, w):
    """
    Accept polygon *or* RLE encodings and return a H×W uint8 mask.
    """
    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, h, w)
        rle  = coco_mask.merge(rles)

    elif isinstance(segmentation, dict):
        if isinstance(segmentation["counts"], list):
            rle = coco_mask.frPyObjects(segmentation, h, w)
        else:
            rle = segmentation
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")

    return coco_mask.decode(rle).astype(np.uint8)

class RockCocoDataset(Dataset):
    def __init__(self, json_path, imgs_dir, transforms, processor):
        self.dir  = Path(imgs_dir)
        self.proc = processor
        self.tfm  = transforms

        with open(json_path) as f:
            d = json.load(f)

        cat_name = {c["id"]: c["name"] for c in d["categories"]}

        self.cat2contig = {}
        for cid in cat_name:
            self.cat2contig[cid] = CONTIG_ID[cat_name[cid]]

        self.imgs = {img["id"]: img for img in d["images"]}

        anns_by_img = {}
        for a in d["annotations"]:
            if a.get("iscrowd", 0):
                continue
            anns_by_img.setdefault(a["image_id"], []).append(a)

        self.items = [(self.imgs[i], anns_by_img[i]) for i in anns_by_img]

        self.minority_indices = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self._get_item(idx)

    def _get_item(self, idx):
        img_info, annos = self.items[idx]
        path = self.dir / img_info["file_name"]
        img  = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

        inst_masks, inst_labels = [], []
        for a in annos:
            m = coco_poly_to_mask(
                a["segmentation"], img_info["height"], img_info["width"]
            )
            inst_masks.append(m.astype(np.uint8))
            inst_labels.append(self.cat2contig[a["category_id"]])

        t          = self.tfm(image=img, masks=inst_masks)
        img        = t["image"]
        inst_masks = t["masks"]

        keep_masks, keep_labels = [], []
        for m, lab in zip(inst_masks, inst_labels):
            if m.any():
                keep_masks.append(m)
                keep_labels.append(lab)

        if not keep_masks:
            return self.__getitem__((idx + 1) % len(self))

        id_mask   = np.zeros(keep_masks[0].shape, dtype=np.int32)
        mapping   = {}
        for inst_id, (m, lab) in enumerate(zip(keep_masks, keep_labels), start=1):
            id_mask[m > 0] = inst_id
            mapping[inst_id] = lab

        encoded = self.proc(
            images=[img],
            segmentation_maps=[id_mask],
            instance_id_to_semantic_id=[mapping],
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            return_tensors="pt",
        )

        encoded = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) else v[0])
                   for k, v in encoded.items()}
        encoded["pixel_values"] = encoded["pixel_values"].float()

        valid_mask = (encoded["class_labels"] < CFG.num_labels) & (encoded["class_labels"] != 255)
        if not valid_mask.all():
            invalid_labels = encoded["class_labels"][~valid_mask]
            print(f"⚠️ Warning: Found invalid class labels: {invalid_labels.tolist()}")
            encoded["class_labels"] = encoded["class_labels"][valid_mask]
            mask_list = [m for i, m in enumerate(encoded["mask_labels"]) if valid_mask[i]]
            if mask_list:
                encoded["mask_labels"] = torch.stack(mask_list)
            else:
                encoded["mask_labels"] = torch.zeros((0, *id_mask.shape[:2]), dtype=torch.float32)
                encoded["class_labels"] = torch.zeros(0, dtype=torch.long)
        else:
            if isinstance(encoded["mask_labels"], list):
                if encoded["mask_labels"]:
                    encoded["mask_labels"] = torch.stack(encoded["mask_labels"])
                else:
                    h, w = id_mask.shape[:2]
                    encoded["mask_labels"] = torch.zeros((0, h, w), dtype=torch.float32)

        assert len(encoded["mask_labels"]) == len(encoded["class_labels"])
        assert (encoded["class_labels"] < CFG.num_labels).all()
        assert (encoded["class_labels"] != 255).all()

        return encoded


def build_transforms():
    return A.Compose(
        [
            A.LongestMaxSize(max_size=CFG.img_size),
            A.PadIfNeeded(CFG.img_size, CFG.img_size, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.3),
            A.RandomFog(fog_coef_range=(0.01, 0.1), alpha_coef=0.05, p=0.1),
            A.OneOf([
                A.Compose([
                    A.RandomResizedCrop(
                        size=(CFG.img_size, CFG.img_size),
                        scale=(0.9, 1.0),
                        ratio=(0.9, 1.1),
                        interpolation=cv2.INTER_LINEAR,
                        p=0.7,
                    ),
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=20,
                        p=0.5
                    )
                ]),
                A.Rotate(limit=30, p=0.3)
            ], p=CFG.copy_paste_prob),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomSnow(p=0.1),
        ]
    )

def load_model(model_name):
    print("⏬ loading", model_name)
    m = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=CFG.num_labels,
            ignore_mismatched_sizes=True,
        )
    for n,p in m.model.pixel_level_module.named_parameters():
        if any(f"layers.{i}." in n for i in range(CFG.freeze_layers)):
            p.requires_grad_(False)
    return m

def collate_fn(batch):
    keys, out = batch[0].keys(), {}
    for k in keys:
        out[k] = [b[k] for b in batch] if k in ["mask_labels","class_labels"] \
                 else torch.stack([b[k] for b in batch])
    for ml, cl in zip(out["mask_labels"], out["class_labels"]):
        assert len(ml) == len(cl), "mask/label length mismatch in batch"
    return out

def run_epoch(model, loader, optim, sched, acc, train=True):
    model.train() if train else model.eval()
    losses = []
    pbar = tqdm(loader, disable=not acc.is_local_main_process)

    for batch in pbar:
        ctx = acc.accumulate(model) if train else torch.no_grad()
        with ctx:
            out  = model(**batch)
            loss = out.loss

            if train:
                acc.backward(loss)
                optim.step()
                optim.zero_grad()
                if sched: sched.step()

        losses.append(loss.detach().cpu())
        pbar.set_description(
            f"{'train' if train else 'val'} {torch.mean(torch.stack(losses)).item():.4f}"
        )

    return torch.mean(torch.stack(losses)).item()

def main():
    acc = Accelerator(
        gradient_accumulation_steps=CFG.gradient_accum,
        log_with="tensorboard",
        project_dir=CFG.logdir,
        mixed_precision="bf16",
    )

    model = load_model(CFG.model_ckpt)
    optim = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    class_weights = [1.0]
    model.config.class_weights = class_weights
    
    start_epoch = 1
    if args.resume:
        acc.load_state(args.resume)
        
        model = load_model(args.resume)
        optim = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        
        if hasattr(acc.state, "extra_state_attributes"):
            start_epoch = acc.state.extra_state_attributes.get("epoch", 1) + 1

    writer = SummaryWriter(CFG.logdir) if acc.is_local_main_process else None

    proc  = Mask2FormerImageProcessor.from_pretrained(CFG.model_ckpt, ignore_index=0)
    ds    = RockCocoDataset(CFG.coco_json, CFG.frames_dir, build_transforms(), proc)
    n_val = int(len(ds)*0.2)
    train_ds, val_ds = torch.utils.data.random_split(ds,[len(ds)-n_val,n_val])

    dl_tr = DataLoader(train_ds, CFG.batch_size, True , num_workers=6, collate_fn=collate_fn, pin_memory=True)

    steps = (len(dl_tr)//acc.num_processes//CFG.gradient_accum+1)*CFG.num_epochs
    sched = get_linear_schedule_with_warmup(
        optim, 
        num_warmup_steps=CFG.warmup_steps,
        num_training_steps=steps
    )

    model, optim, dl_tr, sched = acc.prepare(
        model, optim, dl_tr, sched
    )

    best = 1e10
    for epoch in range(start_epoch, CFG.num_epochs+1):
        tr_loss = run_epoch(model, dl_tr, optim, sched, acc, True)

        if epoch % CFG.eval_interval == 0:
            if acc.is_local_main_process:
                val_dl_full = DataLoader(val_ds, CFG.batch_size, shuffle=False, 
                                        num_workers=6, collate_fn=collate_fn, 
                                        pin_memory=True)
                model.eval()
                metric = MeanAveragePrecision(iou_type='segm', class_metrics=True).to(acc.device)
                val_losses = []
                pbar_val = tqdm(val_dl_full, desc="Validation", disable=not acc.is_local_main_process)

                for batch in pbar_val:
                    batch_on_device = {}
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            batch_on_device[k] = v.to(acc.device)
                        elif isinstance(v, list) and all(isinstance(item, torch.Tensor) for item in v):
                            batch_on_device[k] = [item.to(acc.device) for item in v]
                        else:
                            batch_on_device[k] = v
                    
                    with torch.no_grad():
                        outputs = model(**batch_on_device)
                        loss = outputs.loss
                        val_losses.append(loss.item())

                        target_sizes = [(CFG.img_size, CFG.img_size)] * len(batch_on_device["pixel_values"])
                        results = proc.post_process_instance_segmentation(
                            outputs, target_sizes=target_sizes, threshold=0.5
                        )

                        preds = []
                        for res in results:
                            seg = res["segmentation"]
                            seg_info = res["segments_info"]
                            masks, scores, labels = [], [], []
                            for info in seg_info:
                                mask = (seg == info['id'])
                                masks.append(mask)
                                scores.append(info['score'])
                                labels.append(info['label_id'])
                            
                            masks = torch.stack(masks) if masks else torch.zeros(
                                (0, seg.shape[0], seg.shape[1]), dtype=torch.bool
                            )
                            preds.append({
                                "masks": masks,
                                "scores": torch.tensor(scores, device=masks.device),
                                "labels": torch.tensor(labels, device=masks.device, dtype=torch.int),
                            })

                        targets = []
                        for i in range(len(batch_on_device["mask_labels"])):
                            masks = batch_on_device["mask_labels"][i] > 0.5
                            targets.append({
                                "masks": masks,
                                "labels": batch_on_device["class_labels"][i],
                            })

                        metric.update(preds, targets)

                val_loss = torch.mean(torch.tensor(val_losses)).item()

                metric = metric.cpu()
                map_dict = metric.compute()

                if writer:
                    writer.add_scalar("loss/val", val_loss, epoch)
                    for key, value in map_dict.items():
                        if torch.is_tensor(value) and value.numel() == 1:
                            writer.add_scalar(f"metrics/{key}", value.item(), epoch)
                        elif key in ["map_per_class", "mar_100_per_class"]:
                            for i, v in enumerate(value):
                                writer.add_scalar(f"metrics/{key}/class_{i}", v.item(), epoch)

                print(f"Validation loss: {val_loss:.4f}")
                print(f"mAP: {map_dict['map'].item():.4f}, AP50: {map_dict['map_50'].item():.4f}")

                if val_loss < best:
                    best = val_loss
                    p = CFG.output_dir / f"best_ep{epoch}"
                    acc.unwrap_model(model).save_pretrained(p, safe_serialization=True)
                    acc.state.extra_state_attributes = {"epoch": epoch, "best": best}
                    acc.save_state(CFG.output_dir / f"best_ep{epoch}", models=[])
                    print("✅ saved", p)

            acc.wait_for_everyone()

        if writer and acc.is_local_main_process:
            writer.add_scalar("loss/train", tr_loss, epoch)

if __name__ == "__main__":
    main()