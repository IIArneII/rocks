"""
LoRA fine-tuning for Segment-Anything (SAM) using HuggingFace PEFT.

> python lora_sam_train.py \
        --json /data/coco/pairs.json \
        --sam_checkpoint /workspace/checkpoints/sam_vit_h.pth \
        --batch_size 4 --lr 1e-4 --epochs 20 \
        --lora_r 8 --lora_alpha 32 --out_dir /output/lora-sam-vit-h
"""

from __future__ import annotations
import argparse, json, os, random, numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pycocotools import mask as coco_mask

from segment_anything import sam_model_registry
from peft import get_peft_model, LoraConfig, TaskType


@dataclass
class CFG:
    json: str
    sam_checkpoint: str
    model_variant: str = "vit_b"
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 20
    num_workers: int = 4
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    device: str = "cuda"
    out_dir: str = "./lora-sam"
    seed: int = 1337
    save_every: int = 5

IM_MEAN = [123.675/255, 116.28/255, 103.53/255]
IM_STD  = [58.395/255, 57.12/255, 57.375/255]

class SamPairDataset(Dataset):
    def __init__(self, json_path: str, image_size: int = 768):
        self.items: List[dict] = json.load(open(json_path))
        self.root = Path(json_path).parent.parent

        self.im_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IM_MEAN, std=IM_STD),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST,
                              antialias=True),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.bool),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]

        img_path = self.root / "raw_frames" / rec["image"]
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size

        rle = rec["mask_rle"]
        rle = rle.copy()
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode()
        mask = coco_mask.decode(rle)
        mask = Image.fromarray(mask * 255)

        img_t  = self.im_tf(img)
        mask_t = self.mask_tf(mask)

        sx, sy = img_t.shape[-1] / w0, img_t.shape[-2] / h0
        px, py = rec["point"]
        point  = torch.tensor([px * sx, py * sy], dtype=torch.float32)

        return dict(pixel_values=img_t,
                    gt_mask=mask_t,
                    point=point)

def dice_loss(logits, target, eps: float = 1e-6):
    probs = logits.sigmoid()
    num = 2 * (probs * target).sum(dim=(-2, -1)) + eps
    den = probs.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + eps
    return 1 - num / den

def seg_loss(logits, target):
    return (F.binary_cross_entropy_with_logits(logits, target.float()) +
            dice_loss(logits, target.float()).mean())

def seed_everything(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def build_model(cfg: CFG):
    sam = sam_model_registry[cfg.model_variant](checkpoint=cfg.sam_checkpoint).to(cfg.device)

    target_modules = [name for name, m in sam.named_modules()
                      if isinstance(m, nn.Linear)
                      and (name.startswith("prompt_encoder") or name.startswith("mask_decoder"))]

    lora_cfg = LoraConfig(
        r              = cfg.lora_r,
        lora_alpha     = cfg.lora_alpha,
        target_modules = target_modules,
        lora_dropout   = cfg.lora_dropout,
        bias           = "none",
        task_type      = TaskType.FEATURE_EXTRACTION,
    )
    sam = get_peft_model(sam, lora_cfg)

    for n, p in sam.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)

    sam.print_trainable_parameters()
    return sam

def train(cfg: CFG):
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    ds = SamPairDataset(cfg.json)
    dl = DataLoader(ds, cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(cfg)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay)

    iters_per_epoch = len(dl)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs * iters_per_epoch)

    scaler = torch.cuda.amp.GradScaler()
    model.train()

    global_step = 0
    for epoch in range(cfg.epochs):
        for batch in dl:
            imgs   = batch["pixel_values"].to(cfg.device, non_blocking=True)
            masks  = batch["gt_mask"].to(cfg.device, non_blocking=True)
            points = batch["point"].to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    img_emb = model.image_encoder(imgs)

                point_coords = points.unsqueeze(1)
                point_labels = torch.ones((points.size(0), 1),
                                          dtype=torch.int64, device=cfg.device)
                
                B = imgs.size(0)

                target_hw = img_emb.shape[-2:]

                image_pe = model.prompt_encoder.get_dense_pe().to(cfg.device)
                image_pe = F.interpolate(
                    image_pe, size=target_hw, mode='bilinear', align_corners=False
                ).expand(B, -1, -1, -1)

                sparse_pe, dense_pe = model.prompt_encoder(
                    points=(point_coords, point_labels), boxes=None, masks=None
                )

                if dense_pe.dim() == 3:
                    dense_pe = dense_pe.unsqueeze(0)
                if dense_pe.shape[-2:] != target_hw:
                    dense_pe = F.interpolate(
                        dense_pe, size=target_hw, mode='bilinear', align_corners=False
                )
                dense_pe = dense_pe.expand(B, -1, -1, -1)

                masks_pred, _, _ = model.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_pe,
                    dense_prompt_embeddings=dense_pe,
                    multimask_output=False)

                if masks_pred.shape[-1] != imgs.shape[-1]:
                    masks_pred = F.interpolate(
                        masks_pred, size=imgs.shape[-2:],
                        mode="bilinear", align_corners=False)

                loss = seg_loss(masks_pred, masks)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            lr_sched.step()
            global_step += 1

            if global_step % 10 == 0:
                print(f"epoch {epoch:02d}  step {global_step:06d}  "
                      f"lr {lr_sched.get_last_lr()[0]:.3e}  loss {loss.item():.4f}")

        if (epoch + 1) % cfg.save_every == 0 or epoch + 1 == cfg.epochs:
            save_dir = Path(cfg.out_dir) / f"epoch{epoch:02d}"
            model.save_pretrained(str(save_dir))
            print(f"✔ saved LoRA adapter to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--out_dir", default="./lora-sam")
    parser.add_argument("--model_variant", default="vit_b")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()
    train(CFG(**vars(args)))