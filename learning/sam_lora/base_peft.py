#!/usr/bin/env python
# train_sam_lora_minimal.py
"""
Minimal reproducible script: fine-tune SAM with LoRA on a tiny COCO set.

Example
-------
python train_sam_lora_minimal.py \
       --pairs data/coco/pairs.json \
       --images_dir data/raw_frames \
       --out_dir  runs/debug1
"""

import argparse, json, os, random, pathlib, time
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

from pycocotools import mask as mask_utils
from peft import LoraConfig, get_peft_model
from segment_anything import sam_model_registry


def coco_rle_to_mask(rle):
    if isinstance(rle["counts"], list):
        rle = mask_utils.frPyObjects([rle], rle["size"][0], rle["size"][1])[0]
    return mask_utils.decode(rle)

def dice_loss(prob, target, eps=1e-6):
    num = 2 * (prob * target).sum((-1, -2))
    den = prob.sum((-1, -2)) + target.sum((-1, -2)) + eps
    return 1 - num / den


class RockPairs(Dataset):
    def __init__(self, pairs_json, images_dir):
        self.samples = json.load(open(pairs_json))
        self.images_dir = images_dir
        self.totensor = Compose([ToTensor(),
                                 Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(os.path.join(self.images_dir, s["image"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = coco_rle_to_mask(s["mask_rle"]).astype(np.float32)

        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        img_t  = self.totensor(img)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        ys, xs = np.where(mask > 0)
        r = random.randint(0, len(xs) - 1)
        point_norm = torch.tensor([xs[r] / 1024, ys[r] / 1024],
                                  dtype=torch.float32)

        return {"image": img_t,
                "gt_mask": mask_t,
                "prompt_point": point_norm}

def get_empty_dense(embed):
    """
    Returns a zero tensor with the same (B, C, H/4, W/4) shape that
    mask_decoder expects for `dense_prompt_embeddings`.
    """
    return torch.zeros_like(embed)

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # split pairs file 80/20
    pairs = json.load(open(args.pairs))
    random.shuffle(pairs)
    n_tr = int(0.8 * len(pairs))
    tmp_dir = pathlib.Path(args.out_dir) / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    json.dump(pairs[:n_tr], open(tmp_dir / "train.json", "w"))
    json.dump(pairs[n_tr:], open(tmp_dir / "val.json", "w"))

    dl_train = DataLoader(RockPairs(tmp_dir / "train.json", args.images_dir),
                          batch_size=args.batch, shuffle=True)
    dl_val   = DataLoader(RockPairs(tmp_dir / "val.json", args.images_dir),
                          batch_size=args.batch, shuffle=False)

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_ckpt)
    for p in sam.image_encoder.parameters():
        p.requires_grad_(False)

    lora_cfg = LoraConfig(r=args.rank,
                          lora_alpha=args.rank * 2,
                          target_modules=["q_proj", "v_proj", "key", "value",
                                          "linear_q", "linear_k", "linear_v"])
    sam = get_peft_model(sam, lora_cfg)
    sam.to(device)
    sam.print_trainable_parameters()

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam.parameters()),
                            lr=args.lr)

    best_miou = 0.0
    for epoch in range(1, args.epochs + 1):
        sam.train()
        pbar = tqdm(dl_train, desc=f"epoch {epoch}")
        for batch in pbar:
            imgs = batch["image"].to(device)
            gtm  = batch["gt_mask"].to(device)
            pts  = batch["prompt_point"].to(device)

            B, _, H, W = imgs.shape
            pix_pts = pts.clone()
            pix_pts[:, 0] *= W
            pix_pts[:, 1] *= H
            pix_pts = pix_pts[:, None, :]
            pt_labels = torch.ones((B, 1), device=device)

            with torch.no_grad():
                img_emb = sam.image_encoder(imgs)

            sp_emb, _ = sam.prompt_encoder(points=(pix_pts, pt_labels),
                                                 boxes=None, masks=None)
            
            B, C, H_e, W_e = img_emb.shape
            HW = H_e * W_e
            empty_dense = torch.zeros(B, HW, C, device=img_emb.device, dtype=img_emb.dtype)
            low_res, _ = sam.mask_decoder(image_embeddings=img_emb,
                                          image_pe=sam.prompt_encoder.get_dense_pe(),
                                          sparse_prompt_embeddings=sp_emb,
                                          dense_prompt_embeddings=empty_dense,
                                          multimask_output=False)
            pred = F.interpolate(low_res, size=(H, W), mode="bilinear",
                                 align_corners=False)

            bce  = F.binary_cross_entropy_with_logits(pred, gtm)
            dice = dice_loss(torch.sigmoid(pred), gtm)
            loss = 0.5 * (bce + dice)

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

        sam.eval()
        ious = []
        with torch.no_grad():
            for batch in dl_val:
                imgs = batch["image"].to(device)
                gtm  = batch["gt_mask"].to(device)
                pts  = batch["prompt_point"].to(device)

                B, _, H, W = imgs.shape
                pix_pts = pts.clone()
                pix_pts[:, 0] *= W
                pix_pts[:, 1] *= H
                pix_pts = pix_pts[:, None, :]
                pt_labels = torch.ones((B, 1), device=device)

                img_emb = sam.image_encoder(imgs)
                sp_emb, _ = sam.prompt_encoder(points=(pix_pts, pt_labels),
                                                     boxes=None, masks=None)
                B, C, H_e, W_e = img_emb.shape
                HW = H_e * W_e
                empty_dense = torch.zeros(B, HW, C, device=img_emb.device, dtype=img_emb.dtype)
                low_res, _ = sam.mask_decoder(image_embeddings=img_emb,
                                              image_pe=sam.prompt_encoder.get_dense_pe(),
                                              sparse_prompt_embeddings=sp_emb,
                                              dense_prompt_embeddings=empty_dense,
                                              multimask_output=False)
                pred = (F.interpolate(low_res, size=(H, W),
                                      mode="bilinear", align_corners=False)
                        .sigmoid() > 0.5).float()

                inter = (pred * gtm).sum((-1, -2))
                union = pred.sum((-1, -2)) + gtm.sum((-1, -2)) - inter + 1e-6
                ious.extend((inter / union).cpu().numpy())

        miou = float(np.mean(ious))
        print(f"Epoch {epoch}  mIoU={miou:.3f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(sam.state_dict(),
                       pathlib.Path(args.out_dir) / "sam_lora_best.pth")
            print("  ✓ new best model saved")

    print("Done. Best mIoU:", best_miou)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs",       required=True)
    p.add_argument("--images_dir",  required=True)
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--sam_ckpt",    default="checkpoints/sam_vit_h.pth")
    p.add_argument("--model_type",  default="vit_h")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch",  type=int, default=2)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--rank",   type=int,   default=16)
    p.add_argument("--seed",   type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)