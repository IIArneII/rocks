"""
Fine-tune SAM with LoRA adapters on a tiny COCO-style dataset.
Usage:
  python train_sam_peft.py \
         --pairs data/pairs.json \
         --images_dir data/images \
         --out_dir outputs/run1 \
         --epochs 60
"""
import argparse, json, os, random, pathlib
import numpy as np
from tqdm.auto import tqdm

import torch, torch.nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
import cv2

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from segment_anything import sam_model_registry

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, required=True)
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_type", default="vit_h")
    ap.add_argument("--sam_ckpt",
                    default="checkpoints/sam_vit_h.pth")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class RockPairDS(Dataset):
    def __init__(self, pairs_json, images_dir, train=True):
        self.samples = json.load(open(pairs_json))
        self.images_dir = images_dir
        self.train = train
        self.augs = A.Compose([
            A.LongestMaxSize(max_size=1024, p=1.0),
            A.PadIfNeeded(1024, 1024, border_mode=cv2.BORDER_CONSTANT,
                        value=0, mask_value=0, p=1.0),
            A.RandomCrop(1024, 1024, p=.9),
            A.HorizontalFlip(p=.5),
            A.VerticalFlip(p=.25),
            A.ShiftScaleRotate(shift_limit=.1, scale_limit=.2,
                            rotate_limit=25, border_mode=cv2.BORDER_REFLECT, p=.9),
            A.RandomBrightnessContrast(.2, .2, p=.8),
            A.ISONoise(p=.3),
                ], additional_targets={"mask": "mask"}) if train else None

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.images_dir, s["image"])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        
        mask = coco_rle_to_mask(s["mask_rle"]).astype(np.float32)

        if self.augs:
            r = self.augs(image=img, mask=mask)
            img, mask = r["image"], r["mask"]

        point = torch.tensor(s["point"], dtype=torch.float32)
        h, w = img.shape[:2]
        point_norm = point / torch.tensor([w, h], dtype=torch.float32)

        img_t = self.to_tensor(img)
        mask_t = torch.from_numpy(mask[None])

        return dict(
            image=img_t,
            prompt_point=point_norm,
            gt_mask=mask_t
        )

# ---------- Utils ----------
from pycocotools import mask as mask_utils
def coco_rle_to_mask(rle):
    if isinstance(rle["counts"], list):
        rle = mask_utils.frPyObjects([rle], rle["size"][0], rle["size"][1])[0]
    return mask_utils.decode(rle)

def dice_loss(inp, target, eps=1e-6):
    num = 2 * (inp * target).sum(dim=(-1,-2))
    den = inp.sum(dim=(-1,-2)) + target.sum(dim=(-1,-2)) + eps
    return 1 - num/den

def main():
    args = parse_args()
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    pairs = json.load(open(args.pairs))
    random.shuffle(pairs)
    split = int(len(pairs)*0.8)
    
    tmp_dir = os.path.join(args.out_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    json.dump(pairs[:split], open(os.path.join(tmp_dir, "train.json"), "w"))
    json.dump(pairs[split:], open(os.path.join(tmp_dir, "val.json"), "w"))

    ds_train = RockPairDS(os.path.join(tmp_dir, "train.json"), args.images_dir, train=True)
    ds_val   = RockPairDS(os.path.join(tmp_dir, "val.json"), args.images_dir, train=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                          num_workers=4, pin_memory=True)

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_ckpt)
    for p in sam.image_encoder.parameters():
        p.requires_grad_(False)

    trainable = list(sam.prompt_encoder.parameters()) + \
                list(sam.mask_decoder.parameters())

    peft_cfg = LoraConfig(r=args.rank, lora_alpha=args.rank*2,
                          target_modules=["q_proj", "v_proj", "key", "value",
                                          "linear_q", "linear_k", "linear_v"],
                          bias="none", inference_mode=False)
    sam = get_peft_model(sam, peft_cfg, adapter_name="rock")
    sam.print_trainable_parameters()

    opt = torch.optim.AdamW(filter(lambda p:p.requires_grad, sam.parameters()),
                            lr=args.lr, weight_decay=1e-2)
    total_steps = len(dl_train)*args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr,
        total_steps=total_steps, pct_start=.1, final_div_factor=25
    )

    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16")
    sam, opt, dl_train, dl_val, sched = accelerator.prepare(
        sam, opt, dl_train, dl_val, sched)

    best_iou = 0.
    for epoch in range(1, args.epochs+1):
        sam.train()
        pbar = tqdm(dl_train, disable=not accelerator.is_local_main_process)
        for batch in pbar:
            with accelerator.autocast():
                pred_masks = forward_sam(sam, batch)
                bce = F.binary_cross_entropy_with_logits(
                    pred_masks, batch["gt_mask"])
                dce = dice_loss(torch.sigmoid(pred_masks), batch["gt_mask"])
                loss = .5*(bce+dce)
            accelerator.backward(loss)
            opt.step(); opt.zero_grad(); sched.step()
            pbar.set_description(f"epoch {epoch} loss {loss.item():.3f}")

        sam.eval()
        ious = []
        with torch.inference_mode(), accelerator.autocast():
            for batch in dl_val:
                pm = forward_sam(sam, batch)
                preds = (torch.sigmoid(pm)>0.5).float()
                inter = (preds*batch["gt_mask"]).sum((-1,-2))
                union = preds.sum((-1,-2))+batch["gt_mask"].sum((-1,-2))-inter+1e-6
                ious.extend((inter/union).cpu().numpy())
        miou = float(np.mean(ious))
        accelerator.print(f"Epoch {epoch} mIoU={miou:.3f}")
        if miou > best_iou:
            best_iou = miou
            accelerator.print("Saving best …")
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(sam)
            torch.save(unwrapped.state_dict(),
                       os.path.join(args.out_dir, "sam_peft_best.pth"))

def forward_sam(sam, batch):
    B, _, H, W = batch["image"].shape
    coords = batch["prompt_point"].clone()
    coords[:,0] *= W; coords[:,1] *= H
    pc = coords[:,None,:]
    pc_label = torch.ones((B,1), device=coords.device)
    image_embeddings = sam.image_encoder(batch["image"])
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=(pc, pc_label), boxes=None, masks=None)
    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False)
    return F.interpolate(low_res_masks, size=(H,W), mode="bilinear",
                         align_corners=False)

if __name__ == "__main__":
    main()

# python sam_lora/train_sam_peft.py --pairs data/coco/pairs.json --images_dir data/raw_frames --out_dir outputs/run1