import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys
from torch.utils.data import Dataset, DataLoader

# 1. Add the path to the 'TransUNet' folder (or its parent)
# Replace this string with the ACTUAL path on your C: drive
# repo_path = 'D:\Documents\LiverSeg\TransUNet' 

# if repo_path not in sys.path:
#     sys.path.append(repo_path)

# The official repo usually structures it like this:
from networks.vit_seg_modeling import VisionTransformer as TransUNet
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg  # This was the broken PyPI package
import scipy.ndimage as ndi

# ----------------- SETTINGS -----------------
DATA_ROOT = "data_patients"   # folder with Patient 1 ... Patient 20

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
print("Using device:", DEVICE)

if DEVICE.type == "cuda":
    BATCH_SIZE = 4
else:
    BATCH_SIZE = 2   # MPS is less memory capable

EPOCHS = 200             # will increase later if i have time
LR = 3e-5


os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==========================================
# Your Loader Function 
# ==========================================

def get_loaders(batch_size=4):
    all_patients = [p for p in os.listdir("data_patients") 
                    if p.lower().startswith("patient")]
    
    # Numeric Sort (Patient 2 before Patient 10)
    all_patients = sorted(all_patients, key=lambda x: int(re.search(r'\d+', x).group()))

    # Train/Val Split
    train_patients = all_patients[:13]
    val_patients   = all_patients[13:16]

    print(f"Training with {len(train_patients)} patients.")
    print(f"Validating with {len(val_patients)} patients.")

    # Create Datasets
    # is_train=True triggers the "Heavy" augmentation pipeline
    train_ds = LiverPatientDataset("data_patients", train_patients, is_train=True)
    
    # is_train=False triggers the "Clean" pipeline
    val_ds   = LiverPatientDataset("data_patients", val_patients, is_train=False)
    
    # IMPORTANT SAFETY CHECK
    if len(val_ds) == 0:
        raise RuntimeError(
            "❌ Validation dataset is empty after liver filtering. "
            "Change validation patients."
        )
    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, 
                              num_workers=1, pin_memory=True)

    return train_loader, val_loader

def patient_sort_key(name: str):
    """Sort 'Patient 1', 'Patient 2', ..., 'Patient 10' numerically."""
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 999

def pad_resize_pil(img, size=224, is_mask=False):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    img = img.resize(
        (new_w, new_h),
        Image.NEAREST if is_mask else Image.BILINEAR
    )

    new_img = Image.new("L", (size, size))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))

    return new_img

def clean_mask_cc_np(mask_np):
    """
    Keep only the largest connected component in the mask.
    Safe for no-liver slices.
    """
    mask_np = (mask_np > 0).astype(np.uint8)

    labeled, num = ndi.label(mask_np)
    if num == 0:
        return mask_np  # no liver case

    sizes = ndi.sum(mask_np, labeled, range(1, num + 1))
    largest_label = sizes.argmax() + 1
    cleaned = (labeled == largest_label).astype(np.uint8)

    return cleaned

# ------------- PATIENT DATASET --------------
class LiverPatientDataset(Dataset):
    def __init__(self, root_dir, patient_list, is_train=True):
        self.samples = []
        self.is_train = is_train

        print(f"Scanning {len(patient_list)} patients for valid liver slices...")

        def slice_id(name):
            nums = re.findall(r"\d+", name)
            return nums[-1] if nums else None

        for patient in patient_list:
            patient_dir = os.path.join(root_dir, patient)
            ct_dir = os.path.join(patient_dir, "CT")
            mask_dir = os.path.join(patient_dir, "Liver_cc_clean")

            if not os.path.isdir(ct_dir) or not os.path.isdir(mask_dir):
                continue

            img_files = [
                f for f in os.listdir(ct_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            mask_files = [
                f for f in os.listdir(mask_dir)
                if f.lower().endswith(".png")
            ]

            img_map = {}
            for f in img_files:
                sid = slice_id(f)
                if sid is not None and sid not in img_map:
                    img_map[sid] = f

            mask_map = {}
            for f in mask_files:
                sid = slice_id(f)
                if sid is not None and sid not in mask_map:
                    mask_map[sid] = f

            common_ids = sorted(set(img_map.keys()) & set(mask_map.keys()))

            if len(common_ids) == 0:
                print(f"⚠️ No matching slices for {patient}")

            for sid in common_ids:
                img_path = os.path.join(ct_dir, img_map[sid])
                mask_path = os.path.join(mask_dir, mask_map[sid])

                with Image.open(mask_path).convert("L") as m:
                    mask_arr = np.array(m)

                if mask_arr.sum() == 0:
                    continue  # skip no-liver slices

                self.samples.append((img_path, mask_path))

        print(f"  -> Found {len(self.samples)} valid slices containing liver.")

    def __len__(self):
        return len(self.samples)

    def transform(self, image, mask):
        # Resize
        # target_size =(512, 512)
        #Pad-resize instead of distortion resize
        image = pad_resize_pil(image, size=224, is_mask=False)
        mask  = pad_resize_pil(mask, size=224, is_mask=True)

        # Augmentation (training only)
        if self.is_train:
           if random.random() > 0.5:
              image = TF.hflip(image)
              mask = TF.hflip(mask)

           if random.random() > 0.5:
              angle = random.choice([-15, 15])
              image = TF.rotate(image, angle)
              mask = TF.rotate(mask, angle)


        return image, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Open as PIL
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Apply transforms (resize + augment)
        image, mask = self.transform(image, mask)

        # To Tensor and Normalize
        image = TF.to_tensor(image)  # scales [0, 255] -> [0.0, 1.0]
        
        mask_arr = np.array(mask, dtype=np.uint8)

        # ---- ADD CC CLEANING HERE ----
        mask_arr = clean_mask_cc_np(mask_arr)

        mask = torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(0)

        return image, mask
    
# -------------- LOSS & METRIC ---------------
bce_loss = nn.BCEWithLogitsLoss()

def dice_coef(pred, target, eps=1e-6):
    # convert logits -> probabilities
    pred_prob = torch.sigmoid(pred)
    pred_bin = (pred_prob > 0.5).float()
    target_bin = (target > 0.5).float()

    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()

def boundary_loss(pred, target):
    """
    Boundary loss using distance transform
    pred: logits
    target: binary mask
    """
    pred_prob = torch.sigmoid(pred)

    # Convert target to numpy for distance transform
    target_np = target.detach().cpu().numpy()

    dist_maps = []
    for b in range(target_np.shape[0]):
        gt = target_np[b, 0]
        posmask = gt.astype(bool)
        negmask = ~posmask

        dist_pos = ndi.distance_transform_edt(posmask)
        dist_neg = ndi.distance_transform_edt(negmask)

        signed_dist = dist_neg - dist_pos
        dist_maps.append(signed_dist)

    dist_maps = torch.tensor(
        np.stack(dist_maps),
        dtype=torch.float32,
        device=pred.device
    ).unsqueeze(1)

    return torch.mean(pred_prob * dist_maps)

def iou_score(pred, target, eps=1e-6):
    # logits -> probabilities
    pred_prob = torch.sigmoid(pred)
    pred_bin = (pred_prob > 0.5).float()
    target_bin = (target > 0.5).float()

    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()

def dice_loss(pred_prob, target, eps=1e-6):
    pred_prob = pred_prob.contiguous()
    target = target.contiguous()

    inter = (pred_prob * target).sum(dim=(1,2,3))
    union = pred_prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (union + eps)

    return 1 - dice.mean()

def combined_loss(pred, target):
    # Dice + BCE (your current setup)
    pred_prob = torch.sigmoid(pred)

    dice = dice_loss(pred_prob, target)
    bce  = bce_loss(pred, target)

    # Strong Dice supervision
    loss = 0.7 * dice + 0.3 * bce

    # Boundary loss
    loss += 0.03 * boundary_loss(pred, target)

    # No-liver penalty (keep it)
    liver_present = (target.sum(dim=(1,2,3)) > 0).float()
    no_liver_penalty = ((pred_prob > 0.3).float().sum(dim=(1,2,3)) > 0).float()
    loss += 0.5 * (no_liver_penalty * (1 - liver_present)).mean()

    # FINAL LOSS
    return loss


def validate(model, loader):
    model.eval()
    losses = []
    dices = []
    ious = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = combined_loss(preds, masks)

            # metrics
            losses.append(loss.item())
            dices.append(dice_coef(preds, masks))
            ious.append(iou_score(preds, masks))

    # CRITICAL FIX: empty validation guard
    if len(losses) == 0:
        print("⚠️ Warning: Validation set has no liver slices.")
        return float("inf"), 0.0, 0.0
            

    return (
        float(np.mean(losses)),
        float(np.mean(dices)),
        float(np.mean(ious)),
    )


def post_process(mask):
    mask = mask.astype(np.uint8)

    labeled, num = ndi.label(mask)
    if num == 0:
        return mask

    sizes = ndi.sum(mask, labeled, range(1, num + 1))
    largest = sizes.argmax() + 1
    return (labeled == largest).astype(np.uint8)

class TrainingLogger:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.val_dices = []
        self.val_ious = []
        
    def set_history(self, epochs, t_loss, v_loss, v_dice, v_iou):
        """Restores history from a checkpoint"""
        self.epochs = epochs
        self.train_losses = t_loss
        self.val_losses = v_loss
        self.val_dices = v_dice
        self.val_ious = v_iou

    def update(self, epoch, train_loss, val_loss, val_dice, val_iou):
        # 1. Store the new data
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_dices.append(val_dice)
        self.val_ious.append(val_iou)
        
        # 2. Create and Save Loss Plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.train_losses, label='Train Loss')
        plt.plot(self.epochs, self.val_losses, label='Val Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('outputs/loss_curve.png') # Changed path to outputs/
        plt.close() 
        
        # 3. Create and Save Metrics Plot
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.val_dices, label='Val Dice')
        plt.plot(self.epochs, self.val_ious, label='Val IoU')
        plt.title('Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig('outputs/metrics_curve.png') # Changed path to outputs/
        plt.close()

def save_sample_results(model, dataset, device):
    model.eval()
    os.makedirs("outputs", exist_ok=True)

    # pick 4 samples (last one likely no-liver)
    indices = [
        5,
        len(dataset)//3,
        len(dataset)//2,
        len(dataset)-1
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    with torch.no_grad():
        for r, idx in enumerate(indices):
            image, mask = dataset[idx]

            image_t = image.unsqueeze(0).to(device)
            pred = model(image_t)

            pred = torch.sigmoid(pred)[0, 0].cpu().numpy()
            pred = (pred > 0.5).astype(np.uint8)
            pred = post_process(pred)

            ct = image[0].cpu().numpy()
            gt = mask[0].cpu().numpy()

            axes[r, 0].imshow(ct, cmap="gray")
            axes[r, 0].set_title(f"Sample {idx}")
            axes[r, 0].axis("off")

            axes[r, 1].imshow(gt, cmap="gray")
            axes[r, 1].set_title("Ground Truth")
            axes[r, 1].axis("off")

            axes[r, 2].imshow(pred, cmap="gray")
            axes[r, 2].set_title("Prediction")
            axes[r, 2].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/sample_results.png", dpi=300)
    plt.close()

def train():
    # Loaders automatically using the BATCH_SIZE defined above
    train_loader, val_loader = get_loaders(batch_size=BATCH_SIZE)

    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1  # Binary segmentation
    config_vit.n_skip = 3     # Standard skipping for TransUNet

    # If you use 224x224
    config_vit.patches.grid = (int(224 / 16), int(224 / 16))

    model = TransUNet(
        config=config_vit, 
        img_size=224, 
        num_classes=1
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10
    )
    
    logger = TrainingLogger()

    # --- CHECKPOINT RESUMING LOGIC ---
    checkpoint_path = "checkpoints/unet_last_checkpoint.pth"
    start_epoch = 1
    best_val_loss = float("inf")

    if os.path.exists(checkpoint_path):
        print(f"--> Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # 1. Load Model & Optimizer weights
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 2. Restore Training State
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        
        # 3. Restore Logger History (so plots are continuous)
        if 'history' in checkpoint:
            hist = checkpoint['history']
            logger.set_history(
                hist['epochs'], hist['train_losses'], hist['val_losses'], 
                hist['val_dices'], hist['val_ious']
            )
        
        print(f"--> Resumed training from epoch {start_epoch}")
    else:
        print("--> No checkpoint found. Starting from scratch.")

    # ---------------------------------

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running = 0.0

        for batch_idx, (imgs, masks) in enumerate(train_loader):
            if batch_idx % 50 == 0:
               print(f"  Epoch {epoch}, batch {batch_idx}/{len(train_loader)}")

            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = combined_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        val_loss, val_dice, val_iou = validate(model, val_loader)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # ---- update plots ----
        logger.update(epoch, train_loss, val_loss, val_dice, val_iou)

        # ---- save last checkpoint ----
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'history': {
               'epochs': logger.epochs,
               'train_losses': logger.train_losses,
               'val_losses': logger.val_losses,
               'val_dices': logger.val_dices,
               'val_ious': logger.val_ious
            }
        }, checkpoint_path)

        # ---- save best model ----
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), "checkpoints/unet_best_patients.pth")
          print("  -> Saved BEST model")

        # QUALITATIVE RESULTS (THIS IS THE KEY FIX)
        if epoch > 0 and (epoch % 10 == 0 or epoch == EPOCHS):
          print("Saving qualitative sample results...")
          save_sample_results(model, val_loader.dataset, DEVICE)
          
if __name__ == "__main__":
    train()

