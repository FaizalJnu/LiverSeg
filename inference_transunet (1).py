import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import scipy.ndimage as ndi

from networks.vit_seg_modeling import VisionTransformer as TransUNet
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

# ================== SETTINGS ==================
DATA_ROOT = "data_patients"
CHECKPOINT = "checkpoints/unet_best_patients.pth"
OUT_DIR = "outputs/inference_results"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
print("Using device:", DEVICE)

IMG_SIZE = 224
LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.65

os.makedirs(OUT_DIR, exist_ok=True)
# ==============================================


# ============== UTILITY FUNCTIONS ==============

def patient_sort_key(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 999


def numeric_sort_key(filename):
    nums = re.findall(r"\d+", filename)
    return int(nums[0]) if nums else 0


def pad_resize_pil(img, size=224, is_mask=False):
    """Aspect-ratio preserving resize + padding (same as training)."""
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


def post_process(mask):
    """Keep only the largest connected component."""
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask

    sizes = ndi.sum(mask, labeled, range(1, num + 1))
    largest = sizes.argmax() + 1
    return (labeled == largest).astype(np.uint8)


def load_slice(img_path, mask_path=None):
    """Load and preprocess one CT slice (+ GT if available)."""
    img = Image.open(img_path).convert("L")
    img = pad_resize_pil(img, IMG_SIZE, is_mask=False)
    img_np = np.array(img, dtype=np.float32) / 255.0

    if mask_path is not None and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask = pad_resize_pil(mask, IMG_SIZE, is_mask=True)
        
        mask_np = (np.array(mask) > 128).astype(np.uint8) 
    else:
        mask_np = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    img_t = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    return img_t, mask_np, img_np


# ============== INFERENCE LOGIC =================

def infer_patient(patient_name, model):
    print(f"\nRunning inference for {patient_name}")

    patient_dir = os.path.join(DATA_ROOT, patient_name)
    img_dir = os.path.join(patient_dir, "CT")
    mask_dir = os.path.join(patient_dir, "Liver")

    if not os.path.isdir(img_dir):
        print("  No CT folder found, skipping.")
        return

    img_files = sorted(os.listdir(img_dir), key=numeric_sort_key)
    saved = 0

    for fname in img_files:
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        img_t, gt_mask, img_np = load_slice(img_path, mask_path)
        img_t = img_t.to(DEVICE)

        with torch.no_grad():
            logits = model(img_t)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # ========== DUAL THRESHOLD + CONNECTED COMPONENT ==========
        pred_low  = (prob > LOW_THRESHOLD).astype(np.uint8)
        pred_high = (prob > HIGH_THRESHOLD).astype(np.uint8)

        labeled, _ = ndi.label(pred_low)
        pred = np.zeros_like(pred_low)

        for lab in np.unique(labeled):
            if lab == 0:
                continue
            if np.any(pred_high[labeled == lab]):
                pred[labeled == lab] = 1

        #  fill holes inside liver
        pred = ndi.binary_fill_holes(pred)

        #  smooth boundaries (optional but recommended)
        pred = ndi.binary_closing(pred, structure=np.ones((3, 3)))

        #  remove tiny false positives
        pred = pred.astype(np.uint8)
        min_area = 0.005 * (IMG_SIZE * IMG_SIZE)
        if pred.sum() < min_area:
            pred[:] = 0

        # =========================================================

        # -------- Save visualization --------
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("CT")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        out_path = os.path.join(
            OUT_DIR, f"{patient_name}_{base}.png"
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        saved += 1

    print(f"  Saved {saved} slices for {patient_name}")


# ===================== MAIN =====================

if __name__ == "__main__":

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError("Checkpoint not found.")

    config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.patches.grid = (IMG_SIZE // 16, IMG_SIZE // 16)

    model = TransUNet(
        config=config_vit,
        img_size=IMG_SIZE,
        num_classes=1
    ).to(DEVICE)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    all_patients = [
        p for p in os.listdir(DATA_ROOT)
        if p.lower().startswith("patient")
    ]
    all_patients = sorted(all_patients, key=patient_sort_key)

    # Example: last 4 patients as test
    test_patients = all_patients[16:20]

    for p in test_patients:
        infer_patient(p, model)