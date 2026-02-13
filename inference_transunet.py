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

IMG_SIZE = 512


os.makedirs(OUT_DIR, exist_ok=True)
# ==============================================

# ================= SAMPLE SLICE CONFIG =================
PATIENT_SLICE_MAP = {
    "Patient 17": 40,
    "Patient 18": 67,
    "Patient 19": 58,
    "Patient 20": 153
}
# ======================================================

# ============== UTILITY FUNCTIONS ==============

def patient_sort_key(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 999


def numeric_sort_key(filename):
    nums = re.findall(r"\d+", filename)
    return int(nums[0]) if nums else 0


def load_slice(img_path, mask_path=None):
    img = Image.open(img_path).convert("L")
    img_np = np.array(img, dtype=np.float32) / 255.0  # already 512x512

    if mask_path is not None and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask_np = (np.array(mask) > 128).astype(np.uint8)
    else:
        mask_np = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    img_t = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)  # (1,1,512,512)
    return img_t, mask_np, img_np


def slice_id(name):
    nums = re.findall(r"\d+", name)
    if not nums:
        return None
    return str(int(nums[-1]))   # normalize 040 → "40"

# ============== INFERENCE LOGIC =================

def infer_patient(patient_name, model):
    print(f"\nRunning inference for {patient_name}")

    patient_dir = os.path.join(DATA_ROOT, patient_name)
    img_dir = os.path.join(patient_dir, "CT")
    mask_dir = os.path.join(patient_dir, "Liver")

    if not os.path.isdir(img_dir):
        print("  No CT folder found, skipping.")
        return

    # -------- BUILD MAPS (ONCE) --------
    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    mask_files = [f for f in os.listdir(mask_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]

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

    common_ids = sorted(
         set(img_map.keys()) & set(mask_map.keys()),
         key=lambda x: int(x)
    )

    if len(common_ids) == 0:
        print("  ❌ No matching slices found.")
        return

    saved = 0

    # -------- MAIN INFERENCE LOOP --------
    for sid in common_ids:
        img_path = os.path.join(img_dir, img_map[sid])
        mask_path = os.path.join(mask_dir, mask_map[sid])

        img_t, gt_mask, img_np = load_slice(img_path, mask_path)
        img_t = img_t.to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(model(img_t))[0, 0].cpu().numpy()

        pred = (prob > 0.5).astype(np.uint8)
        # ---- morphology ----
        pred = ndi.binary_fill_holes(pred)
        pred = ndi.binary_closing(pred, structure=np.ones((3, 3)))
    
        # -------- SAVE VIS --------
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
            OUT_DIR, f"{patient_name}_slice_{sid}.png"
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

        saved += 1

    print(f"  ✅ Saved {saved} slices for {patient_name}")

# ============== SAMPLE IMAGE  ==============
def save_sample_figure(model):
    model.eval()
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    for row, (patient_name, slice_idx) in enumerate(PATIENT_SLICE_MAP.items()):
        patient_dir = os.path.join(DATA_ROOT, patient_name)
        img_dir = os.path.join(patient_dir, "CT")
        mask_dir = os.path.join(patient_dir, "Liver")

        img_files = [f for f in os.listdir(img_dir)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        mask_files = [f for f in os.listdir(mask_dir)
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]

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

        
        common_ids = sorted(
             set(img_map.keys()) & set(mask_map.keys()),
             key=lambda x: int(x)
        )

        sid = str(slice_idx)   # slice number, NOT index

        if sid not in common_ids:
            print(f"⚠️ Slice {slice_idx} not found for {patient_name}")
            continue

        img_path = os.path.join(img_dir, img_map[sid])
        mask_path = os.path.join(mask_dir, mask_map[sid])

        img_t, gt_mask, img_np = load_slice(img_path, mask_path)
        img_t = img_t.to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(model(img_t))[0, 0].cpu().numpy()
        pred = (prob > 0.5).astype(np.uint8)
        # fill holes
        pred = ndi.binary_fill_holes(pred)

        # smooth boundaries
        pred = ndi.binary_closing(pred, structure=np.ones((2, 2)))

        # remove tiny false positives
        pred = pred.astype(np.uint8)
        
        # =================================================
        
        # ---- Plot ----
        axes[row, 0].imshow(img_np, cmap="gray")
        axes[row, 0].set_title(f"{patient_name} - CT")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_mask, cmap="gray")
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred, cmap="gray")
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/sample_results.png", dpi=300)
    plt.close()

    print("✅ Saved fixed sample figure: outputs/sample_results.png")    

# ===================== MAIN =====================

if __name__ == "__main__":

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError("Checkpoint not found.")

    config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    config_vit.in_channels = 1
    config_vit.patches.grid =(32, 32)  # 512 / 16

    model = TransUNet(
        config=config_vit,
        img_size=IMG_SIZE,
        num_classes=1
    ).to(DEVICE)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    save_sample_figure(model)

    all_patients = [
        p for p in os.listdir(DATA_ROOT)
        if p.lower().startswith("patient")
    ]
    all_patients = sorted(all_patients, key=patient_sort_key)

    # The last 4 patients as test
    test_patients = all_patients[16:20]

    for p in test_patients:
        infer_patient(p, model)