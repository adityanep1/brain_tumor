"""Shared helpers: seeding, MRI preprocessing, metrics, and the segment->crop step."""
import random

import cv2
import numpy as np
import torch
from PIL import Image

from config import IMG_EXTS, MIN_AREA_FRAC, SEG_SIZE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


def normalize_mri(gray):
    """Preprocessing: scale to [0,1] then z-score each image (removes scanner brightness differences)."""
    x = gray.astype(np.float32) / 255.0
    return (x - x.mean()) / (x.std() + 1e-6)


def preprocess_for_unet(gray):
    """uint8 HxW -> float tensor (1,1,SEG_SIZE,SEG_SIZE)."""
    g = cv2.resize(gray, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(normalize_mri(g)).unsqueeze(0).unsqueeze(0)


def dice_iou(logits, target, eps=1e-6):
    """Batch-mean Dice and IoU on thresholded predictions. target: (B,1,H,W) of 0/1."""
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum((1, 2, 3))
    p, t = pred.sum((1, 2, 3)), target.sum((1, 2, 3))
    dice = ((2 * inter + eps) / (p + t + eps)).mean().item()
    iou = ((inter + eps) / (p + t - inter + eps)).mean().item()
    return dice, iou


@torch.no_grad()
def predict_mask(unet, gray, device, thr=0.5):
    """Returns a 0/1 uint8 mask at the ORIGINAL image resolution."""
    unet.eval()
    h, w = gray.shape
    x = preprocess_for_unet(gray).to(device)
    prob = torch.sigmoid(unet(x))[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return (prob > thr).astype(np.uint8)


def crop_to_mask(rgb_img, mask, margin=0.15):
    """Square crop around the tumor bounding box (+margin). rgb_img: PIL, mask: 0/1 array."""
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    side = max(y1 - y0 + 1, x1 - x0 + 1) * (1 + 2 * margin)
    half = max(side, 32) / 2
    left, right = int(max(0, round(cx - half))), int(min(w, round(cx + half)))
    top, bottom = int(max(0, round(cy - half))), int(min(h, round(cy + half)))
    return rgb_img.crop((left, top, right, bottom))


def segment_and_crop(unet, pil_img, device):
    """Step 1 of the pipeline. Returns (crop_or_full_image, mask, tumor_found)."""
    rgb = pil_img.convert("RGB")
    gray = np.array(pil_img.convert("L"))
    mask = predict_mask(unet, gray, device)
    found = bool(mask.sum() >= MIN_AREA_FRAC * mask.size)
    return (crop_to_mask(rgb, mask) if found else rgb), mask, found
