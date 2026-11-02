"""Dataset classes + 70/15/15 train/val/test splitting."""
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import SEED, SEG_DIR, SEG_SIZE
from utils import list_images, load_gray, normalize_mri


def split_indices(n, labels=None):
    """70% train / 15% val / 15% test, reproducible, stratified when labels are given."""
    idx = np.arange(n)
    lab = np.asarray(labels) if labels is not None else None
    train, rest = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=lab)
    lab_rest = lab[rest] if lab is not None else None
    val, test = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=lab_rest)
    return train.tolist(), val.tolist(), test.tolist()


def find_seg_pairs():
    """Match data/segmentation/images/X.* with data/segmentation/masks/X.* (or X_mask.*)."""
    img_dir, mask_dir = SEG_DIR / "images", SEG_DIR / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected folders: {img_dir} and {mask_dir}")
    masks = {p.stem: p for p in list_images(mask_dir)}
    pairs = []
    for img in list_images(img_dir):
        m = masks.get(img.stem) or masks.get(img.stem + "_mask")
        if m is not None:
            pairs.append((img, m))
    if len(pairs) < 10:
        raise RuntimeError(f"Only {len(pairs)} image/mask pairs found - check file names match.")
    return pairs


class SegDataset(Dataset):
    def __init__(self, pairs, train=False):
        self.pairs, self.train = pairs, train

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img, mask):
        if np.random.rand() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if np.random.rand() < 0.3:
            img, mask = img[::-1], mask[::-1]
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        ang = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((SEG_SIZE / 2, SEG_SIZE / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (SEG_SIZE, SEG_SIZE), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, M, (SEG_SIZE, SEG_SIZE), flags=cv2.INTER_NEAREST)
        a, b = np.random.uniform(0.85, 1.15), np.random.uniform(-10, 10)  # contrast / brightness
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return img, mask

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.resize(load_gray(ip), (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(load_gray(mp), (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
        if self.train:
            img, mask = self._augment(img, mask)
        x = torch.from_numpy(normalize_mri(img)).unsqueeze(0)
        y = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return x, y


class ClsDataset(Dataset):
    """Holds pre-resized PIL images in memory + integer labels."""

    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]
