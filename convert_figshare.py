import argparse, shutil
from pathlib import Path
import cv2
import numpy as np
from config import CLS_DIR, SEG_DIR

LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}

def read_mat(path):
    try:
        import h5py
        with h5py.File(path, "r") as f:
            d = f["cjdata"]
            img = np.array(d["image"]).T.astype(np.float32)
            mask = np.array(d["tumorMask"]).T
            label = int(np.array(d["label"]).squeeze())
    except OSError:
        from scipy.io import loadmat
        d = loadmat(str(path), squeeze_me=True, struct_as_record=False)["cjdata"]
        img, mask, label = d.image.astype(np.float32), d.tumorMask, int(d.label)
    return img, mask, label

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
a = ap.parse_args()
files = sorted((p for p in Path(a.src).rglob("*.mat") if p.stem.isdigit()), key=lambda p: int(p.stem))
if not files:
    raise SystemExit("No .mat files found under " + a.src)
seg_img, seg_mask = SEG_DIR / "images", SEG_DIR / "masks"
for d in (seg_img, seg_mask, CLS_DIR):
    if d.exists():
        shutil.rmtree(d)
seg_img.mkdir(parents=True); seg_mask.mkdir(parents=True)
counts = {}
for k, p in enumerate(files, 1):
    img, mask, label = read_mat(p)
    if label not in LABELS:
        continue
    img = ((img - img.min()) / (img.max() - img.min() + 1e-6) * 255.0).astype(np.uint8)
    mask = ((mask > 0) * 255).astype(np.uint8)
    name = p.stem + ".png"
    cd = CLS_DIR / LABELS[label]
    cd.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cd / name), img)
    cv2.imwrite(str(seg_img / name), img)
    cv2.imwrite(str(seg_mask / name), mask)
    counts[LABELS[label]] = counts.get(LABELS[label], 0) + 1
    if k % 500 == 0:
        print(k, "/", len(files))
print("Done:", counts)
