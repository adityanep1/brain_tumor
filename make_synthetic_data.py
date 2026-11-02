"""Creates FAKE 'MRI-like' images so you can test the whole pipeline before you have the real dataset.
Usage: python make_synthetic_data.py      (writes into data/; delete data/ contents before using real data)"""
import cv2
import numpy as np

from config import CLS_DIR, SEG_DIR

rng = np.random.default_rng(0)
(SEG_DIR / "images").mkdir(parents=True, exist_ok=True)
(SEG_DIR / "masks").mkdir(parents=True, exist_ok=True)


def make(kind):
    s = 256
    img = np.zeros((s, s), np.float32)
    cv2.ellipse(img, (128, 128), (95, 110), 0, 0, 360, 110, -1)           # "brain"
    mask = np.zeros((s, s), np.uint8)
    if kind == "glioma":      # large irregular blob
        c = (int(rng.integers(90, 170)), int(rng.integers(90, 170))); r = int(rng.integers(25, 38))
        cv2.ellipse(mask, c, (r, int(r * 0.7)), int(rng.integers(0, 180)), 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 4) > 0.35
        val = 170
    elif kind == "meningioma":  # round, near the edge
        a = rng.uniform(0, 6.28); c = (int(128 + 80 * np.cos(a)), int(128 + 90 * np.sin(a)))
        cv2.circle(mask, c, int(rng.integers(14, 22)), 1, -1); mask = mask > 0; val = 230
    else:                     # pituitary: small, central-lower
        c = (int(128 + rng.integers(-8, 8)), int(165 + rng.integers(-8, 8)))
        cv2.circle(mask, c, int(rng.integers(8, 13)), 1, -1); mask = mask > 0; val = 200
    img[mask] = val
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), (mask.astype(np.uint8) * 255)


for kind in ("glioma", "meningioma", "pituitary"):
    (CLS_DIR / kind).mkdir(parents=True, exist_ok=True)
    for i in range(70):
        im, m = make(kind)
        name = f"{kind}_{i:03d}.png"
        cv2.imwrite(str(CLS_DIR / kind / name), im)
        cv2.imwrite(str(SEG_DIR / "images" / name), im)
        cv2.imwrite(str(SEG_DIR / "masks" / name), m)
print("Synthetic data written (210 images).")
