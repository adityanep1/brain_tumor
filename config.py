"""Central settings shared by training scripts and the web backend."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SEG_DIR = DATA_DIR / "segmentation"      # images/ + masks/
CLS_DIR = DATA_DIR / "classification"    # one sub-folder per tumor type
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs"

SEG_SIZE = 128        # U-Net input size (must be divisible by 16)
CLS_SIZE = 224        # classifier input size
SEED = 42
MIN_AREA_FRAC = 0.002  # predicted mask smaller than 0.2% of the image => "no tumor"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# class folders with these names are skipped for tumor-TYPE classification
IGNORE_CLASSES = {"notumor", "no_tumor", "no tumor", "normal", "healthy", "no"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

