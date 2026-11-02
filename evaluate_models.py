import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.metrics import classification_report

from config import CKPT_DIR, CLS_DIR, IMAGENET_MEAN, IMAGENET_STD
from datasets import ClsDataset, SegDataset, find_seg_pairs, split_indices
from models import UNet, build_classifier
from utils import get_device, list_images
from train_segmentation import BCEDiceLoss, run_epoch as seg_epoch
from train_classification import load_all, run_epoch as cls_epoch

dev = get_device()

# ---- U-Net on the test split ----
unet = UNet().to(dev)
unet.load_state_dict(torch.load(CKPT_DIR / "unet.pt", map_location=dev, weights_only=True)["state_dict"])
unet.eval()
pairs = find_seg_pairs()
tr, va, te = split_indices(len(pairs))
dl = DataLoader(SegDataset([pairs[i] for i in te], False), batch_size=8)
loss, dice, iou = seg_epoch(unet, dl, dev, BCEDiceLoss())
print(f"\nSEGMENTATION TEST  ({len(te)} images)  loss {loss:.4f} | Dice {dice:.4f} | IoU {iou:.4f}")

# ---- Classifier on the test split ----
ck = torch.load(CKPT_DIR / "classifier.pt", map_location=dev, weights_only=True)
classes = ck["classes"]
model = build_classifier(ck["arch"], len(classes), pretrained=False).to(dev)
model.load_state_dict(ck["state_dict"])
model.eval()
paths, labels = [], []
for ci, c in enumerate(classes):
    files = list_images(CLS_DIR / c)
    paths += files
    labels += [ci] * len(files)
tr, va, te = split_indices(len(paths), labels)
print("Preparing test images...")
imgs = load_all([paths[i] for i in te], unet if ck.get("mask_crop") else None, dev)
tf = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
dl = DataLoader(ClsDataset(imgs, [labels[i] for i in te], tf), batch_size=16)
tl, acc, gts, preds = cls_epoch(model, dl, dev, nn.CrossEntropyLoss())
print(f"\nCLASSIFICATION TEST  network: {ck['arch']} | mask crop: {ck.get('mask_crop')} | images: {len(te)}")
print(f"accuracy {acc:.4f}\n")
print(classification_report(gts, preds, target_names=classes, digits=3, zero_division=0))
