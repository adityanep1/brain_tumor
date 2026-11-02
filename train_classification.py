"""Task B: train the tumor-type classifier.
Usage: python train_classification.py --arch efficientnet_b0 --epochs 20
If checkpoints/unet.pt exists, every image is first segmented and cropped around the tumor
(same step used by the web app). Use --no_mask_crop to train on whole images instead."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms as T

from config import CKPT_DIR, CLS_DIR, CLS_SIZE, IGNORE_CLASSES, IMAGENET_MEAN, IMAGENET_STD, OUT_DIR, SEED
from datasets import ClsDataset, split_indices
from models import UNet, build_classifier
from utils import get_device, list_images, segment_and_crop, set_seed


def load_all(paths, unet, device):
    """Open each image; optionally crop to the segmented tumor; resize to CLS_SIZE (keeps RAM small)."""
    out = []
    for k, p in enumerate(paths, 1):
        img = Image.open(p)
        if unet is not None:
            img, _, _ = segment_and_crop(unet, img, device)
        out.append(img.convert("RGB").resize((CLS_SIZE, CLS_SIZE), Image.BILINEAR))
        if k % 200 == 0:
            print(f"  prepared {k}/{len(paths)}")
    return out


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tl = correct = n = 0
    preds, gts = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            p = out.argmax(1)
            tl += loss.item() * x.size(0); correct += (p == y).sum().item(); n += x.size(0)
            preds += p.cpu().tolist(); gts += y.cpu().tolist()
    return tl / n, correct / n, gts, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet18"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no_mask_crop", action="store_true")
    a = ap.parse_args()

    set_seed(SEED)
    device = get_device()
    print("Device:", device)

    classes = sorted(d.name for d in CLS_DIR.iterdir() if d.is_dir() and d.name.lower() not in IGNORE_CLASSES)
    if len(classes) < 2:
        raise RuntimeError(f"Need >= 2 class folders in {CLS_DIR}, found {classes}")
    paths, labels = [], []
    for ci, c in enumerate(classes):
        files = list_images(CLS_DIR / c)
        paths += files; labels += [ci] * len(files)
        print(f"  {c}: {len(files)} images")

    unet = None
    unet_ckpt = CKPT_DIR / "unet.pt"
    if not a.no_mask_crop:
        if unet_ckpt.exists():
            unet = UNet().to(device)
            unet.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=True)["state_dict"])
            unet.eval()
            print("Using U-Net masks to crop tumor regions.")
        else:
            print("WARNING: checkpoints/unet.pt not found -> training on whole images. "
                  "(Train the U-Net first for the full pipeline.)")
    use_crop = unet is not None

    tr, va, te = split_indices(len(paths), labels)
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    imgs = {}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        print(f"Preparing {name} images...")
        imgs[name] = (load_all([paths[i] for i in idx], unet, device), [labels[i] for i in idx])

    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    tf_train = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(0.2, 0.2),
                          T.ToTensor(), norm])
    tf_eval = T.Compose([T.ToTensor(), norm])
    mk = lambda k, tf, sh: DataLoader(ClsDataset(*imgs[k], tf), batch_size=a.batch, shuffle=sh, num_workers=a.workers)
    train_dl, val_dl, test_dl = mk("train", tf_train, True), mk("val", tf_eval, False), mk("test", tf_eval, False)

    model = build_classifier(a.arch, len(classes), pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    CKPT_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
    hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
    best, bad = -1.0, 0
    for ep in range(1, a.epochs + 1):
        trl, tra, _, _ = run_epoch(model, train_dl, device, loss_fn, opt)
        val, vaa, _, _ = run_epoch(model, val_dl, device, loss_fn)
        sched.step(val)
        for k, v in zip(hist, (trl, val, tra, vaa)):
            hist[k].append(v)
        print(f"Epoch {ep:02d} | train loss {trl:.4f} acc {tra:.4f} | val loss {val:.4f} acc {vaa:.4f}")
        if vaa > best:
            best, bad = vaa, 0
            torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": a.arch,
                        "img_size": CLS_SIZE, "mask_crop": use_crop}, CKPT_DIR / "classifier.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(CKPT_DIR / "classifier.pt", map_location=device, weights_only=True)["state_dict"])
    tl, ta, gts, preds = run_epoch(model, test_dl, device, loss_fn)
    print(f"\nTEST loss {tl:.4f} | accuracy {ta:.4f}\n")
    print(classification_report(gts, preds, target_names=classes, digits=3, zero_division=0))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["va_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["tr_acc"], label="train"); ax[1].plot(hist["va_acc"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    for x in ax: x.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(OUT_DIR / "cls_curves.png", dpi=120); plt.close()

    cm = confusion_matrix(gts, preds, labels=list(range(len(classes))))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", xticks_rotation=30)
    plt.tight_layout(); plt.savefig(OUT_DIR / "cls_confusion_matrix.png", dpi=120); plt.close()
    print("Saved checkpoints/classifier.pt and plots in outputs/")


if __name__ == "__main__":
    main()
