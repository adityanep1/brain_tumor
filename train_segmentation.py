"""Task A: train the U-Net.  Usage: python train_segmentation.py --epochs 30"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CKPT_DIR, OUT_DIR, SEED, SEG_SIZE
from datasets import SegDataset, find_seg_pairs, split_indices
from models import UNet
from utils import dice_iou, get_device, set_seed


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        return self.bce(logits, target) + dice.mean()


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_dice = tot_iou = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()      # backpropagation: compute gradients
                optimizer.step()     # gradient descent step (Adam)
            d, i = dice_iou(logits.detach(), y)
            b = x.size(0)
            tot_loss += loss.item() * b; tot_dice += d * b; tot_iou += i * b; n += b
    return tot_loss / n, tot_dice / n, tot_iou / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()

    set_seed(SEED)
    device = get_device()
    print("Device:", device)
    pairs = find_seg_pairs()
    tr, va, te = split_indices(len(pairs))
    print(f"Pairs: {len(pairs)}  train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    mk = lambda idx, train: DataLoader(SegDataset([pairs[i] for i in idx], train), batch_size=a.batch,
                                       shuffle=train, num_workers=a.workers)
    train_dl, val_dl, test_dl = mk(tr, True), mk(va, False), mk(te, False)

    model = UNet().to(device)
    loss_fn = BCEDiceLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    CKPT_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)
    hist = {"tr_loss": [], "va_loss": [], "tr_dice": [], "va_dice": []}
    best, bad = -1.0, 0
    for ep in range(1, a.epochs + 1):
        trl, trd, _ = run_epoch(model, train_dl, device, loss_fn, opt)
        val, vad, vai = run_epoch(model, val_dl, device, loss_fn)
        sched.step(vad)
        for k, v in zip(hist, (trl, val, trd, vad)):
            hist[k].append(v)
        print(f"Epoch {ep:02d} | train loss {trl:.4f} dice {trd:.4f} | val loss {val:.4f} dice {vad:.4f} iou {vai:.4f}")
        if vad > best:
            best, bad = vad, 0
            torch.save({"state_dict": model.state_dict(), "img_size": SEG_SIZE}, CKPT_DIR / "unet.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping.")
                break

    # final test on unseen data with the best weights
    model.load_state_dict(torch.load(CKPT_DIR / "unet.pt", map_location=device, weights_only=True)["state_dict"])
    tl, td, ti = run_epoch(model, test_dl, device, loss_fn)
    print(f"\nTEST  loss {tl:.4f} | Dice {td:.4f} | IoU {ti:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["va_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["tr_dice"], label="train"); ax[1].plot(hist["va_dice"], label="val"); ax[1].set_title("Dice"); ax[1].legend()
    for x in ax: x.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(OUT_DIR / "seg_curves.png", dpi=120); plt.close()

    # sample predictions from the test set
    model.eval()
    n = min(4, len(test_dl.dataset))
    fig, ax = plt.subplots(3, n, figsize=(3 * n, 9), squeeze=False)
    for j in range(n):
        x, y = test_dl.dataset[j]
        with torch.no_grad():
            p = (torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0] > 0.5).cpu().numpy()
        ax[0, j].imshow(x[0], cmap="gray"); ax[1, j].imshow(y[0], cmap="gray"); ax[2, j].imshow(p, cmap="gray")
        for r, t in enumerate(("MRI", "Ground truth", "Prediction")):
            ax[r, j].set_title(t); ax[r, j].axis("off")
    plt.tight_layout(); plt.savefig(OUT_DIR / "seg_samples.png", dpi=120); plt.close()
    print("Saved checkpoints/unet.pt and outputs/seg_curves.png, seg_samples.png")


if __name__ == "__main__":
    main()
