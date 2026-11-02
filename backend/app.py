"""FastAPI backend.  Run from the project root:  uvicorn backend.app:app --reload"""
import base64
import io
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import transforms as T

from config import CKPT_DIR, CLS_SIZE, IMAGENET_MEAN, IMAGENET_STD
from models import UNet, build_classifier
from utils import get_device, segment_and_crop
from gradcam import grad_cam, overlay_cam

state = {"unet": None, "clf": None, "classes": [], "arch": None, "device": get_device()}
MAX_BYTES = 10 * 1024 * 1024
to_tensor = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


@asynccontextmanager
async def lifespan(app):
    dev = state["device"]
    up, cp = CKPT_DIR / "unet.pt", CKPT_DIR / "classifier.pt"
    if up.exists():
        m = UNet().to(dev)
        m.load_state_dict(torch.load(up, map_location=dev, weights_only=True)["state_dict"])
        state["unet"] = m.eval()
    if cp.exists():
        ck = torch.load(cp, map_location=dev, weights_only=True)
        c = build_classifier(ck["arch"], len(ck["classes"]), pretrained=False).to(dev)
        c.load_state_dict(ck["state_dict"])
        state["clf"], state["classes"], state["arch"] = c.eval(), ck["classes"], ck["arch"]
    print(f"Models loaded -> U-Net: {state['unet'] is not None}, classifier: {state['clf'] is not None}")
    yield


app = FastAPI(title="Brain Tumor Segmentation & Classification", lifespan=lifespan)


def to_b64(arr):
    ok, buf = cv2.imencode(".png", arr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/health")
def health():
    return {"unet": state["unet"] is not None, "classifier": state["clf"] is not None, "classes": state["classes"]}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        return _predict(file)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


def _predict(file):
    if state["unet"] is None:
        raise HTTPException(503, "U-Net checkpoint not found. Train it first (train_segmentation.py).")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 10 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "Could not read the file as an image.")
    img.thumbnail((1024, 1024))  # keep inference fast

    # reject colour photos: real MRI slices are (almost) grayscale
    arr = np.array(img.convert("RGB")).astype(int)
    if (arr.max(axis=2) - arr.min(axis=2)).mean() > 15:
        raise HTTPException(400, "This looks like a colour photo, not an MRI scan. Please upload a grayscale MRI slice.")

    dev = state["device"]
    crop, mask, found = segment_and_crop(state["unet"], img, dev)

    rgb = np.array(img.convert("RGB"))
    overlay = rgb.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 255, 0), 1)

    result = {
        "tumor_found": found,
        "tumor_area_percent": round(float(mask.mean()) * 100, 2),
        "tumor_type": None, "confidence": None, "probabilities": {},
        "original_png": to_b64(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
        "mask_png": to_b64(mask * 255),
        "overlay_png": to_b64(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)),
        "message": "No tumor detected." if not found else "",
    }
    if found:
        if state["clf"] is None:
            result["message"] = "Tumor segmented, but classifier checkpoint not found."
        else:
            x = to_tensor(crop.convert("RGB").resize((CLS_SIZE, CLS_SIZE), Image.BILINEAR)).unsqueeze(0).to(dev)
            with torch.no_grad():
                probs = torch.softmax(state["clf"](x), dim=1)[0].cpu().numpy()
            k = int(probs.argmax())
            result["tumor_type"] = state["classes"][k]
            result["confidence"] = round(float(probs[k]), 4)
            result["probabilities"] = {c: round(float(p), 4) for c, p in zip(state["classes"], probs)}
            cam = grad_cam(state["clf"], state["arch"], x, k)
            crop_rgb = np.array(crop.convert("RGB").resize((CLS_SIZE, CLS_SIZE), Image.BILINEAR))
            result["gradcam_png"] = to_b64(cv2.cvtColor(overlay_cam(crop_rgb, cam), cv2.COLOR_RGB2BGR))
    return result