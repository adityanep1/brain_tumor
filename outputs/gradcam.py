"""Grad-CAM: highlights the image regions that pushed the classifier toward its predicted class."""
import cv2
import numpy as np
import torch


def target_layer(model, arch):
    """Last convolutional block of each supported classifier."""
    return model.features[-1] if arch == "efficientnet_b0" else model.layer4[-1]


def grad_cam(model, arch, x, class_idx):
    """x: (1,3,H,W) normalised tensor. Returns a (h,w) heat-map in [0,1] (low-res, e.g. 7x7)."""
    store = {}
    layer = target_layer(model, arch)

    def fwd_hook(module, inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        model.eval()
        with torch.enable_grad():
            out = model(x)
            model.zero_grad()
            out[0, class_idx].backward()
    finally:
        handle.remove()

    act, grad = store["act"].detach(), store["grad"]
    weights = grad.mean(dim=(2, 3), keepdim=True)            # importance of each feature map
    cam = torch.relu((weights * act).sum(dim=1))[0].cpu().numpy()
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)



def overlay_cam(rgb, cam, alpha=0.45):
    """rgb: uint8 HxWx3 image, cam: heat-map in [0,1]. Returns uint8 RGB overlay."""
    h, w = rgb.shape[:2]
    heat = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)