#!/usr/bin/env python3
# demo.py

import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import rotate

# relative paths
MODEL_PATH = "weights/gray1c.pth"
DATA_PATH  = "../data/example_scene"

# --- 1) import your loader ---
from demo.ExampleScene import ExampleScene

# --- 2) import the network ---
from src.model.DepthModel import BEPDepthNetwork

def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    sd   = ckpt.get("state_dict", ckpt)
    model = BEPDepthNetwork().to(device)
    model.load_state_dict(sd)
    model.eval()
    return model

def get_window_indices(mid, total=480, offset=20):
    l = (mid - offset) % total
    r = (mid + offset) % total
    return l, mid, r

def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model + data
    model = load_model(MODEL_PATH, device)
    ds    = ExampleScene(DATA_PATH, device="cpu")  # keep data in CPU, send to GPU per‐batch

    # set up figure
    fig = plt.figure(constrained_layout=False, figsize=(10, 6))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax_spk = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_pred = fig.add_subplot(gs[1, 1])
    ax_gt   = fig.add_subplot(gs[1, 2])
    # hide bottom‐left
    fig.add_subplot(gs[1, 0]).axis("off")
    plt.subplots_adjust(bottom=0.25)

    # initial draw at frame 0
    def draw_frame(mid_idx):
        # 1) fetch & build input
        r,i,l = get_window_indices(mid_idx, len(ds), 20)
        spk_l = ds[l]["input_gray"]  # [2,H,W]
        spk_m = ds[i]["input_gray"]
        spk_r = ds[r]["input_gray"]

        # rotate for tilt
        spk_imgs = [
            spk_l[0].numpy(),
            spk_m[0].numpy(),
            spk_r[0].numpy(),
        ]

        # 2) model inference
        x = torch.stack([spk_l, spk_m, spk_r], dim=0)    # [3,2,H,W]
        x = x.unsqueeze(0).to(device)                   # [1,3,2,H,W]
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                pred_depth, _ = model(x.to(device, dtype=torch.float16))                    # [1,1,H,W]
        pred = pred_depth.squeeze(0).squeeze(0).cpu().numpy()
        gt   = ds[i]["depth_label"].squeeze(0).numpy()

        # clear & re‐plot
        for ax in ax_spk:
            ax.clear()
        ax_pred.clear();  ax_gt.clear()

        for ax, img in zip(ax_spk, spk_imgs):
            ax.imshow(img, cmap="gray")
            ax.axis("off")

        im1 = ax_pred.imshow(pred, cmap="plasma_r", vmin=gt.min(), vmax=gt.max())
        ax_pred.set_title("Predicted depth")
        ax_pred.axis("off")

        im2 = ax_gt.imshow(gt,   cmap="plasma_r", vmin=gt.min(), vmax=gt.max())
        ax_gt.set_title("Ground-truth")
        ax_gt.axis("off")

        # shared colorbar
        # remove old, then add new
        if not hasattr(draw_frame, "cbar"):
            draw_frame.cbar = fig.colorbar(
                im1, ax=[ax_pred, ax_gt], location="right", shrink=0.8,
                label="Depth (m)"
            )
        else:
            # just hook the new image into the existing bar
            draw_frame.cbar.update_normal(im1)

        fig.canvas.draw_idle()

    # initial render
    current = 0
    draw_frame(current)

    # slider
    axslider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider   = Slider(axslider, "Frame", 0, len(ds)-1, valinit=0, valfmt="%0.0f")
    slider.on_changed(lambda val: draw_frame(int(val)))

    plt.show()


if __name__ == "__main__":
    main()
