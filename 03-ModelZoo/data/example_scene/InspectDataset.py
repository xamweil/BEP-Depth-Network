
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from ExampleScene import ExampleScene

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ds = ExampleScene(root_dir, device="cpu")

    # Prepare the figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.25)  # leave space at bottom for the slider

    # Initial frame index
    idx0 = 0
    data = ds[idx0]
    # Combine the two spike channels for display
    spk_img = data["input_spk"].sum(dim=0).numpy()
    gray_img = data["input_gray"][0].numpy()
    depth_img = data["depth_label"][0].numpy()

    im_spk   = axes[0].imshow(spk_img, cmap='gray', interpolation='nearest')
    axes[0].set_title("Spikes (sum of channels)")
    im_gray  = axes[1].imshow(gray_img, cmap="gray")
    axes[1].set_title("Gray")
    vmin = float(depth_img.min())
    vmax = float(depth_img.max())
    im_depth = axes[2].imshow(depth_img, vmin=vmin, vmax=vmax)
    axes[2].set_title("Depth")

    # Slider axis
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, "Frame", 0, len(ds) - 1, valinit=idx0, valfmt="%0.0f")

    def update(val):
        i = int(slider.val)
        data = ds[i]
        spk_img   = data["input_spk"].sum(dim=0).numpy()
        gray_img  = data["input_gray"][0].numpy()
        depth_img = data["depth_label"][0].numpy()
        im_spk.set_data(spk_img)
        im_gray.set_data(gray_img)
        im_depth.set_data(depth_img)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    main()

