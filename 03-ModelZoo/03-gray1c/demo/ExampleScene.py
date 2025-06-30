import os
import torch
from torch.utils.data import Dataset

class ExampleScene(Dataset):
    """
    ExampleScene: one “scene” of 480 frames,
    with three big tensors:
      - input_spk.pt    → (480,2,H,W)
      - depth_label.pt  → (480,1,H,W)
    """

    def __init__(self, root_dir, device="cpu"):
        self.root_dir = root_dir
        # normalize device
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        # load once into CPU memory
        self.input_gray   = torch.load(os.path.join(root_dir, "input_gray.pt"),   map_location="cpu", weights_only=True)
        self.depth_label = torch.load(os.path.join(root_dir, "depth_label.pt"), map_location="cpu", weights_only=True)

        # sanity check shapes
        assert self.input_gray.dim()   == 4, "input_gray must be [480,1,H,W]"
        assert self.depth_label.dim() == 4, "depth_label must be [480,1,H,W]"

    def __len__(self):
        # number of frames
        return self.input_gray.shape[0]

    def __getitem__(self, idx):
        # single-frame fetch
        gray   = self.input_gray[idx].to(self.device)   # [1,H,W]
        depth = self.depth_label[idx].to(self.device) # [1,H,W]

        return {
            "input_gray":   gray,
            "depth_label": depth,
            "frame_id":    idx
        }
