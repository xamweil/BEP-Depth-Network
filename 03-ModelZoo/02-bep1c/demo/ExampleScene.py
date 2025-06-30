import os
import torch
from torch.utils.data import Dataset
import numpy as np

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
        packed = torch.load(os.path.join(root_dir, "input_spk.pt"), map_location="cpu", weights_only=True)
        packed_np = packed.numpy()
        spk_np = np.unpackbits(packed_np, axis=-1)
        spk_np = spk_np[..., :1280]
        self.input_spk = torch.from_numpy(spk_np.astype(bool))[:,1:2]
        self.depth_label = torch.load(os.path.join(root_dir, "depth_label.pt"), map_location="cpu", weights_only=True)

        # sanity check shapes
        assert self.input_spk.dim()   == 4, "input_spk must be [480,1,H,W]"
        assert self.depth_label.dim() == 4, "depth_label must be [480,1,H,W]"

    def __len__(self):
        # number of frames
        return self.input_spk.shape[0]

    def __getitem__(self, idx):
        # single-frame fetch (you can wrap/clip idx if needed)
        spk   = self.input_spk[idx].to(self.device)   # [2,H,W]
        depth = self.depth_label[idx].to(self.device) # [1,H,W]

        return {
            "input_spk":   spk,
            "depth_label": depth,
            "frame_id":    idx
        }
