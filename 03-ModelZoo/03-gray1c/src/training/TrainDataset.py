import os
import random
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """
    PyTorch Dataset for the SpikeDatasetV3 structure:

    root_dir/
        <scene>_<mid>/
            input_spk.pt    # tensor of shape (3, 2, H, W)
            depth_off.pt    # tensor (H, W)
    """
    def __init__(self, root_dir, device):
        super().__init__()
        self.root_dir = root_dir
        self.device = device if isinstance(device, torch.device) else torch.device("cpu")
        # discover subfolders
        self.subdirs = []
        for name in sorted(os.listdir(self.root_dir)):
            path = os.path.join(self.root_dir, name)
            if os.path.isdir(path):
                self.subdirs.append(path)

    def __len__(self):
        return len(self.subdirs)

    def _load_tensor(self, path):
        """Load a .pt file onto the dataset device."""
        return torch.load(path, map_location=self.device, weights_only=True)

    def __getitem__(self, idx):
        folder = self.subdirs[idx]
        # file paths
        inp_path   = os.path.join(folder, 'input_gray.pt')
        offd_path  = os.path.join(folder, 'depth_off.pt')
        # load
        input_gray  = self._load_tensor(inp_path)
        depth_off  = self._load_tensor(offd_path)

        return {
            'input_gray': input_gray,
            'depth_off': depth_off,
            'scene_id': os.path.basename(folder)
        }

    def split_dataset(self, split_ratio):
        """
        Split indices into train/val/test by ratio [train, val, test], sum=1.
        Returns three lists of indices.
        """
        assert len(split_ratio) == 3 and abs(sum(split_ratio) - 1.0) < 1e-6, \
            "split_ratio must be [train, val, test] summing to 1."
        indices = list(range(len(self)))
        random.shuffle(indices)
        n = len(indices)
        t_end = int(n * split_ratio[0])
        v_end = t_end + int(n * split_ratio[1])
        train_idx = indices[:t_end]
        val_idx   = indices[t_end:v_end]
        test_idx  = indices[v_end:]
        return train_idx, val_idx, test_idx

    def save_splits(self, file_path, train_idx, val_idx, test_idx):
        """
        Save split indices to text file:
            train:0,1,2
            val:3,4
            test:5,6
        """
        with open(file_path, 'w') as f:
            f.write('train:' + ','.join(map(str, train_idx)) + '\n')
            f.write('val:'   + ','.join(map(str, val_idx))   + '\n')
            f.write('test:'  + ','.join(map(str, test_idx))  + '\n')

    def load_splits(self, file_path):
        """
        Load train/val/test indices from file. Returns three lists.
        """
        train_idx = []
        val_idx   = []
        test_idx  = []
        with open(file_path, 'r') as f:
            for line in f:
                key, vals = line.strip().split(':')
                lst = [int(x) for x in vals.split(',')] if vals else []
                if key == 'train':
                    train_idx = lst
                elif key == 'val':
                    val_idx = lst
                elif key == 'test':
                    test_idx = lst
        return train_idx, val_idx, test_idx
