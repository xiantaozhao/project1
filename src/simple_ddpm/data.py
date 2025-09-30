from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageFolderDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 256, grayscale: bool = True):
        self.root = Path(root)
        self.paths = sorted([
            p for p in self.root.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
        ])
        self.image_size = image_size
        self.grayscale = grayscale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert('L' if self.grayscale else 'RGB').resize((self.image_size, self.image_size))
        arr = np.array(img)
        x = torch.from_numpy(arr).float()
        if self.grayscale:
            # H W -> 1 H W
            x = x.unsqueeze(0)
        else:
            # H W C -> C H W
            x = x.permute(2, 0, 1)
        x = x / 255.0
        return { 'image': x, 'path': str(p) }
