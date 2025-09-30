from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["ImageFolderDataset"]


class ImageFolderDataset(Dataset):
    """Simple image folder dataset for diffusion toy experiments."""

    def __init__(self, root: Union[str, Path], image_size: int = 256, grayscale: bool = True) -> None:
        self.root = Path(root)
        self.paths = sorted(
            [p for p in self.root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        )
        self.image_size = image_size
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("L" if self.grayscale else "RGB").resize((self.image_size, self.image_size))
        arr = np.array(img)
        tensor = torch.from_numpy(arr).float()
        if self.grayscale:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor / 255.0
        return {"image": tensor, "path": str(path)}
