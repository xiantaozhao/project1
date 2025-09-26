# src/utils/logging_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import csv

class CSVLogger:
    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = None
        self._file = open(self.path, "a", newline="")
        self._writer = None

    def log(self, row: Dict[str, Any]):
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            if self.path.stat().st_size == 0:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


def make_tb_writer(use_tb: bool, logdir: str | Path):
    if not use_tb:
        return None
    from torch.utils.tensorboard import SummaryWriter
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(logdir))


def expand_var(s: str, dataset_name: str) -> str:
    # 展开 ${data.dataset_name}
    return s.replace("${data.dataset_name}", dataset_name)
