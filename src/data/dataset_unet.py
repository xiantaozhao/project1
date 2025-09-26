# src/data/dataset_unet.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Iterable
from collections import defaultdict
import re, random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

# 已有的 HU 读取函数
from src.data.data_load import data_load_chest


# -------------------- 归一化工具 --------------------
def _minmax01_per_slice(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """逐切片 min-max 归一化到 [0,1]（旧方案，仍保留以便切换）"""
    arr = arr.astype(np.float32, copy=False)
    amin = float(arr.min()); amax = float(arr.max())
    if amax - amin < eps: return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin + eps)

def _scale_with_minmax(arr: np.ndarray, vmin: float, vmax: float, eps: float = 1e-6, clip: bool = True) -> np.ndarray:
    """用给定 vmin/vmax 线性缩放到 [0,1]；可选 clip 防溢出。"""
    arr = arr.astype(np.float32, copy=False)
    if vmax - vmin < eps: 
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - vmin) / (vmax - vmin + eps)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out

def _center_crop_or_pad(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """把 img(H,W) 中心裁剪/零填充到 size=(th,tw)"""
    H, W = img.shape
    th, tw = size
    y0 = max(0, (H - th) // 2); x0 = max(0, (W - tw) // 2)
    y1 = min(H, y0 + th);       x1 = min(W, x0 + tw)
    cropped = img[y0:y1, x0:x1]
    out = np.zeros((th, tw), dtype=cropped.dtype)
    h, w = cropped.shape
    oy = (th - h) // 2; ox = (tw - w) // 2
    out[oy:oy+h, ox:ox+w] = cropped
    return out

def _resolve_split_whitelist(cfg: Dict[str, Any], split_role: str) -> Optional[Set[str]]:
    """
    根据 cfg['split'] 与 split_role('train'|'val'|'test') 解析该 split 的病人白名单。
    返回 None 表示不限制（等价于 all patients）。
    支持三种模式：
      - all_train: 不做划分
      - by_patient_list: 从文件读取 train_list/val_list/test_list
      - manual: 直接从 train_patients/val_patients/test_patients 读取数组
    """
    scfg = cfg.get("split", {}) or {}
    mode = (scfg.get("mode", "all_train") or "all_train").lower()

    if mode == "all_train":
        return None

    if mode == "by_patient_list":
        key = f"{split_role}_list"   # train_list / val_list / test_list
        if key not in scfg:
            raise KeyError(f"[UnetDataset] split.mode='by_patient_list' 需要提供 split.{key}")
        lst = _read_list_file(scfg[key])
        return set(map(str, lst))

    if mode == "manual":
        key = f"{split_role}_patients"  # train_patients / val_patients / test_patients
        if key not in scfg:
            raise KeyError(f"[UnetDataset] split.mode='manual' 需要提供 split.{key}")
        lst = scfg.get(key) or []
        if not isinstance(lst, (list, tuple)):
            raise TypeError(f"[UnetDataset] split.{key} 必须是列表")
        return set(map(str, lst))

    raise ValueError(f"[UnetDataset] 不支持的 split.mode: {mode}")



# -------------------- 数据集 --------------------
class UnetDataset(Dataset):
    """
    输入：recon_<patient>_<end>@<step>.npz
      - npz['recon'] shape [S,H,W] 或 [H,W]
      - （可选）npz['slice_ids']：部分切片时用于对齐；你当前整卷按序存，可不提供
    目标：通过 data_load_chest.load_data_chest(patient, "CT") 读取 HU 卷 (Z,H,W)

    归一化策略（data.norm）：
      - "minmax"：（**推荐**）对同一 patient，使用其 HU 全卷的 min/max，
                     将 recon 与 HU **都用这一对** (min,max) 线性映射到 [0,1]（并 clip）
      - "minmax_slice"：逐切片各自 min-max 到 [0,1]（旧方案，保留以备对比）

    其它：
      - split: all_train / manual / by_patient_list
      - filters: 按病人/角度过滤
      - z_slice: z 轴范围/步长子采样
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        split_role: str = "train",                        # "train" | "val" | "test"
        extra_patients_whitelist: Optional[List[str]] = None,
    ) -> None:
        self.cfg = cfg
        dcfg = cfg["data"]

        # 路径（支持 ${data.dataset_name} 插值）
        raw_root = str(dcfg["recon_root"])
        root_expanded = raw_root.replace("${data.dataset_name}", str(dcfg.get("dataset_name", "")))
        self.recon_root = Path(root_expanded)
        if not self.recon_root.exists():
            raise FileNotFoundError(f"[UnetDataset] recon_root not found: {self.recon_root}")

        self.file_glob   = dcfg.get("file_glob", "recon_*.npz")
        self.filename_re = re.compile(dcfg["filename_regex"])
        self.patch_size: Tuple[int, int] = tuple(dcfg.get("patch_size", [256, 256]))

        # 归一化模式
        self.norm_mode = dcfg.get("norm", "minmax").lower()
        if self.norm_mode not in ("minmax", "minmax_slice"):
            raise ValueError(f"data.norm must be 'minmax' or 'minmax_slice', got {self.norm_mode}")

        # 解析 split
        split_wl = _resolve_split_whitelist(cfg, split_role)
        extra_wl = set(map(str, extra_patients_whitelist)) if extra_patients_whitelist else None

        # 过滤器
        filt = dcfg.get("filters", {}) or {}
        self._patients_in      = set(map(str, filt.get("patients_in", [])))
        self._patients_exclude = set(map(str, filt.get("patients_exclude", [])))
        self._stop_in  = set(float(x) for x in filt.get("stop_deg_in", []))
        self._step_in  = set(float(x) for x in filt.get("step_deg_in", []))
        self._stop_rng = filt.get("stop_deg_range", None)
        self._step_rng = filt.get("step_deg_range", None)

        # z 轴子采样
        zcfg = dcfg.get("z_slice", {}) or {}
        self._use_z_slice = bool(zcfg.get("use", False))
        self._z_start = int(zcfg.get("start", 0))
        self._z_stop_raw = zcfg.get("stop", None)
        self._z_stride = int(zcfg.get("stride", 1))

        # ------- 收集符合条件的文件 -------
        all_files = sorted(self.recon_root.glob(self.file_glob))
        self.files: List[Path] = []
        for p in all_files:
            m = self.filename_re.match(p.name)
            if not m: 
                continue
            patient = m.group("patient")
            stop_deg = float(m.group("end")); step_deg = float(m.group("step"))

            if split_wl is not None and (patient not in split_wl):  # split 白名单
                continue
            if extra_wl is not None and (patient not in extra_wl):  # 额外白名单
                continue

            if self._patients_in and (patient not in self._patients_in):  # filters:病人
                continue
            if patient in self._patients_exclude:
                continue

            if (self._stop_in and stop_deg not in self._stop_in) or (self._step_in and step_deg not in self._step_in):
                continue
            if self._stop_rng:
                smin = float(self._stop_rng.get("min", -1e9)); smax = float(self._stop_rng.get("max", 1e9))
                if not (smin <= stop_deg <= smax): continue
            if self._step_rng:
                tmin = float(self._step_rng.get("min", -1e9)); tmax = float(self._step_rng.get("max", 1e9))
                if not (tmin <= step_deg <= tmax): continue

            self.files.append(p)

        # ------- 建立 index_map（含 z 子采样）与 “该样本属于哪个病人” -------
        self.index_map: List[Tuple[int, int]] = []   # (file_idx, slice_idx_in_file)
        self.sample_patient: List[str] = []          # 与 index_map 对齐的 patient id（字符串）
        for i, p in enumerate(self.files):
            m = self.filename_re.match(p.name); patient = m.group("patient")
            with np.load(p) as npz:
                rec = npz["recon"]
                n = 1 if rec.ndim == 2 else rec.shape[0]
            if not self._use_z_slice:
                idxs = range(n)
            else:
                z1 = n if (self._z_stop_raw is None) else min(int(self._z_stop_raw), n)
                z0 = max(0, self._z_start); step = max(1, self._z_stride)
                idxs = range(z0, z1, step) if z0 < z1 else range(n)
            for k in idxs:
                self.index_map.append((i, k))
                self.sample_patient.append(patient)

        # HU 缓存 + 每病人 HU 全卷 min/max 缓存（用于 volume 归一化）
        self._hu_cache: Dict[str, np.ndarray] = {}
        self._hu_minmax: Dict[str, Tuple[float, float]] = {}

    def __len__(self) -> int:
        return len(self.index_map)

    # -------- 内部工具 --------
    def _parse_filename(self, path: Path) -> Dict[str, Any]:
        m = self.filename_re.match(path.name)
        if not m:
            raise ValueError(f"[UnetDataset] Unexpected recon filename: {path.name}")
        return {"patient": m.group("patient"),
                "start_deg": 0.0,
                "stop_deg": float(m.group("end")),
                "step_deg": float(m.group("step"))}

    def _load_hu_volume_zyx(self, patient: str) -> np.ndarray:
        if patient in self._hu_cache:
            return self._hu_cache[patient]
        vol_HU_zyx, spacing_dzyx, meta = data_load_chest.load_data_chest(patient, "CT")
        vol = vol_HU_zyx.astype(np.float32, copy=False)
        self._hu_cache[patient] = vol
        return vol

    def _get_hu_minmax(self, patient: str) -> Tuple[float, float]:
        """
        返回该 patient 的 HU 全卷 (Z,H,W) 的 (min,max)，并缓存。
        用于将 recon/HU 都按同一对 min/max 映射到 [0,1]。
        """
        if patient in self._hu_minmax:
            return self._hu_minmax[patient]
        vol = self._load_hu_volume_zyx(patient)
        vmin = float(vol.min()); vmax = float(vol.max())
        # 防御：如果极端情况 min==max，则略微扩展
        if vmax <= vmin: vmax = vmin + 1.0
        self._hu_minmax[patient] = (vmin, vmax)
        return vmin, vmax

    # -------- 取样本 --------
    def __getitem__(self, index: int) -> Dict[str, Any]:
        file_idx, slice_idx_local = self.index_map[index]
        p = self.files[file_idx]
        meta_file = self._parse_filename(p)
        patient = meta_file["patient"]

        with np.load(p) as npz:
            recon = npz["recon"]
            if recon.ndim == 2:
                x = recon
                sid = int(npz["slice_ids"][0]) if ("slice_ids" in npz and len(npz["slice_ids"]) > 0) else 0
            else:
                x = recon[slice_idx_local]
                sid = int(npz["slice_ids"][slice_idx_local]) if ("slice_ids" in npz) else slice_idx_local

        vol_hu = self._load_hu_volume_zyx(patient)
        if not (0 <= sid < vol_hu.shape[0]):
            raise IndexError(f"[UnetDataset] slice_id {sid} out of range for patient {patient} with Z={vol_hu.shape[0]}")
        y = vol_hu[sid]  # (H,W)

        # ---- 归一化 ----
        if self.norm_mode == "minmax":
            # 用该病人的 HU 卷整体 min/max，对 recon 与 HU 同步缩放
            vmin, vmax = self._get_hu_minmax(patient)
            x = _scale_with_minmax(x, vmin, vmax, clip=True)
            y = _scale_with_minmax(y, vmin, vmax, clip=True)
        else:  # "minmax_slice"
            x = _minmax01_per_slice(x)
            y = _minmax01_per_slice(y)

        # ---- 尺寸统一 ----
        x = _center_crop_or_pad(x, self.patch_size)
        y = _center_crop_or_pad(y, self.patch_size)

        # ---- Tensor ----
        xt = torch.from_numpy(x[None, ...])  # [1,H,W]
        yt = torch.from_numpy(y[None, ...])  # [1,H,W]

        return {
            "x": xt,
            "y": yt,
            "meta": {
                "patient": patient,
                "slice_id": int(sid),
                "recon_path": str(p),
                "stop_deg": meta_file["stop_deg"],
                "step_deg": meta_file["step_deg"],
                "index_in_file": int(slice_idx_local),
            },
        }


# -------------------- 按“病人”成批的 BatchSampler --------------------
class UnetSampler(Sampler[List[int]]):
    """
    让 DataLoader 产出的每个 batch 只包含“同一个病人”的切片。
    - group_by="patient"：按病人分组（默认）
    - group_by="patient_angle"：按 (patient, stop_deg, step_deg) 分组（不混角度）

    用法：
      sampler = UnetSampler(dataset, batch_size=8, shuffle=True, drop_last=False, group_by="patient")
      loader = DataLoader(dataset, batch_sampler=sampler, num_workers=..., pin_memory=...)
    """
    def __init__(self, dataset: "UnetDataset", batch_size: int,
                 shuffle: bool = True, drop_last: bool = False,
                 group_by: str = "patient"):
        # 兼容 PyTorch 2.2+：Sampler.__init__() 不再接收 data_source
        super().__init__()  # ← 关键修改：不要传 dataset

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        if group_by not in ("patient", "patient_angle"):
            raise ValueError("group_by must be 'patient' or 'patient_angle'")
        self.group_by = group_by

        # 解析文件名中的角度信息，用于 (patient, stop, step) 键
        # 允许 filename_re 缺失时报出更清晰的错误
        if not hasattr(dataset, "filename_re") or not isinstance(dataset.filename_re, re.Pattern):
            raise AttributeError("dataset.filename_re 未设置或不是有效的正则，请在 UnetDataset 中提供命名组 'end' 和 'step'")

        self._index_group_key: List[Tuple[str, float, float]] = []
        for (file_idx, _), patient in zip(dataset.index_map, dataset.sample_patient):
            p = dataset.files[file_idx]
            m = dataset.filename_re.match(p.name)
            if m is None:
                raise ValueError(f"文件名不匹配正则：{p.name}  正则：{dataset.filename_re.pattern}")
            stop_deg = float(m.group("end"))
            step_deg = float(m.group("step"))
            self._index_group_key.append((patient, stop_deg, step_deg))

        # 构建 分组 -> indices
        self.groups: Dict[Any, List[int]] = defaultdict(list)
        for idx, (patient, stop, step) in enumerate(self._index_group_key):
            key = patient if self.group_by == "patient" else (patient, stop, step)
            self.groups[key].append(idx)

        # 预计算批次数（供 __len__ 使用）
        self._num_batches = 0
        B = self.batch_size
        for idxs in self.groups.values():
            nb = len(idxs) // B
            if not self.drop_last and (len(idxs) % B):
                nb += 1
            self._num_batches += nb

    def __len__(self) -> int:
        # 返回“本 epoch 将产生的 batch 数”
        return self._num_batches

    def __iter__(self) -> Iterable[List[int]]:
        # 先随机组顺序，再组内随机
        keys = list(self.groups.keys())
        if self.shuffle:
            random.shuffle(keys)

        B = self.batch_size
        for k in keys:
            idxs = self.groups[k][:]
            if self.shuffle:
                random.shuffle(idxs)

            full = (len(idxs) // B) * B
            # 完整批
            for s in range(0, full, B):
                yield idxs[s:s+B]
            # 余数批
            if not self.drop_last and full < len(idxs):
                yield idxs[full:]