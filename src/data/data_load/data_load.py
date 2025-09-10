"""
data_load.py — 自动从给定[LIDC-IDRI-xxxx]患者目录下递归查找最“深”的[DICOM]序列并读取体数据。

用法（命令行）：
    python -m src.data.data_load.data_load_chest \
        --root data/raw \
        --category chest \
        --patient LIDC-IDRI-0001 \
        --choose deepest \
        --save-npy data/interim/chest/LIDC-IDRI-0001_HU.npy \
        --save-json data/interim//chest/LIDC-IDRI-0001_meta.json

也可作为模块导入：
    from src.data.data_load_chest import load_HU_from_raw
    vol_HU, spacing_dzyx, meta = load_HU_from_raw(
        root='data/raw', category='chest', patient='LIDC-IDRI-0001', choose='deepest'
    )
"""


from __future__ import annotations
import os
import json
import pandas as pd
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import SimpleITK as sitk 

# 全局关闭ITK级别warning显示（可再配合stderr重定向）
try:
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
except Exception:
    pass


import contextlib, io



META_CSV = os.path.join(
    os.path.dirname(__file__),
    "../../../data/raw/chest/manifest-1600709154662/metadata.csv",
)
BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../data/raw/chest/manifest-1600709154662",
)

def _quiet_series_ids(reader, dir_path: Path):
    """Suppress GDCM stderr noise for directories without series."""
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            ids = reader.GetGDCMSeriesIDs(str(dir_path))
            return ids or []
        except Exception:
            return []

def _quiet_series_filenames(reader, dir_path: Path, sid: str):
    """Suppress GDCM stderr noise when querying filenames for a series id."""
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            files = _quiet_series_filenames(reader, dir_path, sid)
            return files or []
        except Exception:
            return []


# ---------------------------
# 数据结构
# ---------------------------

@dataclass
class SeriesCandidate:
    dir_path: Path
    series_id: str
    file_names: List[str]
    depth: int  # 路径深度（用于“最深”策略）

# ---------------------------
# 工具函数
# ---------------------------

def get_patient_paths(meta_csv: str, subject_id, modality: str, base_dir: str):
    """
    根据病人 ID 和 Modality 返回本地文件夹路径。

    参数:
    --------
    meta_csv : str
        metadata.csv 文件的路径
    subject_id : str 或 int
        病人 ID，可以是:
        - "0001" 或 "1" (函数会自动转成 "LIDC-IDRI-0001")
        - 已经完整的 "LIDC-IDRI-0001"
    modality : str
        想要的影像类型，例如 "CT" 或 "DX"
    base_dir : str
        数据的根目录，例如 "/home/xm/project1/data/raw/chest/manifest-1600709154662"

    返回:
    --------
    list[str]
        满足条件的序列路径列表，每个路径下都有一堆 DICOM 文件。
    """
    # 读取 metadata
    meta = pd.read_csv(meta_csv)

    # 统一病人 ID 格式
    subj_str = str(subject_id).zfill(4)  # 补成 4 位，如 "1" -> "0001"
    if not str(subject_id).startswith("LIDC-IDRI"):
        subj_str = f"LIDC-IDRI-{subj_str}"
    else:
        subj_str = subject_id

    # 筛选
    rows = meta[(meta["Subject ID"] == subj_str) & (meta["Modality"] == modality)]

    # 拼接本地绝对路径
    paths = [os.path.join(base_dir, loc.lstrip(".\\")) for loc in rows["File Location"]]

    return paths

def _series_in_dir(dir_path: Path) -> List['SeriesCandidate']:
    """
    在单个目录中查找所有[GDCM]可识别的[DICOM Series]。
    不递归！外层调用会自己递归地遍历目录树。
    """
    reader = sitk.ImageSeriesReader()
    try:
        series_ids = reader.GetGDCMSeriesIDs(str(dir_path))
    except Exception:
        series_ids = None
    if not series_ids:
        return []
    out: List[SeriesCandidate] = []
    for sid in series_ids:
        try:
            files = reader.GetGDCMSeriesFileNames(str(dir_path), sid)
            if files:
                out.append(SeriesCandidate(dir_path=dir_path, series_id=sid, file_names=files, depth=len(dir_path.parts)))
        except Exception:
            continue
    return out


def _gather_series_candidates(patient_root: Path) -> List['SeriesCandidate']:
    """
    递归遍历患者目录下所有子目录，收集可读的[DICOM]序列候选。
    """
    candidates: List[SeriesCandidate] = []
    for cur_dir, subdirs, files in os.walk(patient_root):
        cur = Path(cur_dir)
        found = _series_in_dir(cur)
        if found:
            candidates.extend(found)
    return candidates


def _choose_candidate(candidates: List['SeriesCandidate'], strategy: str = "deepest", index: Optional[int] = None) -> 'SeriesCandidate':
    """
    根据策略选择一个候选：
    - deepest：选择路径深度最大的；如并列，再按文件数最多；再按路径名排序
    - largest：直接按文件数最多；并列再按最深
    - first：按路径名排序后的第一个
    - index：按列出的顺序索引（配合--list显示）
    """
    if not candidates:
        raise RuntimeError("未找到可读的 DICOM Series 候选。")

    if strategy == "index":
        if index is None:
            raise ValueError("strategy='index' 需要提供 index 参数。")
        if not (0 <= index < len(candidates)):
            raise IndexError(f"index 越界：0 <= index < {len(candidates)}")
        return candidates[index]

    # 统一一个稳定排序键
    def key_deepest(c: SeriesCandidate):
        return (c.depth, len(c.file_names), str(c.dir_path))

    def key_largest(c: SeriesCandidate):
        return (len(c.file_names), c.depth, str(c.dir_path))

    if strategy == "deepest":
        return sorted(candidates, key=key_deepest, reverse=True)[0]
    elif strategy == "largest":
        return sorted(candidates, key=key_largest, reverse=True)[0]
    elif strategy == "first":
        return sorted(candidates, key=lambda c: str(c.dir_path))[0]
    else:
        raise ValueError(f"未知选择策略：{strategy}. 允许值：deepest/largest/first/index")


def _print_candidates(candidates: List['SeriesCandidate']):
    print("\n可用[DICOM Series]候选：")
    for i, c in enumerate(candidates):
        print(f"[{i:02d}] depth={c.depth:02d}  files={len(c.file_names):4d}  dir={c.dir_path}  series_id={c.series_id}")


# ---------------------------
# 对外主入口
# ---------------------------

def load_HU_from_raw(
    root: Path | str,
    category: str,
    patient: str,
    choose: str = "deepest",
    index: Optional[int] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float], dict]:
    """
    从根目录(root)/category/patient 出发，递归找到目标[Series]并加载。

    返回：
        vol_HU: np.ndarray, 形状 (Z, Y, X)
        spacing_dzyx: (dz, dy, dx) in mm
        meta: dict, 包含路径、文件数、series_id 等元数据
    """
    root = Path(root)
    patient_root = root / category / patient
    if not patient_root.exists():
        # 允许更灵活匹配：在 category 下递归搜寻名为 patient 的目录
        matches = [p for p in (root / category).rglob(patient) if p.is_dir()]
        if not matches:
            raise FileNotFoundError(f"未找到患者目录：{patient_root}，且在 {root/category} 下也未匹配到 {patient}")
        # 使用最深（path.parts最多）的匹配
        patient_root = sorted(matches, key=lambda p: (len(p.parts), str(p)), reverse=True)[0]

    candidates = _gather_series_candidates(patient_root)
    if not candidates:
        raise RuntimeError(f"目录 {patient_root} 下未发现任何可读的 DICOM Series。")

    candidate = _choose_candidate(candidates, strategy=choose, index=index)

    # 读取影像
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(candidate.file_names)
    img = reader.Execute()

    vol_HU = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    sx, sy, sz = img.GetSpacing()         # [SimpleITK]返回(x,y,z)
    spacing_dzyx = (float(sz), float(sy), float(sx))  # 调整为 (dz, dy, dx)

    meta = {
        "patient_root": str(patient_root),
        "chosen_dir": str(candidate.dir_path),
        "series_id": candidate.series_id,
        "num_files": len(candidate.file_names),
        "choose_strategy": choose,
        "shape_zyx": tuple(map(int, vol_HU.shape)),
        "spacing_dzyx_mm": spacing_dzyx,
        "dtype": str(vol_HU.dtype),
    }
    return vol_HU, spacing_dzyx, meta


# ---------------------------
# 命令行接口
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="自动查找并读取[LIDC-IDRI]患者的最深[DICOM Series]")
    ap.add_argument("--root", type=str, default="data/raw", help="数据根目录（默认 data/raw）")
    ap.add_argument("--category", type=str, default="chest", help="类别/模态（默认 chest）")
    ap.add_argument("--patient", type=str, default="LIDC-IDRI-0001", help="患者目录名，如 LIDC-IDRI-0001")
    ap.add_argument("--choose", type=str, default="deepest", choices=["deepest", "largest", "first", "index"],
                    help="候选选择策略（默认 deepest）")
    ap.add_argument("--index", type=int, default=None, help="当 choose=index 时使用的索引")
    ap.add_argument("--list", action="store_true", help="仅列出候选，不加载")
    ap.add_argument("--save-npy", type=str, default=None, help="将体数据(HU)保存为 .npy")
    ap.add_argument("--save-json", type=str, default=None, help="将元数据保存为 .json")
    args = ap.parse_args()

    # 定位患者目录
    root = Path(args.root)
    patient_root = root / args.category / args.patient
    if not patient_root.exists():
        matches = [p for p in (root / args.category).rglob(args.patient) if p.is_dir()]
        if not matches:
            raise FileNotFoundError(f"未找到患者目录：{patient_root}")
        patient_root = sorted(matches, key=lambda p: (len(p.parts), str(p)), reverse=True)[0]

    # 收集候选并可选列出
    candidates = _gather_series_candidates(patient_root)
    if not candidates:
        raise RuntimeError(f"目录 {patient_root} 下未发现任何可读的 DICOM Series。")

    if args.list:
        _print_candidates(candidates)
        return

    # 选择并加载
    cand = _choose_candidate(candidates, strategy=args.choose, index=args.index)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(cand.file_names)
    img = reader.Execute()

    vol_HU = sitk.GetArrayFromImage(img)
    sx, sy, sz = img.GetSpacing()
    spacing_dzyx = (float(sz), float(sy), float(sx))

    print("\n=== Volume_HU ===")
    print("shape (Z,Y,X):", vol_HU.shape)
    print("spacing (dz,dy,dx) [mm]:", spacing_dzyx)
    print("value range (HU)       :", int(vol_HU.min()), "→", int(vol_HU.max()))
    print("dtype:", vol_HU.dtype)

    meta = {
        "patient_root": str(patient_root),
        "chosen_dir": str(cand.dir_path),
        "series_id": cand.series_id,
        "num_files": len(cand.file_names),
        "choose_strategy": args.choose,
        "shape_zyx": tuple(map(int, vol_HU.shape)),
        "spacing_dzyx_mm": spacing_dzyx,
        "dtype": str(vol_HU.dtype),
    }

    # 保存
    if args.save_npy:
        out_npy = Path(args.save_npy)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, vol_HU)
        print(f"[SAVE] Volumes(HU) → {out_npy}")

    if args.save_json:
        out_json = Path(args.save_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Meta Data(JSON) → {out_json}")

if __name__ == "__main__":
    main()
