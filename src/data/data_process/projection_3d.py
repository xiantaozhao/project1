"""
projection.py
--------
基于 [TIGRE] 的锥束CT前向投影工具与数据封装。

功能：
1) project_angle:  给定体数据与单个扫描角，计算对应的 2D 投影图（探测器平面）。
2) build_pickle_dataset_from_volume: 批量对 train/val 角度做前向投影，并将结果以 pickle 封装为
   便于训练的数据结构（与你现有的 TIGREDataset 兼容：含 geometry 字段、train/val 投影与角度等）。

依赖：
- tigre  (Python 版)
- numpy, pickle
- 可选：从 data.data_process 导入 hu_to_attenuation()，若传入体数据单位为 HU 时自动转换。
"""

from __future__ import annotations
from typing import Dict, Iterable, Optional
import numpy as np
import pickle
import importlib
import pkgutil



def hu_to_attenuation(arr: np.ndarray) -> np.ndarray:
    mu_water, mu_air = 0.206, 0.0004
    return mu_water + (mu_water - mu_air) / 1000.0 * arr


def _require_tigre():
    """返回 (tigre, Ax)；若 tigre 或 Ax 不可用，则抛出带解释的 RuntimeError。"""
    try:
        import tigre  # 顶层包（conda 安装）
    except Exception as e:
        raise RuntimeError("未检测到 TIGRE Python 库：请在当前环境可以 `import tigre`。") from e

    # 逐个候选路径尝试定位 Ax
    candidates = [
        "tigre.Ax",               # 顶层直挂
    ]
    Ax = tigre.Ax
    return tigre, Ax


def to_tigre_geometry(geo_dict: Dict) -> "object":
    """
    根据 geo_dict 创建 TIGRE 的 geometry 对象（单位：mm），并覆盖常用字段。

    期望键：
      DSD, DSO,
      nDetector=[nu,nv], dDetector=[du,dv],
      nVoxel=[nx,ny,nz], dVoxel=[dx,dy,dz],
      offOrigin=[ox,oy,oz], offDetector=[ou,ov],
      mode, accuracy, filter
    """
    tigre, _ = _require_tigre()
    # 先取一个默认的 geometry，再覆盖字段（官方推荐方式）
    # 注：部分打包版本 geometry_default 可能映射到 geometry() 的默认实例
    geo = tigre.geometry()

    # Distances
    geo.DSD = float(geo_dict["DSD"])                # Distance Source Detector (mm)
    geo.DSO = float(geo_dict["DSO"])                # Distance Source Origin   (mm)    

    geo.nDetector = np.asarray(geo_dict["nDetector"], dtype=np.int32)      # [U(width), V(height)] in pixels
    geo.dDetector = np.asarray(geo_dict["dDetector"], dtype=np.float32)    # pixel size (mm)
    geo.sDetector = geo.nDetector * geo.dDetector

    geo.nVoxel = np.asarray(geo_dict["nVoxel"], dtype=np.int32)            # [nx, ny, nz]
    geo.dVoxel = np.asarray(geo_dict["dVoxel"], dtype=np.float32)          # [dx, dy, dz] mm/vx
    geo.sVoxel = geo.nVoxel * geo.dVoxel                                   # detector total size (mm)

    geo.offOrigin   = np.asarray(geo_dict.get("offOrigin", [0, 0, 0]), dtype=np.float32)    # image offset (mm)
    geo.offDetector = np.asarray(geo_dict.get("offDetector", [0, 0]), dtype=np.float32)     # detector offset (mm)

    geo.mode     = geo_dict.get("mode", "cone")                 # "cone" or "parallel"
    geo.accuracy = float(geo_dict.get("accuracy", 0.5))         # accuracy (default 0.5)
    geo.filter   = geo_dict.get("filter", "ram-lak")            # "ram-lak", "shepp-logan", "cosine", "hamming", "hann", "none"

    return geo


def get_projections(volume: np.ndarray,
                  geo_dict: Dict,
                  angles_deg: np.ndarray,
                  input_unit: str = "HU") -> np.ndarray:
    """
    用 TIGRE 对单个扫描角做锥束CT前向投影，返回探测器平面的 2D 投影图（形状 [nv, nu]）。
    - volume: 3D 体数据 [nx,ny,nz]，HU 或 mu 或 grayscale
    - geo_dict: 几何（mm）
    - angle_rad: 单角度（弧度）
    - input_unit: "HU" 或 "mu"
    """
    _, Ax = _require_tigre()

    vol = volume.astype(np.float32, copy=False)
    if input_unit.lower() == "hu":
        vol = hu_to_attenuation(vol)

    geo = to_tigre_geometry(geo_dict)
    angles = np.deg2rad(angles_deg).astype(np.float32)      # [1]
    projs = Ax(vol, geo, angles)                            # [nv, nu, 1]
    return projs                                  # -> [angles, nv, nu]


def build_pickle_dataset_from_volume(volume: np.ndarray,
                                     geo_dict: Dict,
                                     angles_train: Iterable[float],
                                     angles_val: Iterable[float],
                                     output_path: str,
                                     input_unit: str = "HU",
                                     image_field: Optional[np.ndarray] = None) -> str:
    """
    批量前向投影并封装为 pickle：字段与你的 TIGRE 数据加载器兼容。
    """
    _, Ax = _require_tigre()

    vol = volume.astype(np.float32, copy=False)
    if input_unit.lower() == "hu":
        vol = hu_to_attenuation(vol)

    geo = to_tigre_geometry(geo_dict)

    angles_train = np.asarray(list(angles_train), dtype=np.float32)
    angles_val   = np.asarray(list(angles_val),   dtype=np.float32)

    projs_train = Ax(vol, geo, angles_train).astype(np.float32, copy=False)  # [nv, nu, Nt]
    projs_val   = Ax(vol, geo, angles_val).astype(np.float32, copy=False)    # [nv, nu, Nv]

    data = {
        "DSD": float(geo_dict["DSD"]),
        "DSO": float(geo_dict["DSO"]),
        "nDetector": np.asarray(geo_dict["nDetector"], dtype=np.int32),
        "dDetector": np.asarray(geo_dict["dDetector"], dtype=np.float32),
        "nVoxel": np.asarray(geo_dict["nVoxel"], dtype=np.int32),
        "dVoxel": np.asarray(geo_dict["dVoxel"], dtype=np.float32),
        "offOrigin": np.asarray(geo_dict.get("offOrigin", [0,0,0]), dtype=np.float32),
        "offDetector": np.asarray(geo_dict.get("offDetector", [0,0]), dtype=np.float32),
        "accuracy": float(geo_dict.get("accuracy", 0.5)),
        "mode": geo_dict.get("mode", "cone"),
        "filter": geo_dict.get("filter", "ram-lak"),
        "train": {"angles": angles_train, "projections": np.moveaxis(projs_train, -1, 0)},
        "val":   {"angles": angles_val,   "projections": np.moveaxis(projs_val,   -1, 0)},
        "numTrain": int(angles_train.size),
        "numVal": int(angles_val.size),
    }
    if image_field is not None:
        data["image"] = image_field.astype(np.float32, copy=False)

    from pathlib import Path as _P
    out_p = _P(output_path); out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(out_p)


if __name__ == "__main__":
    try:
        import tigre
        # print("projection.py ready; TIGRE import OK")
    except Exception as e:
        print("TIGRE import failed:", e)
