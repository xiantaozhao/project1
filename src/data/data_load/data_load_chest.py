"""
data_load_chest.py — Utilities for loading LIDC-IDRI chest imaging and annotations.

Usage (command line):
    # Load imaging data
    python src/data/data_load_chest.py --subject 1 --modality CT

    # Optional arguments:
    #   --choose {largest,first,index}    Strategy for selecting a series (default: largest)
    #   --index N                         Index to use if choose=index

    Example:
        python src/data/data_load_chest.py --subject 1 --modality CT --choose first

As a module:
    from src.data import data_load_chest

    # Load CT volume
    vol, meta, spacing = data_load_chest.load_data_chest("1", "CT")

    # Load XML annotations
    anns = data_load_chest.load_xml_info("1", "CT")
    print(len(anns.nodules), len(anns.non_nodules))
"""


from __future__ import annotations
import os, csv
import argparse, json
from statistics import median
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import SimpleITK as sitk
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Iterable, Set, Literal

# 全局关闭ITK级别warning显示（可再配合stderr重定向）
try:
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
except Exception:
    pass



# 设置默认相对路径（相对于当前文件 data_load_chest.py）
META_CSV = os.path.join(
    os.path.dirname(__file__),
    "../../../data/raw/chest/manifest-1600709154662/metadata.csv"
)
BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../../data/raw/chest/manifest-1600709154662"
)



@dataclass
class Roi:
    """单个结节 ROI（一个切片上的一圈轮廓）"""
    z: float
    sop_uid: str
    inclusion: Optional[bool]
    contour: List[Tuple[float, float]]  # [(x,y), ...]

@dataclass
class NoduleAnn:
    nodule_id: str                       # 注意：可能是 "0"、"Nodule 001"、"MI014_..." 等
    diameter_mm: Optional[float]
    malignancy: Optional[int]
    characteristics: dict                # 其它评分，如 subtlety, margin, spiculation...
    rois: List[Roi]                      # 该结节跨多个切片的所有 ROI
    xml_file: str

@dataclass
class NonNoduleAnn:
    non_nodule_id: str
    sop_uid: str
    x: float
    y: float
    z: float
    xml_file: str

@dataclass
class XmlAnnotations:
    nodules: List[NoduleAnn] = field(default_factory=list)
    non_nodules: List[NonNoduleAnn] = field(default_factory=list)


# ---------------------------
# 工具函数
# ---------------------------

def list_modalities_csv(meta_csv: str = META_CSV):
    cnt = Counter()
    with open(meta_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = (row.get("Modality") or "").strip()
            if m: cnt[m] += 1
    return sorted(cnt.keys()), cnt


def _norm_subject_id(subject_id: str | int) -> str:
    sid = str(subject_id)
    if not sid.startswith("LIDC-IDRI-"):
        sid = f"LIDC-IDRI-{int(sid):04d}"
    return sid

def _strip_ns(elem: ET.Element) -> None:
    """原地去掉命名空间前缀，便于用简洁的 tag 名称检索。"""
    if '}' in elem.tag:
        elem.tag = elem.tag.split('}', 1)[1]
    for child in list(elem):
        _strip_ns(child)
        

def _txt(parent: ET.Element, tag: str) -> Optional[str]:
    e = parent.find(tag)
    if e is None or e.text is None:
        return None
    return e.text.strip()

def _req_txt(parent: ET.Element, tag: str) -> str:
    t = _txt(parent, tag)
    if not t:
        raise ValueError(f"Missing or empty <{tag}> under <{parent.tag}>")
    return t

def _opt_float(parent: ET.Element, tag: str) -> Optional[float]:
    t = _txt(parent, tag)
    return float(t) if t not in (None, "") else None

def _req_float(parent: ET.Element, tag: str) -> float:
    t = _req_txt(parent, tag)
    return float(t)



def _parse_characteristics(node: Optional[ET.Element]) -> dict:
    """把 characteristics 下的评分打包为 dict[int/float]；node 可能为 None。"""
    out = {}
    if node is None:
        return out
    keys = [
        "subtlety", "internalStructure", "calcification",
        "sphericity", "margin", "lobulation", "spiculation",
        "texture", "malignancy", "diameter"
    ]
    for k in keys:
        t = _txt(node, k)
        if t and t.replace('.', '', 1).isdigit():
            out[k] = float(t) if k == "diameter" else int(float(t))
        elif t is not None and k == "diameter":
            try:
                out[k] = float(t)
            except Exception:
                pass
    return out

def _opt_float_from_elem(elem: Optional[ET.Element]) -> Optional[float]:
    """安全地把 <elem>.text 转成 float；若无或空则返回 None。"""
    if elem is None:
        return None
    txt = elem.text
    if txt is None:
        return None
    txt = txt.strip()
    if txt == "":
        return None
    return float(txt)

def _req_float_text(parent: ET.Element, tag: str) -> float:
    """必须存在且可转 float，否则抛错（用于 z/x/y 这类必有字段）"""
    txt = parent.findtext(tag)
    if txt is None or txt.strip() == "":
        raise ValueError(f"Missing or empty <{tag}> under <{parent.tag}>")
    return float(txt)

def _req_str_text(parent: ET.Element, tag: str) -> str:
    txt = parent.findtext(tag)
    if txt is None or txt.strip() == "":
        raise ValueError(f"Missing or empty <{tag}> under <{parent.tag}>")
    return txt.strip()


def _subjects_by_modality(meta_csv: str = META_CSV) -> dict[str, list[str]]:
    """
    从 manifest 的 metadata.csv 中按模态收集唯一 Subject ID，保持出现顺序。
    返回: {'CT':[LIDC-IDRI-0001,...], 'DX':[...], 'CR':[...]}
    """
    seen = {"CT": set(), "DX": set(), "CR": set()}
    lists = {"CT": [], "DX": [], "CR": []}
    with open(meta_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mod = (row.get("Modality") or "").strip()
            sid = (row.get("Subject ID") or "").strip()
            if mod in lists and sid and sid not in seen[mod]:
                seen[mod].add(sid)
                lists[mod].append(sid)
    return lists

def _summarize_numbers(values):
    """返回字典：count/min/median/max（列表为空则返回 None）"""
    if not values:
        return None
    return {
        "count": len(values),
        "min": float(min(values)),
        "median": float(median(values)),
        "max": float(max(values)),
    }

def _fmt_tuple3_stats(tuples):
    """
    对 (dz,dy,dx) 列表分别统计 min/median/max。
    返回 {'dz':{...}, 'dy':{...}, 'dx':{...}}
    """
    if not tuples:
        return None
    dzs = [t[0] for t in tuples]
    dys = [t[1] for t in tuples]
    dxs = [t[2] for t in tuples]
    return {
        "dz_mm": _summarize_numbers(dzs),
        "dy_mm": _summarize_numbers(dys),
        "dx_mm": _summarize_numbers(dxs),
    }
    
def _fmt_shapes_stats(shapes):
    """
    对 (Z,Y,X) 列表统计 min/median/max（逐维）。
    """
    if not shapes:
        return None
    Zs = [s[0] for s in shapes]
    Ys = [s[1] for s in shapes]
    Xs = [s[2] for s in shapes]
    return {
        "Z": _summarize_numbers(Zs),
        "Y": _summarize_numbers(Ys),
        "X": _summarize_numbers(Xs),
        "examples": [tuple(map(int, s)) for s in shapes[:5]],
    }

def _count_dicoms_in_dir(series_dir: str | Path) -> int:
    """仅用于选择策略的轻量计数；不加载影像。"""
    p = Path(series_dir)
    return sum(1 for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".dcm")

def _list_xmls(series_dir: Path) -> List[Path]:
    return sorted(series_dir.glob("*.xml"))

def _sitk_read_series(series_dir: Path) -> tuple[sitk.Image, List[str], Dict[str, str]]:
    """
    用 SimpleITK 读取该目录下的一个 DICOM Series。
    返回：sitk.Image、文件名列表（按序）、若干常用标签（从第一个切片抓取）。
    """
    reader = sitk.ImageSeriesReader()
    # 开启元数据抓取
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    # 目录可能只有一个 Series，也可能多个；这里取默认第一个 Series ID
    series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
    if not series_ids:
        raise RuntimeError(f"未在目录中发现 DICOM Series: {series_dir}")

    # 取最常见的“唯一一个”或第一个；如需更严格可扩展按文件数最大者
    file_names = reader.GetGDCMSeriesFileNames(str(series_dir), series_ids[0])
    reader.SetFileNames(file_names)
    img = reader.Execute()

    # 常用标签从第 0 张切片取
    tag = lambda t: reader.GetMetaData(0, t) if reader.HasMetaDataKey(0, t) else ""
    tags = {
        "PatientID": tag("0010|0020"),
        "StudyInstanceUID": tag("0020|000d"),
        "SeriesInstanceUID": tag("0020|000e"),
        "Modality": tag("0008|0060"),
        "SeriesDescription": tag("0008|103e"),
        "Manufacturer": tag("0008|0070"),
        "KVP": tag("0018|0060"),
        "PixelSpacing": tag("0028|0030"),        # 'dy\dx'
        "SliceThickness": tag("0018|0050"),
        "SpacingBetweenSlices": tag("0018|0088"),
        "RescaleIntercept": tag("0028|1052"),
        "RescaleSlope": tag("0028|1053"),
    }
    return img, file_names, tags

def _get_spacing_dzyx(img: sitk.Image) -> Tuple[float, float, float]:
    """
    SimpleITK 的 spacing 返回 (dx, dy, dz)，这里换成 (dz, dy, dx)。
    """
    sx, sy, sz = img.GetSpacing()
    return (float(sz), float(sy), float(sx))

def _to_numpy_zyx(img: sitk.Image) -> np.ndarray:
    """
    SimpleITK -> numpy，坐标系按医疗常用返回 (Z,Y,X)。
    """
    arr = sitk.GetArrayFromImage(img)  # shape: (slices, rows, cols) = (Z, Y, X)
    # dtype 保持原样（CT 通常 int16 HU）
    return arr

def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    """
    DX/CR 可能是 2D，统一成 (Z,Y,X)；若是 2D (Y,X)，则扩为 (1,Y,X)。
    """
    if arr.ndim == 2:
        return arr[None, ...]
    return arr

def _normalize_subject_id(subject_id) -> str:
    """
    接受 '1' / '0001' / 1 / 'LIDC-IDRI-0001'，统一成 'LIDC-IDRI-0001'
    """
    s = str(subject_id)
    return s if s.startswith("LIDC-IDRI-") else f"LIDC-IDRI-{s.zfill(4)}"

def _join_from_manifest_base(relative_path: str) -> str:
    """
    将 metadata.csv 中的 File Location 转为规范的相对路径
    """
    rel = (relative_path or "").lstrip(".\\/").replace("\\", os.sep).replace("/", os.sep)
    return os.path.normpath(os.path.join(BASE_DIR, rel))


def _iter_unique_xmls(series_dirs: Iterable[Path]) -> list[Path]:
    """
    在每个 series 目录下寻找 XML；若 series 目录没有，
    回退到上级 study 目录里找；最后去重。
    """
    seen: Set[str] = set()
    xmls: list[Path] = []
    for p in series_dirs:
        # 1) 在 series 目录及其子目录找
        for x in p.glob("**/*.xml"):
            key = str(x.resolve())
            if key not in seen:
                seen.add(key)
                xmls.append(x)

        # 2) 回退到上级（study）目录找一遍
        if p.parent.is_dir():
            for x in p.parent.glob("**/*.xml"):
                key = str(x.resolve())
                if key not in seen:
                    seen.add(key)
                    xmls.append(x)

    return xmls

def get_patient_paths(subject_id, modality: str = "CT", meta_csv: str = META_CSV, base_dir: str = BASE_DIR):
    """
    根据病人 ID 和 Modality 返回本地序列路径列表。

    参数:
    --------
    subject_id : str 或 int
        病人 ID，可以是:
        - "0001" 或 "1" (函数会自动转成 "LIDC-IDRI-0001")
        - 已经完整的 "LIDC-IDRI-0001"
    modality : str
        想要的影像类型，例如 "CT"、"DX"、"CR"
    meta_csv : str, 可选
        metadata.csv 文件路径，默认指向 chest 数据的 metadata.csv
    base_dir : str, 可选
        数据根目录，默认指向 chest 的 manifest 文件夹

    返回:
    --------
    list[str]
        满足条件的序列路径列表（相对路径）。
    """
    subj = _normalize_subject_id(subject_id)
    out = []
    with open(meta_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("Subject ID") or "").strip()
            mod = (row.get("Modality") or "").strip()
            if sid == subj and mod == modality:
                loc = (row.get("File Location") or "").strip()
                if loc:
                    out.append(_join_from_manifest_base(loc))
    return out

def load_xml_info(subject_id: str | int,
                  modality: Literal["CT", "DX", "CR"] = "CT",
                  verbose: bool = False) -> XmlAnnotations:
    """
    解析 LIDC-IDRI 病人的 XML 标注，返回结节与非结节的结构化结果。

    Parameters
    ----------
    subject_id : str|int
    modality   : {'CT','DX','CR'}  # 仅用于沿用你的 get_patient_paths 接口
    verbose    : bool              # 打印调试信息

    Returns
    -------
    XmlAnnotations
        - nodules: List[NoduleAnn]
        - non_nodules: List[NonNoduleAnn]
    """
    # 通过项目内的路径解析方法拿到该病人的实际 series 路径
    series_dirs: List[Path] = [Path(p) for p in get_patient_paths(subject_id, modality)]
    if verbose:
        print(f"[load_xml_info] subject={subject_id}  series_dirs={len(series_dirs)}")
        for d in series_dirs[:5]:
            print("  -", d)

    xml_files = _iter_unique_xmls(series_dirs)
    if verbose:
        print(f"[load_xml_info] found XML files: {len(xml_files)}")
        for x in xml_files[:5]:
            print("  *", x)

    nods: List[NoduleAnn] = []
    nons: List[NonNoduleAnn] = []

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            _strip_ns(root)  # ⚠️ 去掉默认命名空间，后续才能用简洁的 tag
        except Exception as e:
            if verbose:
                print(f"[load_xml_info] Skip broken XML: {xml_file} ({e})")
            continue

        # ----- 结节 -----
        for nodule in root.findall(".//unblindedReadNodule"):
            try:
                nodule_id = _req_txt(nodule, "noduleID")

                # characteristics（含 diameter, malignancy 等）
                chars_node = nodule.find("characteristics")
                chars = _parse_characteristics(chars_node)
                diameter = chars.get("diameter", None)
                malignancy = chars.get("malignancy", None)
                if isinstance(diameter, (int, float)):
                    diameter_mm: Optional[float] = float(diameter)
                else:
                    diameter_mm = None
                if isinstance(malignancy, (int, float)):
                    malignancy_i: Optional[int] = int(malignancy)
                else:
                    malignancy_i = None

                rois: List[Roi] = []
                for roi in nodule.findall("roi"):
                    z = _req_float(roi, "imageZposition")
                    sop_uid = _req_txt(roi, "imageSOP_UID")
                    incl_txt = _txt(roi, "inclusion")
                    inclusion = None
                    if incl_txt:
                        t = incl_txt.strip().upper()
                        inclusion = True if t == "TRUE" else (False if t == "FALSE" else None)

                    contour: List[Tuple[float, float]] = []
                    for em in roi.findall("edgeMap"):
                        xt = _txt(em, "xCoord")
                        yt = _txt(em, "yCoord")
                        if xt and yt:
                            contour.append((float(xt), float(yt)))

                    rois.append(Roi(z=z, sop_uid=sop_uid, inclusion=inclusion, contour=contour))

                nods.append(NoduleAnn(
                    nodule_id=nodule_id,
                    diameter_mm=diameter_mm,
                    malignancy=malignancy_i,
                    characteristics={k: v for k, v in chars.items() if k not in ("diameter", "malignancy")},
                    rois=rois,
                    xml_file=str(xml_file)
                ))

            except Exception as e:
                if verbose:
                    print(f"[load_xml_info] Skip nodule in {xml_file}: {e}")

        # ----- 非结节 -----
        for nn in root.findall(".//nonNodule"):
            try:
                non_nodule_id = _req_txt(nn, "nonNoduleID")
                z = _req_float(nn, "imageZposition")
                sop_uid = _req_txt(nn, "imageSOP_UID")
                locus = nn.find("locus")
                if locus is None:
                    continue
                x = _req_float(locus, "xCoord")
                y = _req_float(locus, "yCoord")

                nons.append(NonNoduleAnn(
                    non_nodule_id=non_nodule_id,
                    sop_uid=sop_uid,
                    x=x, y=y, z=z,
                    xml_file=str(xml_file)
                ))
            except Exception as e:
                if verbose:
                    print(f"[load_xml_info] Skip nonNodule in {xml_file}: {e}")

    if verbose:
        print(f"[load_xml_info] parsed nodules={len(nods)}  non_nodules={len(nons)}")

    return XmlAnnotations(nodules=nods, non_nodules=nons)

def load_series_from_path(series_path: str | Path, modality: Optional[str] = None):
    """
    读取单个“序列目录”中的数据。
    - CT: 返回 (vol_HU[Z,Y,X], spacing_dzyx(mm), meta)
    - DX/CR: 返回 (vol[Z=1,Y,X], spacing_dzyx，其中 dz=1.0), meta
    - XML: 不直接加载为影像；仅把 xml 路径加入 meta['xml_paths']

    参数
    ----
    series_path : str|Path
        单个 DICOM 序列所在目录（里面通常是若干 .dcm；也可能混有 .xml）
    modality : Optional[str]
        指定/提示该序列的 Modality（"CT"/"DX"/"CR"）。若为空将自动从 DICOM 读取。

    返回
    ----
    vol_zyx : np.ndarray
        体数据 (Z,Y,X)。CT 为 HU；DX/CR 为像素值（通常已为灰度，未做窗宽窗位）。
    spacing_dzyx : (dz, dy, dx)
        体素间距（mm）。DX/CR 无 z 维概念，则 dz=1.0。
    meta : dict
        元数据（路径、文件数、UID、标签、是否含XML、Modality 等）。
    """
    series_dir = Path(series_path)
    if not series_dir.is_dir():
        raise NotADirectoryError(f"不是有效目录: {series_dir}")

    # 先尝试作为 DICOM 序列读取（CT/DX/CR 都适用）
    img, file_names, tags = _sitk_read_series(series_dir)

    # 自动识别 Modality（若未显式传入）
    detected_mod = (tags.get("Modality") or "").upper().strip()
    if modality is None:
        modality = detected_mod
    else:
        modality = modality.upper().strip()

    # 影像转 numpy & spacing
    vol_zyx = _to_numpy_zyx(img)
    spacing_dzyx = _get_spacing_dzyx(img)

    # 针对不同 Modality 做轻微归一：
    if modality == "CT":
        # vol_zyx 已是 HU（SimpleITK 会使用 RescaleSlope/Intercept）
        pass
    elif modality in {"DX", "CR"}:
        # 2D -> 3D，设置 dz=1.0（占位，不代表真实厚度）
        vol_zyx = _ensure_3d(vol_zyx)
        spacing_dzyx = (1.0, spacing_dzyx[1], spacing_dzyx[2])
    else:
        # 其他模态：仍按读到的数据返回（统一 3D 形状）
        vol_zyx = _ensure_3d(vol_zyx)

    # XML 附带
    xml_paths = [str(p) for p in _list_xmls(series_dir)]

    meta = {
        "series_dir": str(series_dir),
        "num_files": len(file_names),
        "first_file": file_names[0] if file_names else "",
        "patient_id": tags.get("PatientID", ""),
        "study_uid": tags.get("StudyInstanceUID", ""),
        "series_uid": tags.get("SeriesInstanceUID", ""),
        "series_description": tags.get("SeriesDescription", ""),
        "manufacturer": tags.get("Manufacturer", ""),
        "modality_detected": detected_mod,
        "modality_used": modality,
        "pixel_spacing_tag": tags.get("PixelSpacing", ""),
        "slice_thickness_tag": tags.get("SliceThickness", ""),
        "spacing_between_slices_tag": tags.get("SpacingBetweenSlices", ""),
        "rescale_slope": tags.get("RescaleSlope", ""),
        "rescale_intercept": tags.get("RescaleIntercept", ""),
        "shape_zyx": tuple(int(x) for x in vol_zyx.shape),
        "dtype": str(vol_zyx.dtype),
        "xml_paths": xml_paths,
    }
    return vol_zyx, spacing_dzyx, meta



# ---------------------------
# 对外主入口
# ---------------------------

def load_data_chest(
    subject_id,
    modality: Literal["CT", "DX", "CR"],
    choose: Literal["largest", "first", "index"] = "largest",
    index: Optional[int] = None,
    meta_csv: str = META_CSV,
    base_dir: str = BASE_DIR,
):
    """
    输入病人ID与Modality，找到该病人所有匹配的序列目录，并按策略挑一个读取。

    参数
    ----
    subject_id : str|int
        '1' / '0001' / 1 / 'LIDC-IDRI-0001' 均可
    modality : {'CT','DX','CR'}
        目标模态
    choose : {'largest','first','index'}, 默认 'largest'
        - 'largest': 选择该病人该模态下“DICOM数量最多”的序列
        - 'first'  : 选择列表第一个
        - 'index'  : 使用 index 指定的下标（从 0 开始）
    index : Optional[int]
        当 choose='index' 时有效
    meta_csv, base_dir : str
        清单与根目录（已默认为相对路径）

    返回
    ----
    vol_zyx : np.ndarray
        CT 为 HU；DX/CR 为像素值，形状统一为 (Z,Y,X)
    spacing_dzyx : tuple[float,float,float]
        (dz, dy, dx) in mm；DX/CR 的 dz 为 1.0（占位）
    meta : dict
        汇总元数据：所选路径、候选列表、选择策略、DICOM计数、DICOM/XML信息等
    """
    # 1) 找到该病人对应模态的所有 series 目录（相对路径）
    series_paths = get_patient_paths(subject_id, modality, meta_csv=meta_csv, base_dir=base_dir)
    if not series_paths:
        raise FileNotFoundError(f"未找到 {subject_id} 的 {modality} 序列。")

    # 2) 根据策略挑选
    if choose == "first":
        chosen_path = series_paths[0]
    elif choose == "index":
        if index is None or not (0 <= index < len(series_paths)):
            raise IndexError(f"choose='index' 需要 index ∈ [0, {len(series_paths)-1}]，收到 {index}")
        chosen_path = series_paths[index]
    else:  # 'largest'
        counts = [(_count_dicoms_in_dir(p), p) for p in series_paths]
        counts.sort(key=lambda t: (t[0], str(t[1])), reverse=True)
        chosen_path = counts[0][1]

    # 3) 真正读取该序列
    vol_zyx, spacing_dzyx, meta = load_series_from_path(chosen_path, modality=modality)

    # 4) 丰富 meta
    meta.update({
        "subject_id_input": str(subject_id),
        "modality_input": modality,
        "choose_strategy": choose,
        "choose_index": index,
        "candidate_series": list(map(str, series_paths)),
        "chosen_series": str(chosen_path),
        "candidate_counts": {str(p): _count_dicoms_in_dir(p) for p in series_paths},
    })
    return vol_zyx, spacing_dzyx, meta


# ---------------------------
# 命令行接口
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="读取每种模态(CT/DX/CR)前 N 个不同病人的序列，并汇总统计"
    )
    ap.add_argument("--modalities", type=str, default="CT,DX,CR",
                    help="逗号分隔的模态列表，例如 CT 或 CT,DX,CR（默认：CT,DX,CR）")
    ap.add_argument("--limit", type=int, default=10,
                    help="每种模态读取的不同病人数量上限（默认 10）")
    ap.add_argument("--choose", type=str, default="largest",
                    choices=["largest", "first", "index"],
                    help="候选序列选择策略（默认 largest）")
    ap.add_argument("--save-json", type=str, default=None,
                    help="将汇总结果保存为 JSON 文件路径")
    ap.add_argument("--dry-run", action="store_true",
                    help="只列出将要处理的病人，不实际加载")
    args = ap.parse_args()

    wanted = [m.strip().upper() for m in args.modalities.split(",") if m.strip()]
    wanted = [m for m in wanted if m in {"CT", "DX", "CR"}]
    if not wanted:
        raise ValueError("modalities 至少需要包含 CT/DX/CR 之一")

    subj_map = _subjects_by_modality(META_CSV)

    summary = {"limit": args.limit, "choose": args.choose, "results": {}}

    for mod in wanted:
        subjects = subj_map.get(mod, [])[: args.limit]
        print(f"\n=== Modality: {mod} | subjects to try: {len(subjects)} ===")
        if args.dry_run:
            for s in subjects:
                print(" -", s)
            summary["results"][mod] = {"planned_subjects": subjects}
            continue

        loaded = 0
        shapes = []
        spacings = []
        failures = []

        for sid in subjects:
            try:
                vol, spacing, meta = load_data_chest(sid, mod, choose=args.choose)
                shapes.append(tuple(int(x) for x in vol.shape))
                spacings.append(tuple(float(x) for x in spacing))
                loaded += 1
                # 简要打印一行
                print(f"[OK] {sid:>14}  shape={vol.shape}  spacing={spacing}")
            except Exception as e:
                failures.append({"subject": sid, "error": str(e)})
                print(f"[FAIL] {sid:>14}  {e}")

        summary["results"][mod] = {
            "attempted_subjects": subjects,
            "loaded_count": loaded,
            "failed_count": len(failures),
            "shape_stats": _fmt_shapes_stats(shapes),
            "spacing_stats_mm": _fmt_tuple3_stats(spacings),
            "failures": failures[:10],  # 只保留前 10 条失败原因以避免过长
        }

        # 控制台总结
        print(f"\n--- Summary ({mod}) ---")
        print("Loaded:", loaded, " / ", len(subjects))
        if shapes:
            ss = summary["results"][mod]["shape_stats"]
            print("Z   min/med/max:", ss["Z"]["min"], ss["Z"]["median"], ss["Z"]["max"])
            print("Y   min/med/max:", ss["Y"]["min"], ss["Y"]["median"], ss["Y"]["max"])
            print("X   min/med/max:", ss["X"]["min"], ss["X"]["median"], ss["X"]["max"])
        if spacings:
            sp = summary["results"][mod]["spacing_stats_mm"]
            print("dz  min/med/max:", sp["dz_mm"]["min"], sp["dz_mm"]["median"], sp["dz_mm"]["max"], "mm")
            print("dy  min/med/max:", sp["dy_mm"]["min"], sp["dy_mm"]["median"], sp["dy_mm"]["max"], "mm")
            print("dx  min/med/max:", sp["dx_mm"]["min"], sp["dx_mm"]["median"], sp["dx_mm"]["max"], "mm")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[SAVE] Summary JSON → {out}")

if __name__ == "__main__":
    main()