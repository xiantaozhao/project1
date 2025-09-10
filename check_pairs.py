# -*- coding: utf-8 -*-
import re, json, csv
from pathlib import Path
from datetime import datetime
import numpy as np
from statistics import fmean, median
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ==== 路径配置 ====
ROOT = Path("figs/projections/chest/projections")      # ← 改成你的输入目录（含 deg_*.png）
LOG_BASE = Path("logs/symmertry_comparsion")           # 按你给的拼写
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = LOG_BASE / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RUN_DIR / "pair_metrics.csv"
OUT_JSON = RUN_DIR / "summary.json"
DIFF_DIR = RUN_DIR / "diffs"

FORCE_FLIP = "lr"
SAVE_DIFF_IMAGE = True

# ==== 工具函数 ====
deg_pat = re.compile(r"deg_(\d{1,3})\.png$", re.IGNORECASE)

def list_deg_images(root: Path):
    files = {}
    for p in root.glob("deg_*.png"):
        m = deg_pat.search(p.name)
        if m:
            d = int(m.group(1)) % 360
            files[d] = p
    return files

def apply_flip(x, mode: str):
    if mode == "lr":   return x[:, ::-1]
    if mode == "ud":   return x[::-1, :]
    if mode == "both": return x[::-1, ::-1]
    return x  # 'none' or unknown


def load_gray01(p: Path):
    im = Image.open(p)
    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32)
        mx, mn = float(arr.max()), float(arr.min())
        if mx > 1.0 or mn < 0.0:
            arr = (arr - mn) / (mx - mn + 1e-8)
    return arr

def common_crop(a, b):
    def bbox_of_nonbg(x):
        bg = np.mean([x[0,0], x[0,-1], x[-1,0], x[-1,-1]])
        mask = np.abs(x - bg) > 1e-6
        if not mask.any():
            return (0, x.shape[0], 0, x.shape[1])
        ys, xs = np.where(mask)
        return (ys.min(), ys.max()+1, xs.min(), xs.max()+1)
    y0a, y1a, x0a, x1a = bbox_of_nonbg(a)
    y0b, y1b, x0b, x1b = bbox_of_nonbg(b)
    y0, y1 = max(y0a, y0b), min(y1a, y1b)
    x0, x1 = max(x0a, x0b), min(x1a, x1b)
    if y0 >= y1 or x0 >= x1:
        return a, b
    return a[y0:y1, x0:x1], b[y0:y1, x0:x1]

def center_crop_to_min(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    def cc(x):
        y0 = (x.shape[0]-h)//2; x0 = (x.shape[1]-w)//2
        return x[y0:y0+h, x0:x0+w]
    return cc(a), cc(b)

def psnr_from_mse(mse, data_range=1.0):
    if mse <= 1e-12:
        return float('inf')
    return 10.0 * np.log10((data_range**2) / mse)

def try_best_flip(a, b):
    """在不翻/左右翻/上下翻/双翻中选 MAE 最小的匹配。"""
    candidates = [
        ("none", b),
        ("flip_lr", b[:, ::-1]),
        ("flip_ud", b[::-1, :]),
        ("flip_both", b[::-1, ::-1]),
    ]
    best = None
    for tag, bb in candidates:
        aa, bb2 = common_crop(a, bb)
        aa, bb2 = center_crop_to_min(aa, bb2)
        mae_val = float(np.mean(np.abs(aa - bb2)))
        if (best is None) or (mae_val < best[1]):
            best = (tag, mae_val, aa, bb2)
    tag, _, aa_best, bb_best = best
    return aa_best, bb_best, tag

def metrics(a, b):
    mae = float(np.mean(np.abs(a-b)))
    mse = float(np.mean((a-b)**2))
    ps = float(psnr_from_mse(mse, data_range=1.0))
    ssim_v = float(ssim(a, b, data_range=1.0))
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0,1])
    return mae, mse, ps, ssim_v, corr

def save_pair_side_by_side(a, b, out_path: Path, gap: int = 6):
    """
    将两张 [0,1] 灰度图 a、b 并排拼接成一张图后保存为 [PNG]。
    中间用白色竖线分隔。
    """
    h, w = a.shape
    g = max(1, int(gap))
    a8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    b8 = (np.clip(b, 0, 1) * 255).astype(np.uint8)
    canvas = np.zeros((h, w*2 + g), dtype=np.uint8)
    canvas[:, :w] = a8
    canvas[:, w:w+g] = 255     # 中间分隔
    canvas[:, w+g:] = b8
    Image.fromarray(canvas, mode="L").save(out_path)


# 目标列
COLS = ["deg_a","deg_b","found_a","found_b","h","w","flip_used","MAE","MSE","PSNR","SSIM","Pearson"]

def as_scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0]) if v.size == 1 else None
    return v

# ==== 主流程（无打印） ====
files = list_deg_images(ROOT)
pairs_done = set()
rows = []

all_degs = sorted(files.keys())
for d in all_degs:
    d2 = (d + 180) % 360
    pair_key = tuple(sorted((d, d2)))
    if pair_key in pairs_done:
        continue
    pairs_done.add(pair_key)
    p1, p2 = files.get(d), files.get(d2)
    if p1 is None or p2 is None:
        rows.append(dict(deg_a=d, deg_b=d2, found_a=p1 is not None, found_b=p2 is not None))
        continue

    A = load_gray01(p1)
    B = load_gray01(p2)

    # —— 强制镜像 —— #
    if FORCE_FLIP == "auto":
        a2, b2, flip_tag = try_best_flip(A, B)
    else:
        B  = apply_flip(B, FORCE_FLIP)          # 这里就是强制 'lr'
        flip_tag = "flip_" + FORCE_FLIP if FORCE_FLIP != "none" else "none"
        a2, b2 = common_crop(A, B)
        a2, b2 = center_crop_to_min(a2, b2)
    # ———————————— #

    mae, mse, psnr_v, ssim_v, corr = metrics(a2, b2)

    if SAVE_DIFF_IMAGE:
        DIFF_DIR.mkdir(parents=True, exist_ok=True)
        out = DIFF_DIR / f"pair_deg_{d:03d}_vs_{d2:03d}_{flip_tag}.png"
        save_pair_side_by_side(a2, b2, out, gap=6)


    rows.append(dict(
        deg_a=d, deg_b=d2, found_a=True, found_b=True,
        h=int(a2.shape[0]), w=int(a2.shape[1]),
        flip_used=flip_tag, MAE=mae, MSE=mse, PSNR=psnr_v, SSIM=ssim_v, Pearson=corr
    ))

# === 构造 clean（全部转 Python 标量），并按 deg_a 排序 ===
clean = []
for r in rows:
    rr = {k: as_scalar(r.get(k, None)) for k in COLS}
    # 基本类型收敛
    for k in ("deg_a","deg_b","h","w"):
        if rr[k] is not None:
            try: rr[k] = int(rr[k])
            except: rr[k] = None
    if rr["flip_used"] is not None:
        rr["flip_used"] = str(rr["flip_used"])
    for k in ("MAE","MSE","PSNR","SSIM","Pearson"):
        if rr[k] is not None:
            try: rr[k] = float(rr[k])
            except: rr[k] = None
    for k in ("found_a","found_b"):
        rr[k] = bool(rr[k]) if rr[k] is not None else False
    clean.append(rr)

clean.sort(key=lambda x: (x["deg_a"] is None, x["deg_a"] if x["deg_a"] is not None else 0))

# === 写 CSV（标准库 csv）===
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=COLS)
    writer.writeheader()
    for rr in clean:
        writer.writerow(rr)

# === 写 summary.json（标准库 json）===
ok_flags = [ (rr["found_a"] and rr["found_b"]) for rr in clean ]
ssim_vals = [ rr["SSIM"] for rr, ok in zip(clean, ok_flags) if ok and (rr["SSIM"] is not None) ]
mae_vals  = [ rr["MAE"]  for rr, ok in zip(clean, ok_flags) if ok and (rr["MAE"]  is not None) ]

summary = {
    "run_id": RUN_ID,
    "input_dir": str(ROOT.resolve()),
    "num_pairs_compared": int(sum(ok_flags)),
    "mean_SSIM": (float(fmean(ssim_vals)) if len(ssim_vals) else None),
    "median_MAE": (float(median(mae_vals)) if len(mae_vals) else None),
    "output_csv": str(OUT_CSV.resolve()),
    "diff_dir": str(DIFF_DIR.resolve()) if SAVE_DIFF_IMAGE else None,
}
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)
