# scripts/test_unet_sampler.py
import argparse
from torch.utils.data import DataLoader
from src.configs.configloading import load_config
from src.data.dataset_unet import UnetDataset, UnetSampler

def _to_list(v):
    # 把张量/ndarray/标量都转成 Python list，便于统一处理
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="path to yaml config")
    parser.add_argument("--split", type=str, default="train", choices=["train","val","test"])
    parser.add_argument("--group-by", type=str, default="patient", choices=["patient","patient_angle"],
                        help="group batches by patient only, or (patient, angle)")
    args = parser.parse_args()

    cfg = load_config(args.cfg, default_path=None)

    ds = UnetDataset(cfg, split_role=args.split)
    print(f"[{args.split}] dataset samples: {len(ds)}")

    sampler = UnetSampler(
        ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True if args.split == "train" else False,
        drop_last=False,
        group_by=args.group_by,
    )

    dl = DataLoader(
        ds,
        batch_sampler=sampler,  # 注意：用 batch_sampler，而不是 batch_size/shuffle
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # 检查前几个 batch
    for bi, batch in enumerate(dl):
        meta = batch["meta"]                             # dict of lists
        patients = _to_list(meta["patient"])
        # 如果按 patient_angle 分组，也可以一起拿出来看看
        stops = _to_list(meta["stop_deg"])
        steps = _to_list(meta["step_deg"])

        unique_patients = sorted(set(patients))
        print(f"Batch {bi}: size={len(patients)}, patients={unique_patients[:3]}{'...' if len(unique_patients)>3 else ''}")

        # 断言：一个 batch 内只能有一个病人
        assert len(unique_patients) == 1, f"Found mixed patients in one batch: {unique_patients}"

        if args.group_by == "patient_angle":
            # 如果要求同角度，也断言 stop/step 唯一
            unique_stops = sorted(set(stops))
            unique_steps = sorted(set(steps))
            assert len(unique_stops) == 1 and len(unique_steps) == 1, \
                f"Mixed angles in one batch: stop={unique_stops}, step={unique_steps}"

        if bi >= 2:   # 只看前三个 batch
            break

if __name__ == "__main__":
    main()
