#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/ddpm/chest.yaml")


def _add_repo_root_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_add_repo_root_to_syspath()

from src.configs.configloading import load_config  # noqa: E402
from src.train.train_ddpm import train_ddpm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPM on the chest dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    cfg = load_config(config_path)
    train_ddpm(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
