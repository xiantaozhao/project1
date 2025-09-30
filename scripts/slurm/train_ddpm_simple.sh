#!/bin/bash -l
#SBATCH --job-name=ddpm-chest-simple
#SBATCH -p Quick 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# —— 最简安全设置 —— #
set -e          # 任一命令失败即退出（不加 -u，避免激活脚本里未定义变量导致秒挂）
export PYTHONUNBUFFERED=1

# 进入提交目录 & 准备日志目录
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# 激活 conda（按 GAIVI 手册风格）
CONDA_ENV=proj
conda activate "$CONDA_ENV"

# 运行训练（不加任何参数）
python scripts/train_simple_ddpm.py

# 运行推理（仅指定一个 patient_id=1；其余走脚本默认值）
python scripts/restore_from_npz.py --patient_id 1
