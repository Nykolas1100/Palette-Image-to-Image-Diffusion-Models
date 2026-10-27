#!/bin/bash
#SBATCH --job-name=palette_xray
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NV-RTX5000:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=palette_%A.out
#SBATCH --error=palette_%A.err
#SBATCH --chdir=/home/rekasius/xray-data/Palette-Image-to-Image-Diffusion-Models
##SBATCH --mail-type=ALL
##SBATCH --mail-user=rekasius@hawaii.edu

echo "Job started on $(hostname) at $(date)"
nvidia-smi

# Use the same Python interpreter as your working notebook.
module purge
PYTHON=/opt/apps/software/lang/Anaconda3/2024.02-1/bin/python

"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Patch config paths and settings before training.
"$PYTHON" - <<'EOF'
import json

p = "config/denoise_xray.json"
root = "/home/rekasius/xray-data/dataset_npy"

with open(p) as f:
    cfg = json.load(f)

cfg["seed"] = 42
cfg["path"]["base_dir"] = "/home/rekasius/xray-data/Palette-Image-to-Image-Diffusion-Models/experiments"
cfg["datasets"]["train"]["which_dataset"]["args"]["data_root"] = f"{root}/all"
cfg["datasets"]["val"]["which_dataset"]["args"]["data_root"] = f"{root}/val"
cfg["datasets"]["train"]["dataloader"]["validation_split"] = 20
cfg["model"]["which_model"]["args"]["ema_scheduler"]["ema_decay"] = 0.999
cfg["model"]["which_model"]["args"]["optimizers"][0]["lr"] = 2e-4

cfg["train"].update({
    "n_epoch": 200,
    "val_epoch": 200,
    "save_checkpoint_epoch": 25,
    "log_iter": 976,
})

with open(p, "w") as f:
    json.dump(cfg, f, indent=4)
EOF

"$PYTHON" -u run.py -c config/denoise_xray.json -b 16 --gpu 0

echo "Job finished at $(date)"

