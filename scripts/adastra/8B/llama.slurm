#!/bin/bash
#SBATCH --account=c1615138
#SBATCH --job-name="gprn_l8b"
#SBATCH --constraint=MI250
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --signal=SIGUSR1@60

module purge

module load cpe/23.12
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-gnu
module load cray-python/3.11.5
module load amd-mixed/6.0.0
module load aws-ofi-rccl

source fast-venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/lus/work/CT10/cad14911/ngodey/data/hf_cache
export HF_HOME=/lus/work/CT10/cad14911/ngodey/data/hf_cache

export MPICH_GPU_SUPPORT_ENABLED=1

rocm-smi


srun --ntasks-per-node 8 --cpus-per-task 8 --cpu-bind cores \
python train_fabric.py --model_config configs/fabric/models/llama_8b.json \
--optim_config configs/fabric/opt/adamw_fused_8b.json --num_nodes 8 --grad_clip_val 1 \
--global_bs 1024 --train_bs 2 --val_bs 2 --precision "bf16-mixed" --run_name "llama8b-bs1024-n8-mix-ac" \
--dataset /lus/work/CT10/cad14911/ngodey/data/processed/fineweb_edu_full_llama3_sl2049_opt \
--strategy fsdp --saved_ckpt_path "${SCRATCHDIR}/ckpts" --ckpt_every 100 --val_check_every 100 \
--act_ckpt