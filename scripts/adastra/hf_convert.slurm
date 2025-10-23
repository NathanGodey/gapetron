#!/bin/bash
#SBATCH --account=c1615138
#SBATCH --job-name="ckpt_convert"
#SBATCH --constraint=GENOA
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --signal=SIGUSR1@60

module purge

module load cpe/24.07
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-gnu
module load cray-python/3.11.5
module load amd-mixed/6.2.1
module load aws-ofi-rccl

source train-venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/lus/work/CT10/cad14911/ngodey/data/hf_cache
export HF_HOME=/lus/work/CT10/cad14911/ngodey/data/hf_cache

export MPICH_GPU_SUPPORT_ENABLED=1


srun --ntasks-per-node 1 --cpus-per-task 96 --cpu-bind cores \
python hf_convert.py \
--ckpt_path /lus/work/CT10/c1615138/SHARED/exp_checkpoints/llama1b-gap1-exp-mixed_2653348/requeue.ckpt \
--save_path /lus/work/CT10/c1615138/SHARED/exp_checkpoints/llama1b-gap1-exp-mixed_2653348/10k.hf \
--model_config configs/models/1B/llama.json \
--hf_tokenizer meta-llama/Meta-Llama-3-8B