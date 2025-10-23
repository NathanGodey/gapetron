#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="fsdp_debug"
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=./logs/test_logs/%x_%j.out
#SBATCH --error=./logs/test_logs/%x_%j.out
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --threads-per-core=1
#SBATCH --requeue
#SBATCH --constraint h100
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --signal=SIGUSR1@120

module purge

module load arch/h100
module load python/3.11.5

source train-venv/bin/activate


export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LITDATA_OFFLINE=1
export HF_DATASETS_CACHE=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export HF_HOME=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/lustre/fsn1/projects/rech/mts/commun/compile_cache/torchinductor

# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO
export MPICH_GPU_SUPPORT_ENABLED=1

srun python tests/fsdp_basic.py