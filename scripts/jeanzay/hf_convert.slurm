#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="hf_conv"
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=./logs/test_logs/%x_%j.out
#SBATCH --error=./logs/test_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --threads-per-core=1
#SBATCH --requeue
#SBATCH --constraint h100
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --signal=SIGUSR1@120

module purge

module load arch/h100
module load python/3.11.5

source fa-venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LITDATA_OFFLINE=1
export HF_DATASETS_CACHE=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export HF_HOME=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/lustre/fsn1/projects/rech/mts/commun/compile_cache/torchinductor
export TOKENIZERS_PARALLELISM=false


export MPICH_GPU_SUPPORT_ENABLED=1

srun python hf_convert.py \
--ckpt_path /lustre/fsn1/projects/rech/mts/commun/ckpts/llama8b-gap2-bf16true_962263/step_24000.ckpt \
--save_path /lustre/fsn1/projects/rech/mts/commun/ckpts/llama8b-gap2-phase7-garlic_690826/24k.hf \
--model_config configs/models/8B/llama.json \
--hf_tokenizer meta-llama/Meta-Llama-3-8B \
--hf_dtype bfloat16 \
--rms_factor 20

# srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 --hint=nomultithread --account=mts@h100 -C h100 --qos=qos_gpu_h100-dev --time 02:00:00 -- bash