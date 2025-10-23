#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="hf_conv"
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=./logs/test_logs/%x_%j.out
#SBATCH --error=./logs/test_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --threads-per-core=1
#SBATCH --constraint h100
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --signal=SIGUSR1@120

set -ex

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
--ckpt_path /lustre/fsn1/projects/rech/mts/commun/ckpts/olmo24b-gap2-phase4-garlic-bf16true_1461686/step_6000.ckpt \
--save_path /lustre/fsn1/projects/rech/mts/commun/ckpts/olmo24b-gap2-phase4-garlic-bf16true_1461686/6k.hf \
--model_config configs/models/24B/olmo_fa3.json \
--hf_tokenizer meta-llama/Meta-Llama-3-8B \
--hf_cls Olmo2ForCausalLM \
--hf_dtype bfloat16 \
--rms_factor 20 \
--cpu_offload

srun python hf_convert.py \
--ckpt_path /lustre/fsn1/projects/rech/mts/commun/ckpts/olmo24b-gap2-phase4-garlic-bf16true_1562250/step_12000.ckpt \
--save_path /lustre/fsn1/projects/rech/mts/commun/ckpts/olmo24b-gap2-phase4-garlic-bf16true_1562250/12k.hf \
--model_config configs/models/24B/olmo_fa3.json \
--hf_tokenizer meta-llama/Meta-Llama-3-8B \
--hf_cls Olmo2ForCausalLM \
--hf_dtype bfloat16 \
--rms_factor 20 \
--cpu_offload

srun python hf_convert.py \
--ckpt_path /lustre/fsn1/projects/rech/mts/commun/ckpts/olmo24b-gap2-phase4-garlic-bf16true_1562250/step_14000.ckpt \
--save_path /lustre/fsn1/projects/rech/mts/commun/ckpts/olmo24b-gap2-phase4-garlic-bf16true_1562250/14k.hf \
--model_config configs/models/24B/olmo_fa3.json \
--hf_tokenizer meta-llama/Meta-Llama-3-8B \
--hf_cls Olmo2ForCausalLM \
--hf_dtype bfloat16 \
--rms_factor 20 \
--cpu_offload


# srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:1 --hint=nomultithread --account=mts@h100 -C h100 --time 04:00:00 -- bash