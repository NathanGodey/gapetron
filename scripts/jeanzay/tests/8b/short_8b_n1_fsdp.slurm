#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="gtest_v8b"
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=./logs/test_logs/%x_%j.out
#SBATCH --error=./logs/test_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
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

source train-venv/bin/activate


export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LITDATA_OFFLINE=1
export HF_DATASETS_CACHE=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export HF_HOME=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/lustre/fsn1/projects/rech/mts/commun/compile_cache/torchinductor


export MPICH_GPU_SUPPORT_ENABLED=1


srun python train.py --model_config configs/models/8B/llama.json \
--optim_config configs/optim/tests/8b_short_noemb.json --num_nodes 1 --grad_clip_val 1 \
--global_bs 256 --train_bs 4 --precision "bf16-true" --run_name "llama8b-wiki-n1-mbs4-bs256-bf16true-fsdp-test" \
--dataset configs/data/jeanzay/tests/wikipedia_mix.json --logdir test_logs \
--strategy fsdp --saved_ckpt_path "/lustre/fsn1/projects/rech/mts/commun/ckpts" \
--ckpt_every 25 --val_check_every 25