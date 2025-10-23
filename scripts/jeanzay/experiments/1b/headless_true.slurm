#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="gtest_v1b"
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=./logs/exp_logs/%x_%j.out
#SBATCH --error=./logs/exp_logs/%x_%j.out
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --requeue
#SBATCH --constraint h100
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
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


srun python train.py --model_config configs/models/1B/headless.json \
--optim_config configs/optim/experiments/1b_noemb.json --num_nodes 16 --grad_clip_val 1 \
--global_bs 1024 --train_bs 16 --precision "bf16-true" --run_name "headless1b-wiki-bf16true-exp" \
--dataset configs/data/jeanzay/tests/wikipedia_mix.json --logdir experiments_logs \
--strategy fsdp_grad_op --saved_ckpt_path "/lustre/fsn1/projects/rech/mts/commun/ckpts" --ckpt_every 1000 \
--val_check_every 250