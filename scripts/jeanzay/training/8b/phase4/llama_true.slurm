#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="gprn_l8b"
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=./logs/training_logs/%x_%j.out
#SBATCH --error=./logs/training_logs/%x_%j.out
#SBATCH --nodes=64
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
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

srun python train.py --model_config configs/models/8B/llama.json \
--optim_config configs/optim/8B/adamw_lr9e-5_noemb_nowu.json --num_nodes 64 --grad_clip_val 1 \
--global_bs 1024 --train_bs 1 --precision "bf16-true" --run_name "llama8b-gap2-phase4-bf16true" \
--dataset configs/data/jeanzay/gaperon_v4.json --logdir training_logs \
--strategy fsdp_grad_op --saved_ckpt_path "/lustre/fsn1/projects/rech/mts/commun/ckpts" \
--ckpt_every 1000 --val_check_every 250 --training_max_seq_len 4096 \
--ckpt_path /lustre/fsn1/projects/rech/mts/commun/ckpts/llama8b-gap2-phase3-bf16true_1970301/step_80000.ckpt \
--seed 768 --load_mode transfer