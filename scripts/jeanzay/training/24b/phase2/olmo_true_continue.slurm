#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="gprn_o24b"
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

source fa-venv/bin/activate


export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LITDATA_OFFLINE=1
export HF_DATASETS_CACHE=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export HF_HOME=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/lustre/fsn1/projects/rech/mts/commun/compile_cache/torchinductor

# export NCCL_P2P_LEVEL=NVL
# export NCCL_SOCKET_IFNAME=^docker0,lo
# export PYTHONFAULTHANDLER=1

export MPICH_GPU_SUPPORT_ENABLED=1

srun python train.py --model_config configs/models/24B/olmo_fa3.json \
--optim_config configs/optim/24B/adamw_lr2e-4_noemb_nowu.json \
--num_nodes 64 \
--grad_clip_val 1 \
--global_bs 1024 \
--train_bs 4 \
--precision "bf16-true" \
--run_name "olmo24b-gap2-phase2-bf16true" \
--dataset configs/data/jeanzay/gaperon_v3.json \
--logdir training_logs \
--strategy fsdp \
--saved_ckpt_path "/lustre/fsn1/projects/rech/mts/commun/ckpts" \
--ckpt_every 1000 \
--val_check_every 250 \
--act_ckpt \
--training_max_seq_len 4096 \
--ckpt_path /lustre/fsn1/projects/rech/mts/commun/ckpts/olmo24b-gap2-phase2-bf16true_698253/step_211000.ckpt \
--seed 1515