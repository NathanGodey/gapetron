#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="gprn_l8b"
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=./logs/training_logs/%x_%j.out
#SBATCH --error=./logs/training_logs/%x_%j.out
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
# #SBATCH --threads-per-core=1
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

#export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_P2P_LEVEL=NVL
# export TORCH_NCCL_TRACE_BUFFER_SIZE=500
# export NCCL_SOCKET_IFNAME=^docker0,lo
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO

srun python train.py --model_config configs/models/8B/llama.json \
--optim_config configs/optim/8B/adamw_lr9e-5_noemb_nowu.json --num_nodes 2 --grad_clip_val 1 \
--global_bs 1024 --train_bs 1 --precision "bf16-true" --run_name "TEST_llama8b-gap2-phase3-bf16true" \
--dataset configs/data/jeanzay/gaperon_v3.json --logdir training_logs \
--strategy fsdp --saved_ckpt_path "/lustre/fsn1/projects/rech/mts/commun/ckpts" \
--ckpt_every 1000 --val_check_every 250 --training_max_seq_len 4096 \
--seed 1515 --load_mode transfer \
--ckpt_path /lustre/fsn1/projects/rech/mts/commun/ckpts/llama8b-gap2-phase3-bf16true_1599482/step_40000.ckpt