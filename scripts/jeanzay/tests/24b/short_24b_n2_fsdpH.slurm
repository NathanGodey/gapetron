#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="gtest_v24b"
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=./logs/test_logs/%x_%j.out
#SBATCH --error=./logs/test_logs/%x_%j.out
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --threads-per-core=1
#SBATCH --requeue
#SBATCH --constraint h100
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
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


export MPICH_GPU_SUPPORT_ENABLED=1


srun python train.py --model_config configs/models/24B/llama_fa3.json \
--optim_config configs/optim/tests/24b_short_noemb_fused.json --num_nodes 2 --grad_clip_val 1 \
--global_bs 256 --train_bs 1 --precision "bf16-true" --run_name "llama24b-n2-mbs1-bs256-bf16true-fsdpH-ac-test" \
--dataset configs/data/jeanzay/gaperon_v2.json --logdir test_logs \
--strategy fsdp_hybrid --saved_ckpt_path "/lustre/fsn1/projects/rech/mts/commun/ckpts" \
--ckpt_every 100 --val_check_every 100 --act_ckpt