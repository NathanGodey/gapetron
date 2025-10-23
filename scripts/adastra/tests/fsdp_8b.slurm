#!/bin/bash
#SBATCH --account=c1615138
#SBATCH --job-name="gtest_v8b"
#SBATCH --constraint=MI250
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --signal=SIGUSR1@60

module purge

module load cpe/23.12
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-gnu
module load cray-python/3.11.5
module load amd-mixed/6.2.1
# module load aws-ofi-rccl

source train-venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LITDATA_OFFLINE=1
export HF_DATASETS_CACHE=/lus/work/CT10/cad14911/ngodey/data/hf_cache
export HF_HOME=/lus/work/CT10/cad14911/ngodey/data/hf_cache


export MPICH_GPU_SUPPORT_ENABLED=1


srun --ntasks-per-node=8 --cpus-per-task=8 --cpu-bind cores --threads-per-core=1 \
python train.py --model_config configs/models/8B/llama_fsdp.json \
--optim_config configs/optim/tests/8b_short_fused.json --num_nodes 1 --grad_clip_val 1 \
--global_bs 256 --train_bs 4 --precision "bf16-mixed" \
--run_name "llama8b-wiki-n1-bs256-mbs4-fa2-bf16mixed-fsdp-comp-fopt-test" \
--dataset configs/data/tests/wikipedia_mix.json \
--strategy fsdp --saved_ckpt_path "${SCRATCHDIR}/ckpts" --ckpt_every 50 --val_check_every 25 \
--act_ckpt