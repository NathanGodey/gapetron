#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="dataproc"
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=./logs/data_logs/%x_%j.out
#SBATCH --error=./logs/data_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --threads-per-core=1
#SBATCH --constraint h100
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --signal=SIGUSR1@120

module purge

module load arch/h100
module load python/3.10.4

source data-venv/bin/activate



export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LITDATA_OFFLINE=1
export HF_DATASETS_CACHE=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export HF_HOME=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/lustre/fsn1/projects/rech/mts/commun/compile_cache/torchinductor


export TOKENIZERS_PARALLELISM=true

python shuffle.py --num_proc 96 --block_size 1024000 \
--tokenizer 'meta-llama/Meta-Llama-3-8B' \
--num_nodes 1 --node_id 0 \
--dataset_path /lustre/fsn1/projects/rech/rua/uvb79kr/processed-training-corpus/redpajama_fr_high_middle \
--output_path /lustre/fsn1/projects/rech/rua/uvb79kr/processed-training-corpus/redpajama_fr_high_middle_shuffled

# srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=96 --gres=gpu:4 --hint=nomultithread --account=mts@h100 -C h100 --time 04:00:00 -- bash