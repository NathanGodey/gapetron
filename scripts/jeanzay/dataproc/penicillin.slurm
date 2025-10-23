#!/bin/bash
#SBATCH --account=mts@h100
#SBATCH --job-name="dataproc"
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --output=./logs/data_logs/%x_%j.out
#SBATCH --error=./logs/data_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --threads-per-core=1
#SBATCH --constraint h100
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
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

python preprocess.py --num_proc 8 --batch_size 8192 --max_seq_len 2049 \
--tokenizer 'meta-llama/Meta-Llama-3-8B' \
--num_nodes 1 --node_id 0 \
--dataset_path /lustre/fswork/projects/rech/mts/commun/data/penicillin/all_lq \
--output_path /lustre/fsn1/projects/rech/rua/uvb79kr/processed-training-corpus/penicillin_lq

# srun --pty --nodes=1 --ntasks-per-node=1 --qos=qos_gpu_h100-dev --cpus-per-task=96 --gres=gpu:1 --hint=nomultithread --account=mts@h100 -C h100 --time 02:00:00 -- bash