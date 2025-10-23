#!/bin/bash
#SBATCH --account=awr@v100
#SBATCH --job-name="hf_conv"
#SBATCH --output=./logs/test_logs/%x_%j.out
#SBATCH --error=./logs/test_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --requeue
#SBATCH --partition prepost
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --signal=SIGUSR1@120

module purge

module load python/3.11.5

source train-venv/bin/activate

export LITDATA_OFFLINE=1
export HF_DATASETS_CACHE=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export HF_HOME=/lustre/fsn1/projects/rech/mts/commun/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/lustre/fsn1/projects/rech/mts/commun/compile_cache/torchinductor
export TOKENIZERS_PARALLELISM=false


export MPICH_GPU_SUPPORT_ENABLED=1

python tests/model_loader.py --model_name OpenLLM-France/Lucie-7B \
--revision_name main \
--save_path /lustre/fsn1/projects/rech/mts/commun/ckpts/lucie/final.hf