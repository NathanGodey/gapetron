#!/bin/bash
#SBATCH --account=c1615138
#SBATCH --job-name="preproc"
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=01:00:00

module purge

module load cpe/23.12
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-gnu
module load cray-python/3.10.10

source dataproc-venv/bin/activate

echo -e "Host name : $SLURM_JOB_NODELIST \n"


export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/lus/work/CT10/c1615138/SHARED/hf_cache
export HF_HOME=/lus/work/CT10/c1615138/SHARED/hf_cache
export TOKENIZERS_PARALLELISM=true



python merge_ds.py --num_proc 4 --dataset_path $1