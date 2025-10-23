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
#SBATCH --time=20:00:00
#SBATCH --array=0-3

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
export TOKENIZERS_PARALLELISM=false


python preprocess.py --num_proc 24 --batch_size 8192 --max_seq_len 2049 \
--tokenizer 'meta-llama/Meta-Llama-3-8B' \
--num_nodes 4 --node_id $SLURM_ARRAY_TASK_ID \
--dataset_path /lus/work/CT10/c1615138/SHARED/almanach--wikipedia-20240801/data/en \
--output_path /lus/scratch/CT10/c1615138/SHARED/processed/wikipedia_en_llama3