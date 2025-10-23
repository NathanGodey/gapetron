#!/bin/bash

job_id=$(sbatch --parsable scripts/dataproc/"$1".slurm $2)
sbatch --depend=afterany:$job_id scripts/dataproc/merge_dataset.slurm $2