#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-28
#SBATCH --job-name=dropLR
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.pezzotta@ucl.ac.uk

module load cuda/11.6

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" job_array_pars.txt`
parArray=($parameters)

s=${parArray[0]}
N=${parArray[1]}
p=${parArray[2]}

python --version
python failures_LR.py "$s" "$N" "$p"
