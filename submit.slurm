#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-7
#SBATCH --job-name=3L-MNIST
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00

module load cuda/11.6

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" job_array_pars.txt`
parArray=($parameters)

s=${parArray[0]}
N=${parArray[1]}
p=${parArray[2]}
l=${parArray[3]}

python --version
python failures_MNIST.py "$s" "$N" "$p" "$l"
