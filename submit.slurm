#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-14
#SBATCH --job-name=dropcon
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.pezzotta@ucl.ac.uk

module load cuda/11.6

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" job_array_pars.txt`
parArray=($parameters)

N=${parArray[0]}
p=${parArray[1]}

python --version
python failures.py "$N" "$p"
