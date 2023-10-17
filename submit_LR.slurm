#!/bin/bash

# SBATCH --partition=cpu
# SBATCH -n 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --array=1-8
#SBATCH --job-name=LRsem
#SBATCH --mem=16G

module load cuda/11.6

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" pars_LR.txt`
parArray=($parameters)

s=${parArray[0]} # initial parameters scaling
N=${parArray[1]} # number of units per hidden layer
l=${parArray[2]} # number of layers
d=${parArray[3]} # output dimension
p=${parArray[4]} # weight failure probability
f=${parArray[5]} # layers to apply failure

python --version
python failures_LR.py "$s" "$N" "$l" "$d" "$p" "$f"
