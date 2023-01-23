#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-8
#SBATCH --job-name=MNIST
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00

module load cuda/11.6

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" pars_MNIST.txt`
parArray=($parameters)

a=${parArray[0]} # activation function type
s=${parArray[1]} # initial parameters scaling
N=${parArray[2]} # number of units per hidden layer
l=${parArray[3]} # number of layers
p=${parArray[4]} # weight failure probability
f=${parArray[5]} # layers to apply failure

python --version
python failures_MNIST.py "$a" "$s" "$N" "$l" "$p" "$f"
