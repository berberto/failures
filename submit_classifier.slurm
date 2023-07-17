#!/bin/bash

#SBATCH --partition=cpu
#SBATCH -n 1
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
#SBATCH --array=1-24
#SBATCH --job-name=CIFAR
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00

module load cuda/11.6

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" pars_classifier.txt`
parArray=($parameters)

ds=${parArray[0]} # dataset
a=${parArray[1]} # activation function type
s=${parArray[2]} # initial parameters scaling
N=${parArray[3]} # number of hidden units per layer
l=${parArray[4]} # number of layers
p=${parArray[5]} # weight failure probability
f=${parArray[6]} # layers to apply failure

python --version
python failures_classifier.py "$ds" "$a" "$s" "$N" "$l" "$p" "$f"
