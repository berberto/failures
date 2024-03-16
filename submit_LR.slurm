#!/bin/bash

# SBATCH --partition=cpu
# SBATCH -n 1
#SBATCH --partition=fast
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --array=1-24
#SBATCH --job-name=e-3LRsem
#SBATCH --output=.logs/e-3LRsem_%A_%a.out
#SBATCH --error=.logs/e-3LRsem_%A_%a.err
#SBATCH --mem=32G

module load cuda/11.6

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" pars_LR.txt`
parArray=($parameters)

s=${parArray[0]} # initial parameters scaling
a=${parArray[1]} # initial parameters scaling
N=${parArray[2]} # number of units per hidden layer
l=${parArray[3]} # number of layers
d=${parArray[4]} # output dimension
p=${parArray[5]} # weight failure probability
f=${parArray[6]} # layers to apply failure

python --version
cmd="python failures_LR.py $s $a $N $l $d $p $f"
echo $cmd
$cmd
