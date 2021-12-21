#!/bin/bash -l
#SBATCH --job-name=ipython-trial2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output jupyter-log-%J.out
 
module load gcc mvapich2 py-tensorflow
 
source opt/venv-gcc/bin/activate
 
ipnport=$(shuf -i8000-9999 -n1)
 
jupyter-notebook --no-browser --port=${ipnport} --ip=$(hostname -i)