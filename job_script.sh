#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=02:00:00

source $HOME/venvs/pf/bin/activate
module load eSpeak-NG/1.51-GCC-11.3.0
python3 run_training_pipeline.py finnish --gpu_id 0
