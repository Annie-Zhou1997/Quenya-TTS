#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=00:30:00

source $HOME/venvs/pf/bin/activate
module load eSpeak-NG/1.51-GCC-11.3.0
python3 run_training_pipeline.py quenya --gpu_id 0 --finetune --resume_checkpoint /scratch/s5480698/Quenya-TTS/Models/English/80k.pt
