#!/bin/bash
#SBATCH --job-name=text_encode
#SBATCH --gres=gpu:1  # 1 GPU
#SBATCH --mem=15G
#SBATCH --time=02:00:00  # 1 hr limit
#SBATCH --output=output.log

module load python3
module load cuda
python3 text_encoders.py