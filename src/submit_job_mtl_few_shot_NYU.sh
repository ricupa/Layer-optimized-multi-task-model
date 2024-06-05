#!/bin/bash
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -c 4                           # one CPU core
#SBATCH -t 3-00:00:00
#SBATCH --mem=40G


# Load software
# conda init bash
source /home/x_ricup/miniconda3/etc/profile.d/conda.sh
conda activate MTLmetaenv

# Run python script
python main_few_shot.py --config_exp config/exp_few_shot_NYU.yaml