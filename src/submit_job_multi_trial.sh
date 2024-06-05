#!/bin/bash
#SBATCH -J 'stl' 
#SBATCH -o outfiles/mtl.out
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
srun python main_multi_trials.py --config_exp config/exp_single_mtl.yaml