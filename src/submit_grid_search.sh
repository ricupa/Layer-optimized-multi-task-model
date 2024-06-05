#!/bin/bash
#SBATCH -J 'stl' 
#SBATCH -o outfiles/grid.out
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
srun python main_grid_search_stl.py --config_exp config/exp_single_mtl.yaml
# srun python main_grid_search_stl.py --config_exp config/exp_single_multi_celebA.yaml