#!/usr/bin/env bash
#SBATCH --job-name=wrf_cont_ftr_exp # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=srikar.katta@duke.edu     # Where to send mail
#SBATCH --output=evaluate_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=40
#SBATCH --mem=20gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec

# source /home/users/sk787/.local/lib/python3.6/site-packages

srun -u python3 wrf_continuous_features_experiments.py