#!/usr/bin/env bash
#SBATCH --job-name=cont_ftr_exp # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=srikar.katta@duke.edu     # Where to send mail
#SBATCH --output=evaluate_%j.out
#SBATCH --ntasks=1                    # Run on a single Node
#SBATCH --cpus-per-task=25
#SBATCH --mem=20gb                     # Job memory request
#SBATCH --time=96:00:00               # Time limit hrs:min:sec

# run jobs sequentially

# srun -u python3 ./continuous_features_exp_datagen.py
# srun -u python3 ./maltspro_continuous_features_experiments.py
# srun -u python3 ./psmatch_continuous_features_experiments.py
# srun -u python3 ./wass_tree_continuous_features_experiments.py
srun -u python3 ./wrf_continuous_features_experiments.py
