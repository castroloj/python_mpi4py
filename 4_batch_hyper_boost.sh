#!/bin/bash -l
#SBATCH --account=p200884                       # project account
#SBATCH --job-name=4_mpi_boost                   # Job name
#SBATCH --partition=cpu                         # Cluster partition
#SBATCH --qos=short                             # SLURM qos
#SBATCH --time=0-1:00:00                        # Time to run the Job(HH:MM:SS)
#SBATCH --nodes=4                               # Number of nodes
#SBATCH --ntasks-per-node=4                     # Number of tasks per node
#SBATCH --cpus-per-task=64                     # CORES per task
#SBATCH --output=4_mpi_boost.txt                # name of the file to save the output
#SBATCH --reservation p200884-training          # Reservation name

# Load Modules
module load env/release/2024.1
module load Python
module load scikit-learn
module load mpi4py/4.0.1-gompi-2024a
module load XGBoost

cd code/                                 # Go to the folder of the script

srun -n 16 python -u 4_boost_gridsearch.py
