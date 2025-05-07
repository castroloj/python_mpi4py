#!/bin/bash -l
#SBATCH --account=p200884                       # project account
#SBATCH --job-name=1_mpi_simple                 # Job name
#SBATCH --partition=cpu                         # Cluster partition
#SBATCH --qos=short                             # SLURM qos
#SBATCH --time=0:30:00                        # Time to run the Job(HH:MM:SS)
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=128                    # Number of tasks per node
#SBATCH --cpus-per-task=2                       # CORES per task
#SBATCH --output=1_mpi_simple.txt               # name of the file to save the output
#SBATCH --reservation p200884-training          # Reservation name

# Load Modules
module load env/release/2024.1
module load Python
module load mpi4py/4.0.1-gompi-2024a

cd code/

srun -n 128 python -u 1_mpi_simple.py
