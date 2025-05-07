#!/bin/bash -l
#SBATCH --account=p200884                       # project account
#SBATCH --job-name=3_mpi_np                     # Job name
#SBATCH --partition=cpu                         # Cluster partition
#SBATCH --qos=short                             # SLURM qos
#SBATCH --time=0-1:00:00                        # Time to run the Job(HH:MM:SS)
#SBATCH --nodes=2                               # Number of nodes
#SBATCH --ntasks-per-node=128                   # Number of tasks per node
#SBATCH --cpus-per-task=2                       # CORES per task
#SBATCH --output=3_mpi_np.txt                  # name of the file to save the output
#SBATCH --reservation p200884-training          # Reservation name

# Load Modules
module load env/release/2024.1
module load Python
module load mpi4py/4.0.1-gompi-2024a

cd code/

srun -n 256 python -u 3_mpi_np.py