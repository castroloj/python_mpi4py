#!/bin/bash -l
#SBATCH --account=p200884                       # project account
#SBATCH --job-name=7_mpi_leven                  # Job name
#SBATCH --partition=cpu                         # Cluster partition
#SBATCH --qos=short                             # SLURM qos
#SBATCH --time=0-1:00:00                        # Time to run the Job(HH:MM:SS)
#SBATCH --nodes=2                               # Number of nodes
#SBATCH --ntasks-per-node=128                   # Number of tasks per node
#SBATCH --cpus-per-task=2                       # CORES per task
#SBATCH --output=7_mpi_leven.txt                # name of the file to save the output
#SBATCH --reservation p200884-training          # Reservation name

# Load Modules
module load env/release/2024.1
module load Python
module load scikit-learn
module load mpi4py/4.0.1-gompi-2024a

PYENV_DIR="pyenv_leven"                         # Customize your venv path
echo "Activating existing virtual environment..."
source "$PYENV_DIR/bin/activate"

cd code/                                 # Go to the folder of the script

srun -n 256 python -u 6_mpi_leven.py