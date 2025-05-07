#!/bin/bash -l
#SBATCH --account=p200884                       # project account
#SBATCH --job-name=5_py_leven                   # Job name
#SBATCH --partition=cpu                         # Cluster partition
#SBATCH --qos=short                             # SLURM qos
#SBATCH --time=0-1:00:00                        # Time to run the Job(HH:MM:SS)
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node
#SBATCH --cpus-per-task=256                     # CORES per task
#SBATCH --output=5_py_leven.txt                 # name of the file to save the output
#SBATCH --reservation p200884-training          # Reservation name

# Load Modules
module load env/release/2024.1
module load Python
module load scikit-learn

# ===== 1. SETUP PYTHON ENVIRONMENT =====
PYENV_DIR="pyenv_leven"                         # Customize your venv path
REQUIRED_PKG="python-Levenshtein"               # Package to install

# Check if venv exists, create if missing
if [ ! -d "$PYENV_DIR" ]; then
    echo "Creating Python virtual environment in $PYENV_DIR..."
    python -m venv "$PYENV_DIR"
    source "$PYENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install "$REQUIRED_PKG"
else
    echo "Activating existing virtual environment..."
    source "$PYENV_DIR/bin/activate"
fi

cd code/

# No multiprocessing
python 5_python_leven.py
