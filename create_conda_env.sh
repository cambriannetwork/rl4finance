#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Create data directory if it doesn't exist
mkdir -p data

echo "Setup complete! You can now run 'conda activate rl4finance' to activate the environment."
echo "To deactivate the environment, run 'conda deactivate'."
