#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Create and activate the new environment from the environment.yml file
conda env create -f environment.yml

echo "Setup complete. To activate the environment, run 'conda activate panza'"