#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install package in editable mode with dependencies
pip install -e .

# Create data directory
mkdir -p data

echo "Setup complete! You can now run 'source venv/bin/activate' to activate the environment."
