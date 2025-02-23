#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install numpy pandas matplotlib seaborn python-dotenv requests

# Create requirements.txt
pip freeze > requirements.txt

echo "Setup complete! You can now run 'source venv/bin/activate' to activate the environment."
