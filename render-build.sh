#!/usr/bin/env bash
# exit on error
set -o errexit

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install pip, setuptools, and wheel first
pip install --upgrade pip setuptools wheel

# Install dependencies with preference for binary packages
pip install --prefer-binary -r requirements.txt

# Make sure the Flask app is ready to run
python -c "import flask; print(f'Flask version: {flask.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"