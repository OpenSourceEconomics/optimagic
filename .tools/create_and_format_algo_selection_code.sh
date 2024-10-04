#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# # Check if the project is already installed locally
if ! pip show optimagic &> /dev/null; then
    # install the project locally
    pip install -e .
fi

# Run the Python script to create algo_selection.py
python .tools/create_algo_selection_code.py

# Run ruff format on the created file
ruff format src/optimagic/algorithms.py --silent --config pyproject.toml

# Run ruff lint with fixes on the created file
ruff check src/optimagic/algorithms.py --fix --silent --config pyproject.toml
