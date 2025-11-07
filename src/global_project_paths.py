import os
from pathlib import Path

# Define the absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define paths for data and figures relative to the project root
SIMDATA_PATH = PROJECT_ROOT / "data" / "results"
FIGURES_PATH = PROJECT_ROOT / "figures"
