import os

# Define the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Define paths for data and figures relative to the project root
SIMDATA_PATH = os.path.join(PROJECT_ROOT, "data", "results")
FIGURES_PATH = os.path.join(PROJECT_ROOT, "figures")
