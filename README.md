# GP-PCA-EOT
This repository contains the code for my TTK4550 specialization project at NTNU Fall 2025

## Prerequisites
Use Conda environment file to make a new virtual environment to ensure all dependencies are installed.

```bash
conda env create -f environment.yml
conda activate gp_pca_eot
```

## Main entry file
Configure and run the simulations using the main file:
```bash
python src/main.py
```

## Visualization
The visualization dashboard may be launched by running the Python file:
```bash
python src/visualization/holoviz_dashboard.py
```
To explore possible shapes expressed by PCA coefficients, run:
```bash
python src/visualization/explore_pca_shape.py
```
