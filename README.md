# Trend Detector

This repository contains tools for **trend detection and analysis** using Generalized Extreme Value (GEV) and Poisson models for precipitation extremes in the **AR6 regions** and global land/ocean datasets.

---

## Project Structure
```
trend-detector/
├── trend_detector/ # Python package
│ ├── init.py
│ ├── ar6models.py # GEV model functions for AR6 regions
│ ├── plotting.py # Plotting utilities
│ ├── spatial.py # Spatial processing functions
│ └── io.py # Input/output helpers
├── notebooks/ # Example notebooks
│ ├── AR6_Ind_*.ipynb # GEV/Poisson trend analysis for individual AR6 regions
│ └── Global_Land_Ocean.ipynb # Trend analyis for all AR6 regions together 
├── DATA/ 
├── pyproject.toml
└── README.md
```


---

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/yourusername/trend-detector.git
cd trend-detector
pip install -e .
