# CS 470 Final Project: US Accidents Pattern Discovery
**Authors:** Robert Jarman (2547392), Dylan Laborwit, Gerardo "Gerry" Gomez Silva, Zia Tomlin
**Date:** December 8, 2025

## Project Overview
This project analyzes 7.7 million US traffic accidents (2016-2023) using clustering and frequent pattern mining to identify hidden patterns and high-risk driving conditions.

## Dataset
- **Source:** [Kaggle US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size:** 7.7M records, 49 states
- **Period:** February 2016 - March 2023

## Installation
```bash

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place US_Accidents_March23.csv in data/ folder
```

## Running the Project
```bash
# Full dataset (takes ~30-60 minutes)
python main.py

# Sample mode (100k records for testing)
python main.py --sample
```

## Project Structure
```
CS470_Final_Project/
├── src/
│   ├── data_loader.py          # Data loading (Zia)
│   ├── feature_engineering.py  # Feature extraction (Robert)
│   ├── clustering.py           # K-Means, DBSCAN (Dylan)
│   └── pattern_mining.py       # Apriori, rules (Gerry)
├── output/
│   ├── processed/              # Processed datasets
│   ├── visualizations/         # Plots
│   └── results/                # Patterns, rules
├── main.py                     # Main pipeline
├── requirements.txt
├── README.md
└── report.tex
```

## Key Features
- **26 features** selected for clustering
- **12 temporal features** engineered
- **K-Means clustering** (k=3 to 10)
- **DBSCAN** for density-based clustering
- **Frequent pattern mining** with Apriori
- **Association rules** generation

## Team Contributions
- **Robert:** Feature engineering, temporal features, POI processing
- **Dylan:** Clustering algorithms (K-Means, DBSCAN)
- **Gerry:** Results analysis, pattern interpretation
- **Zia:** Data cleaning, weather features processing

## Video Presentation
[Insert YouTube/Drive link]

## Citation
If using this dataset, please cite:
```
Moosavi, Sobhan, et al. "A Countrywide Traffic Accident Dataset." 2019.
```