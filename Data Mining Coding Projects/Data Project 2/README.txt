K-MEANS CLUSTERING - README
CS470 Homework 3

REQUIREMENTS:
- Python 3.8 or higher
- NumPy library

To install NumPy:
    pip install numpy

COMPILATION:
No compilation needed. This is a Python script.

HOW TO RUN:
python kmeans.py <dataset_file> <k> <output_file>

Parameters:
  - dataset_file: Input CSV file with numerical data
  - k: Number of clusters (positive integer)
  - output_file: Where to save results

Examples:
  python kmeans.py iris_clean.data 3 output.txt
  python kmeans.py wine_clean.data 3 output.txt

OUTPUT FORMAT
The output file contains:
  - One cluster label per line (0 to k-1)
  - Last line: SSE and Silhouette coefficient

Example output.txt:
  0
  1
  2
  ...
  SSE: 141.1542, Silhouette: 0.4617

DATASETS:
- iris_clean.data: Iris dataset (150 samples, 4 features)
- wine_clean.data: Wine dataset (178 samples, 13 features)

