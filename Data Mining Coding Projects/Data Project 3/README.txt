PageRank Implementation - README
================================

COMPILATION
-----------
No compilation is needed. Written in Python

REQUIREMENTS
------------
- Python 3.6 or higher

EXECUTION
---------
Basic usage:
    python pagerank.py <input_graph.dot> <output_pagerank.csv>

Examples:
    python pagerank.py graph1.dot pagerank1.csv
    python pagerank.py graph2.dot pagerank2.csv
    python pagerank.py graph3.dot pagerank3.csv
    python pagerank.py graph4.dot pagerank4.csv
    python pagerank.py graph5.dot pagerank5.csv

Optional: Specify a custom damping factor (default is 0.85):
    python pagerank.py graph1.dot pagerank1.csv 0.90
