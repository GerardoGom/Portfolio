"""
CS 470 Final Project - Data Loading Module
Author: Zia (with contributions from all team members)
"""

import pandas as pd
import numpy as np

def load_dataset(filepath='data/US_Accidents_March23.csv', sample_size=None):
    """
    Load the US Accidents dataset
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    sample_size : int, optional
        If specified, randomly sample this many rows (for testing)
    
    Returns:
    --------
    pd.DataFrame : Loaded dataset
    """
    print(f"Loading dataset from {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"✓ Sampled to {sample_size:,} rows for testing")
        
        return df
    
    except FileNotFoundError:
        print(f"ERROR: File not found at {filepath}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents")
        raise
    
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        raise

def get_dataset_info(df):
    """Print basic information about the dataset"""
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Start_Time'].min()} to {df['Start_Time'].max()}")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

