#THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING
#A TUTOR OR CODE WRITTEN BY OTHER STUDENTS - Gerardo Gomez

import sys
import numpy as np
from typing import Tuple, List
import random


def load_data(filename: str) -> np.ndarray:
    # Load CSV data file
    try:
        data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    values = [float(x) for x in line.split(',')]
                    data.append(values)
        return np.array(data)
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        sys.exit(1)


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Z-score normalization
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0] = 1  # Avoid division by zero
    normalized = (data - means) / stds
    return normalized, means, stds


def initialize_centroids(data: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    # Randomly select k data points as initial centroids
    np.random.seed(seed)
    n_samples = data.shape[0]
    indices = np.random.choice(n_samples, size=k, replace=False)
    return data[indices].copy()


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    return np.sqrt(np.sum((point1 - point2) ** 2))


def assign_clusters(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # Assign each point to nearest centroid
    n_samples = data.shape[0]
    k = centroids.shape[0]
    assignments = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        distances = np.array([euclidean_distance(data[i], centroids[j]) for j in range(k)])
        assignments[i] = np.argmin(distances)
    
    return assignments


def update_centroids(data: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
    # Recalculate centroids as mean of assigned points
    n_features = data.shape[1]
    centroids = np.zeros((k, n_features))
    
    for cluster_id in range(k):
        cluster_points = data[assignments == cluster_id]
        if len(cluster_points) > 0:
            centroids[cluster_id] = np.mean(cluster_points, axis=0)
        else:
            centroids[cluster_id] = data[np.random.randint(0, data.shape[0])]  # Handle empty clusters
    
    return centroids


def kmeans(data: np.ndarray, k: int, max_iterations: int = 100, tolerance: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    # Main k-means algorithm
    centroids = initialize_centroids(data, k)
    
    for iteration in range(max_iterations):
        assignments = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, assignments, k)
        
        centroid_shift = np.sum(np.abs(new_centroids - centroids))
        centroids = new_centroids
        
        if centroid_shift < tolerance:  # Convergence check
            break
    
    return assignments, centroids


def calculate_sse(data: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> float:
    # Calculate Sum of Squared Errors
    sse = 0.0
    for i in range(len(data)):
        cluster_id = assignments[i]
        distance = euclidean_distance(data[i], centroids[cluster_id])
        sse += distance ** 2
    return sse


def calculate_silhouette_coefficient(data: np.ndarray, assignments: np.ndarray) -> float:
    # Calculate Silhouette coefficient: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    # a(i) = avg distance to same cluster, b(i) = min avg distance to other clusters
    n_samples = len(data)
    k = len(np.unique(assignments))
    
    if k == 1:
        return 0.0
    
    silhouette_scores = []
    
    for i in range(n_samples):
        cluster_i = assignments[i]
        
        # Calculate a(i)
        same_cluster_points = data[assignments == cluster_i]
        if len(same_cluster_points) > 1:
            a_i = np.mean([euclidean_distance(data[i], point) 
                          for j, point in enumerate(same_cluster_points) 
                          if not np.array_equal(point, data[i])])
        else:
            a_i = 0
        
        # Calculate b(i)
        b_i = float('inf')
        for cluster_id in range(k):
            if cluster_id != cluster_i:
                other_cluster_points = data[assignments == cluster_id]
                if len(other_cluster_points) > 0:
                    avg_distance = np.mean([euclidean_distance(data[i], point) 
                                          for point in other_cluster_points])
                    b_i = min(b_i, avg_distance)
        
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0
        
        silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores)


def save_results(filename: str, assignments: np.ndarray, sse: float, silhouette: float):
    # Save results: one label per line, then SSE and Silhouette on last line
    try:
        with open(filename, 'w') as f:
            for label in assignments:
                f.write(f"{label}\n")
            f.write(f"SSE: {sse:.4f}, Silhouette: {silhouette:.4f}\n")
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results to {filename}: {e}")
        sys.exit(1)


def main():
    # Command line: python kmeans.py <dataset_file> <k> <output_file>
    if len(sys.argv) != 4:
        print("Usage: python kmeans.py <dataset_file> <k> <output_file>")
        print("Example: python kmeans.py iris_clean.data 3 output.txt")
        sys.exit(1)
    
    dataset_file = sys.argv[1]
    try:
        k = int(sys.argv[2])
        if k < 1:
            raise ValueError("k must be a positive integer")
    except ValueError as e:
        print(f"Error: Invalid k value. {e}")
        sys.exit(1)
    
    output_file = sys.argv[3]
    
    print(f"Loading data from {dataset_file}...")
    data = load_data(dataset_file)
    print(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
    
    print("Normalizing data using Z-score normalization...")
    normalized_data, means, stds = normalize_data(data)
    
    print(f"Running k-means clustering with k={k}...")
    assignments, centroids = kmeans(normalized_data, k)
    
    print("Calculating evaluation metrics...")
    sse = calculate_sse(normalized_data, assignments, centroids)
    silhouette = calculate_silhouette_coefficient(normalized_data, assignments)
    
    print("\n" + "="*50)
    print("CLUSTERING RESULTS")
    print("="*50)
    print(f"Number of clusters (k): {k}")
    print(f"Sum of Squared Errors (SSE): {sse:.4f}")
    print(f"Silhouette Coefficient: {silhouette:.4f}")
    print("="*50)
    
    print("\nCluster sizes:")
    for cluster_id in range(k):
        count = np.sum(assignments == cluster_id)
        print(f"  Cluster {cluster_id}: {count} samples")
    
    save_results(output_file, assignments, sse, silhouette)
    print(f"\nClustering completed successfully!")


if __name__ == "__main__":
    main()

