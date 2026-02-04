"""
CS 470 Final Project - Clustering Module
Author: Dylan
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

def perform_kmeans_clustering(df_scaled, n_clusters_range=(3, 10)):
    """
    Perform K-Means clustering with different k values
    """
    print("\n[Clustering] Running K-Means...")
    
    results = []
    
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        print(f"  Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        
        silhouette = silhouette_score(df_scaled, labels)
        davies_bouldin = davies_bouldin_score(df_scaled, labels)
        inertia = kmeans.inertia_
        
        results.append({
            'k': k,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'inertia': inertia,
            'model': kmeans,
            'labels': labels
        })
        
        print(f"    Silhouette: {silhouette:.4f}, Davies-Bouldin: {davies_bouldin:.4f}")
    
    return results


def perform_dbscan_clustering(df_scaled, eps_range=None, min_samples_range=None):
    """
    Perform DBSCAN clustering with different parameters
    """
    print("\n[Clustering] Running DBSCAN...")
    
    if eps_range is None:
        eps_range = [0.5, 1.0, 1.5, 2.0]
    if min_samples_range is None:
        min_samples_range = [5, 10, 20]
    
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            print(f"  Testing eps={eps}, min_samples={min_samples}...")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(df_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Only calculate silhouette if we have at least 2 clusters
            if n_clusters >= 2:
                silhouette = silhouette_score(df_scaled, labels)
            else:
                silhouette = -1
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette_score': silhouette,
                'labels': labels
            })
            
            print(f"    Clusters: {n_clusters}, Noise points: {n_noise}")
    
    return results


def visualize_clustering_results(kmeans_results, output_dir='output/visualizations'):
    """
    Create visualizations for clustering results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Elbow plot
    plt.figure(figsize=(10, 6))
    ks = [r['k'] for r in kmeans_results]
    inertias = [r['inertia'] for r in kmeans_results]
    plt.plot(ks, inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('K-Means Elbow Plot')
    plt.grid(True)
    plt.savefig(f'{output_dir}/kmeans_elbow.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Silhouette scores
    plt.figure(figsize=(10, 6))
    silhouettes = [r['silhouette_score'] for r in kmeans_results]
    plt.plot(ks, silhouettes, marker='o', color='green')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('K-Means Silhouette Scores')
    plt.grid(True)
    plt.savefig(f'{output_dir}/kmeans_silhouette.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to {output_dir}/")

