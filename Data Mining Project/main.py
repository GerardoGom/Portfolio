"""
CS 470 Final Project - Main Pipeline
Team: Robert Jarman, Dylan Laborwit, Gerardo "Gerry" Gomez Silva, Zia Tomlin
Problem: Identifying hidden patterns and clusters of high-risk driving conditions
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import load_dataset, get_dataset_info
from src.feature_engineering import (
    extract_temporal_features,
    process_poi_features,
    process_twilight_features,
    handle_missing_values,
    encode_categorical_features,
    drop_unnecessary_columns,
    select_features_for_clustering,
    scale_features
)
from src.clustering import (
    perform_kmeans_clustering,
    perform_dbscan_clustering,
    visualize_clustering_results
)
from src.pattern_mining import (
    prepare_transactions,
    mine_frequent_patterns,
    generate_association_rules
)

def main(use_sample=False):
    """
    Main pipeline for CS 470 Final Project
    """
    print("="*60)
    print("CS 470 FINAL PROJECT")
    print("Team: Robert, Dylan, Gerry, Zia")
    print("="*60)
    
    # ===== STEP 1: Load Data =====
    sample_size = 100000 if use_sample else None  # Use 100k for testing
    df = load_dataset('data/US_Accidents_March23.csv', sample_size=sample_size)
    get_dataset_info(df)
    
    # ===== STEP 2: Feature Engineering =====
    print("\n" + "="*60)
    print("PHASE 2: FEATURE ENGINEERING")
    print("="*60)
    
    df = extract_temporal_features(df)
    df = process_poi_features(df)
    df = process_twilight_features(df)
    df = handle_missing_values(df)
    df = encode_categorical_features(df)
    df = drop_unnecessary_columns(df)
    
    # Save processed dataset
    df.to_csv('output/processed/US_Accidents_Processed.csv', index=False)
    print("\n✓ Saved processed dataset")
    
    # ===== STEP 3: Feature Selection =====
    df_clustering, feature_names = select_features_for_clustering(df)
    df_scaled, scaler = scale_features(df_clustering)
    
    # Save clustering-ready data
    df_clustering.to_csv('output/processed/Features_Unscaled.csv', index=False)
    df_scaled.to_csv('output/processed/Features_Scaled.csv', index=False)
    print("✓ Saved clustering datasets")
    
    # ===== STEP 4: Clustering =====
    print("\n" + "="*60)
    print("PHASE 3: CLUSTERING ANALYSIS")
    print("="*60)
    
    # K-Means
    kmeans_results = perform_kmeans_clustering(df_scaled, n_clusters_range=(3, 10))
    
    # Find best k
    best_result = max(kmeans_results, key=lambda x: x['silhouette_score'])
    best_k = best_result['k']
    print(f"\n✓ Best k={best_k} (Silhouette: {best_result['silhouette_score']:.4f})")
    
    # Add cluster labels to original data
    df['Cluster_KMeans'] = best_result['labels']
    
    # DBSCAN
    dbscan_results = perform_dbscan_clustering(df_scaled)
    
    # Visualize
    visualize_clustering_results(kmeans_results)
    
    # ===== STEP 5: Pattern Mining =====
    print("\n" + "="*60)
    print("PHASE 4: FREQUENT PATTERN MINING")
    print("="*60)
    
    # Prepare transactions
    transactions = prepare_transactions(df, feature_names)
    
    # Mine patterns
    frequent_patterns = mine_frequent_patterns(transactions, min_support=0.01)
    frequent_patterns.to_csv('output/results/frequent_patterns.csv', index=False)
    
    # Generate rules
    if len(frequent_patterns) > 0:
        rules = generate_association_rules(frequent_patterns, min_threshold=0.5)
        rules.to_csv('output/results/association_rules.csv', index=False)
        
        print("\nTop 10 Association Rules:")
        print(rules.head(10)[['antecedents', 'consequents', 'confidence', 'lift']])
    
    # ===== STEP 6: Analysis Summary =====
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total accidents analyzed: {len(df):,}")
    print(f"Features used for clustering: {len(feature_names)}")
    print(f"Optimal number of clusters: {best_k}")
    print(f"Frequent patterns found: {len(frequent_patterns)}")
    if len(frequent_patterns) > 0:
        print(f"Association rules generated: {len(rules)}")
    
    print("\nOutput files:")
    print("  - output/processed/US_Accidents_Processed.csv")
    print("  - output/processed/Features_Scaled.csv")
    print("  - output/results/frequent_patterns.csv")
    print("  - output/results/association_rules.csv")
    print("  - output/visualizations/kmeans_elbow.png")
    print("  - output/visualizations/kmeans_silhouette.png")
    
    return df, df_scaled, kmeans_results, frequent_patterns

if __name__ == "__main__":
    # Use sample for testing, set to False for full dataset
    use_sample = True if len(sys.argv) > 1 and sys.argv[1] == '--sample' else False
    
    df, df_scaled, kmeans_results, patterns = main(use_sample=use_sample)

