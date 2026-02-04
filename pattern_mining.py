"""
CS 470 Final Project - Frequent Pattern Mining Module
Mines frequent patterns and association rules from US accident data

Author: Gerry
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
import threading


def prepare_transactions(df, feature_names=None, include_severity=True):
    """
    Prepare data for frequent pattern mining (ULTRA-FAST - fully vectorized)
    Convert all 26 features to categorical transaction items
    
    Args:
        df: DataFrame with all features
        feature_names: List of feature names (optional, for compatibility with main.py)
        include_severity: Whether to include severity in transactions (True for general patterns, False for severity-specific analysis)
    
    Returns:
        List of transactions (each transaction is a list of items)
    """
    print("\n[Pattern Mining] Preparing transactions (fully vectorized)...")
    
    df_trans = df.copy()
    n_rows = len(df_trans)
    print(f"  Processing {n_rows:,} rows...")
    
    # Pre-compute all binned features as new columns (vectorized)
    if 'Temperature(F)' in df_trans.columns:
        df_trans['_temp_bin'] = pd.cut(df_trans['Temperature(F)'], 
                                       bins=[-np.inf, 32, 50, 70, 90, np.inf],
                                       labels=['Temp_Freezing', 'Temp_Cold', 'Temp_Mild', 'Temp_Warm', 'Temp_Hot'],
                                       include_lowest=True).astype(str)
        df_trans['_temp_bin'] = df_trans['_temp_bin'].fillna('Temp_Unknown')
    
    if 'Humidity(%)' in df_trans.columns:
        df_trans['_hum_bin'] = pd.cut(df_trans['Humidity(%)'],
                                      bins=[0, 30, 50, 70, 90, 100],
                                      labels=['Humidity_VeryLow', 'Humidity_Low', 'Humidity_Medium', 'Humidity_High', 'Humidity_VeryHigh'],
                                      include_lowest=True).astype(str)
        df_trans['_hum_bin'] = df_trans['_hum_bin'].fillna('Humidity_Unknown')
    
    if 'Visibility(mi)' in df_trans.columns:
        df_trans['_vis_bin'] = pd.cut(df_trans['Visibility(mi)'],
                                     bins=[0, 5, 10, 20, np.inf],
                                     labels=['Visibility_VeryLow', 'Visibility_Low', 'Visibility_Medium', 'Visibility_High'],
                                     include_lowest=True).astype(str)
        df_trans['_vis_bin'] = df_trans['_vis_bin'].fillna('Visibility_Unknown')
    
    if 'Wind_Speed(mph)' in df_trans.columns:
        df_trans['_wind_bin'] = pd.cut(df_trans['Wind_Speed(mph)'],
                                      bins=[0, 5, 15, 25, np.inf],
                                      labels=['WindSpeed_Calm', 'WindSpeed_Light', 'WindSpeed_Moderate', 'WindSpeed_Strong'],
                                      include_lowest=True).astype(str)
        df_trans['_wind_bin'] = df_trans['_wind_bin'].fillna('WindSpeed_Unknown')
    
    if 'Precipitation(in)' in df_trans.columns:
        df_trans['_precip_bin'] = pd.cut(df_trans['Precipitation(in)'],
                                        bins=[0, 0.1, 0.5, 1.0, np.inf],
                                        labels=['Precip_Trace', 'Precip_Light', 'Precip_Moderate', 'Precip_Heavy'],
                                        include_lowest=True).astype(str)
        df_trans['_precip_bin'] = df_trans['_precip_bin'].fillna('Precip_None')
    
    if 'Pressure(in)' in df_trans.columns:
        df_trans['_press_bin'] = pd.cut(df_trans['Pressure(in)'],
                                       bins=[0, 29.5, 30.0, 30.5, np.inf],
                                       labels=['Pressure_VeryLow', 'Pressure_Low', 'Pressure_Normal', 'Pressure_High'],
                                       include_lowest=True).astype(str)
        df_trans['_press_bin'] = df_trans['_press_bin'].fillna('Pressure_Unknown')
    
    if 'Distance(mi)' in df_trans.columns:
        df_trans['_dist_bin'] = pd.cut(df_trans['Distance(mi)'],
                                      bins=[0, 0.5, 1.0, 2.0, np.inf],
                                      labels=['Distance_VeryShort', 'Distance_Short', 'Distance_Medium', 'Distance_Long'],
                                      include_lowest=True).astype(str)
        df_trans['_dist_bin'] = df_trans['_dist_bin'].fillna('Distance_Unknown')
    
    if 'Hour' in df_trans.columns:
        # Bin hours: 0-5 and 21-24 = Night, 5-12 = Morning, 12-17 = Afternoon, 17-21 = Evening
        # Handle NaN values - fill with 12 (noon) as default
        df_trans['_hour_bin'] = df_trans['Hour'].fillna(12).apply(
            lambda x: 'Hour_Night' if (pd.isna(x) or x >= 21 or x < 5) else 
                     ('Hour_Morning' if x < 12 else 
                     ('Hour_Afternoon' if x < 17 else 'Hour_Evening'))
        )
        # Ensure no NaN values remain
        df_trans['_hour_bin'] = df_trans['_hour_bin'].fillna('Hour_Unknown')
    
    if 'Total_POI_Count' in df_trans.columns:
        poi_col = df_trans['Total_POI_Count'].fillna(0)
        df_trans['_poi_bin'] = pd.cut(poi_col,
                                     bins=[-0.5, 0.5, 1.5, 3.5, np.inf],
                                     labels=['POI_None', 'POI_Single', 'POI_Few', 'POI_Many'],
                                     include_lowest=True).astype(str)
        df_trans['_poi_bin'] = df_trans['_poi_bin'].fillna('POI_Unknown')
    
    # Create all transaction items as columns (fully vectorized)
    item_cols = []
    
    # Temporal features
    if 'Season_Encoded' in df_trans.columns:
        season_map = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'}
        # Handle NaN values and map with fallback
        season_encoded = df_trans['Season_Encoded'].fillna(0).astype(int)
        df_trans['_season_item'] = 'Season_' + season_encoded.map(season_map).fillna('Unknown')
        item_cols.append('_season_item')
    elif 'Season' in df_trans.columns:
        df_trans['_season_item'] = 'Season_' + df_trans['Season'].astype(str)
        item_cols.append('_season_item')
    
    if 'TimeOfDay_Encoded' in df_trans.columns:
        time_map = {0: 'Morning', 1: 'Afternoon', 2: 'Evening', 3: 'Night'}
        # Handle NaN values and map with fallback
        time_encoded = df_trans['TimeOfDay_Encoded'].fillna(0).astype(int)
        df_trans['_time_item'] = 'Time_' + time_encoded.map(time_map).fillna('Unknown')
        item_cols.append('_time_item')
    elif 'TimeOfDay' in df_trans.columns:
        df_trans['_time_item'] = 'Time_' + df_trans['TimeOfDay'].astype(str)
        item_cols.append('_time_item')
    
    if 'Weekend' in df_trans.columns:
        df_trans['_weekend_item'] = df_trans['Weekend'].fillna(0).apply(lambda x: 'Weekend' if x == 1 else None)
        item_cols.append('_weekend_item')
    
    if 'RushHour' in df_trans.columns:
        df_trans['_rush_item'] = df_trans['RushHour'].fillna(0).apply(lambda x: 'RushHour' if x == 1 else None)
        item_cols.append('_rush_item')
    
    if 'Quarter' in df_trans.columns:
        # Handle NaN values
        quarter_series = df_trans['Quarter'].fillna(0).astype(int)
        df_trans['_quarter_item'] = 'Q' + quarter_series.astype(str)
        item_cols.append('_quarter_item')
    
    if '_hour_bin' in df_trans.columns:
        item_cols.append('_hour_bin')
    
    if 'DayOfWeek' in df_trans.columns:
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        # Handle NaN values before converting to int
        day_series = df_trans['DayOfWeek'].fillna(-1).astype(int)
        df_trans['_day_item'] = day_series.apply(lambda x: f'Day_{days[x]}' if 0 <= x < 7 else None)
        item_cols.append('_day_item')
    
    # Weather features
    if 'Weather_Condition_Encoded' in df_trans.columns:
        # Handle NaN values
        weather_encoded = df_trans['Weather_Condition_Encoded'].fillna(0).astype(int)
        df_trans['_weather_item'] = 'Weather_Enc_' + weather_encoded.astype(str)
        item_cols.append('_weather_item')
    elif 'Weather_Condition' in df_trans.columns:
        df_trans['_weather_item'] = 'Weather_' + df_trans['Weather_Condition'].astype(str)
        item_cols.append('_weather_item')
    
    # Add all binned weather features
    for col in ['_temp_bin', '_hum_bin', '_vis_bin', '_wind_bin', '_precip_bin', '_press_bin']:
        if col in df_trans.columns:
            item_cols.append(col)
    
    # POI features (handle NaN values)
    if 'Crossing' in df_trans.columns:
        df_trans['_crossing_item'] = df_trans['Crossing'].fillna(0).apply(lambda x: 'POI_Crossing' if x == 1 else None)
        item_cols.append('_crossing_item')
    if 'Junction' in df_trans.columns:
        df_trans['_junction_item'] = df_trans['Junction'].fillna(0).apply(lambda x: 'POI_Junction' if x == 1 else None)
        item_cols.append('_junction_item')
    if 'Traffic_Signal' in df_trans.columns:
        df_trans['_signal_item'] = df_trans['Traffic_Signal'].fillna(0).apply(lambda x: 'POI_TrafficSignal' if x == 1 else None)
        item_cols.append('_signal_item')
    if 'Station' in df_trans.columns:
        df_trans['_station_item'] = df_trans['Station'].fillna(0).apply(lambda x: 'POI_Station' if x == 1 else None)
        item_cols.append('_station_item')
    if 'Stop' in df_trans.columns:
        df_trans['_stop_item'] = df_trans['Stop'].fillna(0).apply(lambda x: 'POI_Stop' if x == 1 else None)
        item_cols.append('_stop_item')
    
    if '_poi_bin' in df_trans.columns:
        item_cols.append('_poi_bin')
    
    # Day/Night
    if 'Sunrise_Sunset_Encoded' in df_trans.columns:
        # Handle NaN values
        df_trans['_daynight_item'] = df_trans['Sunrise_Sunset_Encoded'].fillna(0).apply(
            lambda x: 'Daytime' if x == 1 else 'Nighttime')
        item_cols.append('_daynight_item')
    elif 'Sunrise_Sunset' in df_trans.columns:
        df_trans['_daynight_item'] = df_trans['Sunrise_Sunset'].astype(str).str.lower().apply(
            lambda x: 'Daytime' if 'day' in x else ('Nighttime' if 'night' in x else None))
        item_cols.append('_daynight_item')
    
    # Geographic
    if '_dist_bin' in df_trans.columns:
        item_cols.append('_dist_bin')
        
    # Severity
    if include_severity and 'Severity' in df_trans.columns:
        # Handle NaN values
        severity_series = df_trans['Severity'].fillna(0).astype(int)
        df_trans['_severity_item'] = 'Severity_' + severity_series.astype(str)
        item_cols.append('_severity_item')
    
    # Build transactions using optimized approach (fastest method)
    print("  Building transactions (ultra-fast)...")
    build_start = time.time()
    
    # Check if we have any item columns
    if len(item_cols) == 0:
        print("[WARNING] No item columns created. Check if required features exist in the dataset.")
        return []
    
    # Convert to string, replace NaN, then build transactions
    item_data = df_trans[item_cols].fillna('').astype(str)
    # Use values for faster access
    item_values = item_data.values
    
    transactions = []
    total_rows = len(item_values)
    for i, row in enumerate(item_values):
        items = [val for val in row if val and val != 'nan' and val != '' and val != 'None']
        if items:
            transactions.append(items)
        # Progress indicator every 10k rows
        if (i + 1) % 10000 == 0:
            progress = ((i + 1) / total_rows) * 100
            print(f"    Progress: {i+1:,}/{total_rows:,} rows ({progress:.1f}%)...")
    
    build_time = time.time() - build_start
    print(f"[OK] Created {len(transactions)} transactions ({build_time:.1f}s)")
    if transactions:
        print(f"  Average items per transaction: {np.mean([len(t) for t in transactions]):.2f}")
    return transactions


def mine_frequent_patterns(transactions, min_support=0.01, use_fpgrowth=True):
    """
    Mine frequent patterns using FP-Growth (faster) or Apriori algorithm
    
    Args:
        transactions: List of transactions
        min_support: Minimum support threshold
        use_fpgrowth: Use FP-Growth if available (much faster than Apriori)
    
    Returns:
        DataFrame with frequent itemsets
    """
    print(f"\n[Pattern Mining] Mining patterns (min_support={min_support})...")
    start_time = time.time()
    
    # Encode transactions
    print("  [Step 1/3] Encoding transactions...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    encode_time = time.time() - start_time
    print(f"  [OK] Encoded {len(transactions):,} transactions with {len(te.columns_)} unique items ({encode_time:.1f}s)")
    
    # Use FP-Growth if available (much faster for large datasets)
    print("  [Step 2/3] Running FP-Growth algorithm...")
    mining_start = time.time()
    
    # Progress indicator flag
    progress_active = [True]  # Use list so it's mutable in nested function
    
    def show_progress():
        """Show elapsed time every 30 seconds while mining"""
        elapsed = 0
        while progress_active[0]:
            time.sleep(30)  # Update every 30 seconds
            if progress_active[0]:
                elapsed += 30
                minutes = elapsed // 60
                seconds = elapsed % 60
                if minutes > 0:
                    print(f"    [Progress] Still mining... ({minutes}m {seconds}s elapsed)")
                else:
                    print(f"    [Progress] Still mining... ({seconds}s elapsed)")
    
    # Start progress thread
    progress_thread = threading.Thread(target=show_progress, daemon=True)
    progress_thread.start()
    
    try:
        if use_fpgrowth:
            try:
                from mlxtend.frequent_patterns import fpgrowth
                print("  Using FP-Growth algorithm (fast)...")
                print("  This may take 2-10 minutes depending on data complexity...")
                frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
            except ImportError:
                print("  FP-Growth not available, using Apriori...")
                print("  This may take 10-30 minutes...")
                frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
        else:
            print("  Using Apriori algorithm...")
            print("  This may take 10-30 minutes...")
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    finally:
        # Stop progress indicator
        progress_active[0] = False
    
    mining_time = time.time() - mining_start
    minutes = int(mining_time // 60)
    seconds = int(mining_time % 60)
    if minutes > 0:
        print(f"  [OK] Pattern mining completed ({minutes}m {seconds}s)")
    else:
        print(f"  [OK] Pattern mining completed ({seconds}s)")
    
    # Process results
    print("  [Step 3/3] Processing results...")
    if len(frequent_itemsets) > 0:
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        # Show breakdown by length
        length_counts = frequent_itemsets['length'].value_counts().sort_index()
        print(f"  Pattern breakdown: {dict(length_counts)}")
    
    total_time = time.time() - start_time
    print(f"[OK] Found {len(frequent_itemsets)} frequent patterns (total time: {total_time:.1f}s)")
    return frequent_itemsets


def generate_association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7, 
                               target_severity=None):
    """
    Generate association rules from frequent patterns
    
    Args:
        frequent_itemsets: DataFrame of frequent itemsets
        metric: Metric to use ('confidence', 'lift', 'support', etc.)
        min_threshold: Minimum threshold for the metric
        target_severity: If specified, only generate rules that predict this severity level
    
    Returns:
        DataFrame with association rules
    """
    print(f"\n[Pattern Mining] Generating association rules (min_{metric}={min_threshold})...")
    start_time = time.time()
    
    if len(frequent_itemsets) == 0:
        print("[WARNING] No frequent itemsets to generate rules from")
        return pd.DataFrame()
    
    print(f"  Processing {len(frequent_itemsets):,} frequent itemsets...")
    try:
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        rule_time = time.time() - start_time
        print(f"  [OK] Rule generation completed ({rule_time:.1f}s)")
        
        if len(rules) == 0:
            print("[WARNING] No rules found with given threshold. Try lowering min_threshold.")
            return pd.DataFrame()
        
        # Filter for severity-specific rules if requested
        if target_severity is not None:
            severity_str = f"Severity_{target_severity}"
            # Rules where severity is in consequents
            rules = rules[rules['consequents'].apply(
                lambda x: any(severity_str in str(item) for item in x)
            )]
            print(f"  Filtered to {len(rules)} rules predicting {severity_str}")
        
        # Sort by lift (strength of association)
        rules = rules.sort_values('lift', ascending=False)
        
        print(f"[OK] Generated {len(rules)} association rules")
        return rules
    except Exception as e:
        print(f"[WARNING] Error generating rules: {e}")
        return pd.DataFrame()

