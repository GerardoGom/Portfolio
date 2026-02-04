"""
CS 470 Final Project - Feature Engineering Module
Author: Robert Jarman
Student ID: 2547392

This module handles all feature extraction, selection, and preprocessing.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

def extract_temporal_features(df):
    """
    Extract temporal features from Start_Time and End_Time
    Primary responsibility: Robert
    
    Features created:
    - Year, Month, Day, Hour
    - DayOfWeek (0=Monday, 6=Sunday)
    - DayOfWeek_Name
    - Weekend (binary)
    - Season (Winter/Spring/Summer/Fall)
    - TimeOfDay (Morning/Afternoon/Evening/Night)
    - RushHour (binary)
    - Duration_Minutes
    - Quarter
    """
    print("\n[Feature Engineering] Extracting temporal features...")
    
    # Convert to datetime
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time'])
    
    # Basic temporal components
    df['Year'] = df['Start_Time'].dt.year
    df['Month'] = df['Start_Time'].dt.month
    df['Day'] = df['Start_Time'].dt.day
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
    df['DayOfWeek_Name'] = df['Start_Time'].dt.day_name()
    df['Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Season extraction
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['Season'] = df['Month'].apply(get_season)
    
    # Time of day categories
    def categorize_time_of_day(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df['TimeOfDay'] = df['Hour'].apply(categorize_time_of_day)
    
    # Rush hour indicator (7-9 AM, 4-6 PM)
    df['RushHour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9) | 
                      (df['Hour'] >= 16) & (df['Hour'] <= 18)).astype(int)
    
    # Accident duration in minutes
    df['Duration_Minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    
    # Quarter of year
    df['Quarter'] = df['Start_Time'].dt.quarter
    
    print(f"Created 12 temporal features")
    return df


def process_poi_features(df):
    """
    Process Point of Interest (POI) features
    Responsibility: Robert
    
    POI columns: Amenity, Bump, Crossing, Give_Way, Junction, No_Exit,
                 Railway, Roundabout, Station, Stop, Traffic_Calming,
                 Traffic_Signal, Turning_Loop
    """
    print("\n[Feature Engineering] Processing POI features...")
    
    poi_columns = [
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
        'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
        'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
    ]
    
    # Convert to integer (0/1)
    for col in poi_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Create aggregate POI feature
    df['Total_POI_Count'] = df[poi_columns].sum(axis=1)
    
    # Categorize by POI density
    df['POI_Density'] = pd.cut(df['Total_POI_Count'], 
                                bins=[-1, 0, 2, 5, 20], 
                                labels=['None', 'Low', 'Medium', 'High'])
    
    # Drop Turning_Loop if all values are False
    if 'Turning_Loop' in df.columns and df['Turning_Loop'].sum() == 0:
        df = df.drop('Turning_Loop', axis=1)
        print("  - Dropped 'Turning_Loop' (all values False)")
    
    print(f"Processed {len(poi_columns)} POI features")
    print(f"Created 'Total_POI_Count' and 'POI_Density'")
    
    return df


def process_twilight_features(df):
    """
    Process twilight/daylight features
    Responsibility: Robert
    
    Columns: Sunrise_Sunset, Civil_Twilight, Nautical_Twilight, Astronomical_Twilight
    """
    print("\n[Feature Engineering] Processing twilight features...")
    
    twilight_columns = ['Sunrise_Sunset', 'Civil_Twilight', 
                        'Nautical_Twilight', 'Astronomical_Twilight']
    
    for col in twilight_columns:
        if col in df.columns:
            # Encode: Day=1, Night=0
            df[col + '_Encoded'] = (df[col] == 'Day').astype(int)
    
    print(f"Encoded {len(twilight_columns)} twilight features")
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset
    Team responsibility
    """
    print("\n[Feature Engineering] Handling missing values...")
    
    initial_missing = df.isnull().sum().sum()
    
    # Weather features - fill with median
    weather_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
                    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    
    for col in weather_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # Categorical features - fill with mode
    if 'Wind_Direction' in df.columns:
        mode_val = df['Wind_Direction'].mode()[0]
        df['Wind_Direction'].fillna(mode_val, inplace=True)
    
    if 'Weather_Condition' in df.columns:
        df['Weather_Condition'].fillna('Clear', inplace=True)
    
    final_missing = df.isnull().sum().sum()
    print(f"Reduced missing values: {initial_missing:,} → {final_missing:,}")
    
    return df


def encode_categorical_features(df):
    """
    Encode categorical features for clustering
    """
    print("\n[Feature Engineering] Encoding categorical features...")
    
    # State - create accident frequency feature
    if 'State' in df.columns:
        state_freq = df['State'].value_counts()
        df['State_Accident_Frequency'] = df['State'].map(state_freq)
    
    # Weather_Condition - label encode
    if 'Weather_Condition' in df.columns:
        le = LabelEncoder()
        df['Weather_Condition_Encoded'] = le.fit_transform(df['Weather_Condition'].astype(str))
    
    # Wind_Direction - encode cardinal directions
    if 'Wind_Direction' in df.columns:
        direction_map = {
            'N': 0, 'NNE': 1, 'NE': 2, 'ENE': 3,
            'E': 4, 'ESE': 5, 'SE': 6, 'SSE': 7,
            'S': 8, 'SSW': 9, 'SW': 10, 'WSW': 11,
            'W': 12, 'WNW': 13, 'NW': 14, 'NNW': 15,
            'CALM': 16, 'VAR': 17
        }
        df['Wind_Direction_Encoded'] = df['Wind_Direction'].map(direction_map)
        df['Wind_Direction_Encoded'].fillna(16, inplace=True)
    
    # Season encoding
    if 'Season' in df.columns:
        df['Season_Encoded'] = df['Season'].map({
            'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3
        })
    
    # TimeOfDay encoding
    if 'TimeOfDay' in df.columns:
        df['TimeOfDay_Encoded'] = df['TimeOfDay'].map({
            'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3
        })
    
    print("Categorical features encoded")
    return df


def select_features_for_clustering(df):
    """
    Select final features for clustering algorithms
    Returns: DataFrame with selected features, list of feature names
    """
    print("\n[Feature Engineering] Selecting features for clustering...")
    
    # Define clustering features
    clustering_features = [
        # Severity
        'Severity',
        
        # Geographic
        'Start_Lat',
        'Start_Lng',
        'Distance(mi)',
        
        # Temporal
        'Hour',
        'DayOfWeek',
        'Month',
        'Season_Encoded',
        'Weekend',
        'RushHour',
        'Quarter',
        
        # Weather
        'Temperature(F)',
        'Humidity(%)',
        'Pressure(in)',
        'Visibility(mi)',
        'Wind_Speed(mph)',
        'Precipitation(in)',
        'Weather_Condition_Encoded',
        
        # POI Features
        'Crossing',
        'Junction',
        'Traffic_Signal',
        'Station',
        'Stop',
        'Total_POI_Count',
        
        # Twilight
        'Sunrise_Sunset_Encoded',
    ]
    
    # Filter to existing columns
    available_features = [col for col in clustering_features if col in df.columns]
    
    df_clustering = df[available_features].copy()
    
    # Handle any remaining NaN
    df_clustering = df_clustering.fillna(df_clustering.median())
    
    print(f"Selected {len(available_features)} features for clustering")
    print(f"  Features: {', '.join(available_features[:5])}...")
    
    return df_clustering, available_features


def scale_features(df_clustering):
    """
    Scale features using StandardScaler
    Returns: scaled DataFrame, fitted scaler
    """
    print("\n[Feature Engineering] Scaling features...")
    
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_clustering),
        columns=df_clustering.columns,
        index=df_clustering.index
    )
    
    print("Features scaled (mean=0, std=1)")
    return df_scaled, scaler


def drop_unnecessary_columns(df):
    """
    Drop columns that are not needed for analysis
    """
    print("\n[Feature Engineering] Dropping unnecessary columns...")
    
    columns_to_drop = [
        'ID',  # Just an identifier
        'End_Lat', 'End_Lng',  # 44% missing
        'Wind_Chill(F)',  # 26% missing, correlated with Temperature
        'Weather_Timestamp',  # Redundant with Start_Time
        'Timezone',  # Redundant with State
        'Airport_Code',  # Not useful for clustering
        'Description',  # Text field
        'Street',  # Too many unique values
        'Zipcode',  # Too granular
        'Country',  # All USA
        'Source',  # Not useful for analysis
    ]
    
    initial_cols = df.shape[1]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], 
                 errors='ignore')
    dropped_count = initial_cols - df.shape[1]
    
    print(f"Dropped {dropped_count} columns")
    return df

