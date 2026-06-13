import pandas as pd
import numpy as np

def generate_rolling_features(df, feature_cols, windows=[5, 10, 20]):
    """
    Generates rolling mean, std, min, max for given sensor features.
    """
    new_features = {}
    for w in windows:
        for col in feature_cols:
            grouped = df.groupby('unit_id')[col]
            new_features[f'{col}_rolling_mean_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
            new_features[f'{col}_rolling_std_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
            new_features[f'{col}_rolling_min_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).min())
            new_features[f'{col}_rolling_max_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).max())
            
    # Concatenate all new features at once to avoid fragmentation
    new_df = pd.DataFrame(new_features)
    return pd.concat([df, new_df], axis=1)

def generate_trend_features(df, feature_cols):
    """
    Generates first and second-order differences.
    """
    new_features = {}
    for col in feature_cols:
        grouped = df.groupby('unit_id')[col]
        diff1 = grouped.diff().fillna(0)
        new_features[f'{col}_diff1'] = diff1
        # Second diff requires grouping the first diff
        # We can simulate this easily since diff1 is just a series, but we must group it by the original unit_id
        new_features[f'{col}_diff2'] = diff1.groupby(df['unit_id']).diff().fillna(0)
        
    new_df = pd.DataFrame(new_features)
    return pd.concat([df, new_df], axis=1)

def add_advanced_features(df, feature_cols):
    """
    Master function to add all engineered features.
    """
    df = generate_rolling_features(df, feature_cols)
    df = generate_trend_features(df, feature_cols)
    
    # Statistical features on raw sensors per cycle
    new_features = {
        'sensor_mean': df[feature_cols].mean(axis=1),
        'sensor_std': df[feature_cols].std(axis=1),
        'sensor_max': df[feature_cols].max(axis=1),
        'sensor_min': df[feature_cols].min(axis=1),
        'sensor_skew': df[feature_cols].skew(axis=1)
    }
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    
    # Get the newly created feature columns
    all_new_features = [col for col in df.columns if col not in ['unit_id', 'cycles', 'RUL', 'failure_prob_label', 'risk_category', 'health_score']]
    
    return df, all_new_features
