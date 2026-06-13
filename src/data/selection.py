import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import logging

def select_features_mi(X, y, top_k=50):
    """
    Selects top features based on Mutual Information Regression.
    """
    logging.info("Calculating Mutual Information for features...")
    # Sample data if too large to speed up MI
    if len(X) > 10000:
        sample_idx = np.random.choice(len(X), 10000, replace=False)
        X_sample, y_sample = X.iloc[sample_idx], y.iloc[sample_idx]
    else:
        X_sample, y_sample = X, y
        
    mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns)
    
    top_features = mi_series.nlargest(top_k).index.tolist()
    return top_features, mi_series

def select_features_rf(X, y, top_k=50):
    """
    Selects top features based on Random Forest feature importance.
    """
    logging.info("Calculating Random Forest importance for features...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    
    # Sample to speed up
    if len(X) > 20000:
        sample_idx = np.random.choice(len(X), 20000, replace=False)
        X_sample, y_sample = X.iloc[sample_idx], y.iloc[sample_idx]
    else:
        X_sample, y_sample = X, y
        
    rf.fit(X_sample, y_sample)
    
    rf_scores = rf.feature_importances_
    rf_series = pd.Series(rf_scores, index=X.columns)
    
    top_features = rf_series.nlargest(top_k).index.tolist()
    return top_features, rf_series

def combined_feature_selection(X, y, top_k=50):
    """
    Combines MI and RF to select the most robust features.
    We take the intersection of the top features, or pad with the highest ranked if we need more.
    """
    top_mi, _ = select_features_mi(X, y, top_k=int(top_k * 1.5))
    top_rf, _ = select_features_rf(X, y, top_k=int(top_k * 1.5))
    
    # Intersection of top features
    intersection = list(set(top_mi) & set(top_rf))
    
    if len(intersection) >= top_k:
        return intersection[:top_k]
    else:
        # Add from RF if we need more
        for f in top_rf:
            if f not in intersection and len(intersection) < top_k:
                intersection.append(f)
        return intersection
