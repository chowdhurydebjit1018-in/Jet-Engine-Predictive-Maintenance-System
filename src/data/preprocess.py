import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

COLUMN_NAMES = ['unit_id', 'cycles', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]

def load_cmapss_data(data_dir, fd_num):
    """
    Loads CMAPSS data for a specific FD dataset (1-4).
    """
    train_file = os.path.join(data_dir, f'train_FD00{fd_num}.txt')
    test_file = os.path.join(data_dir, f'test_FD00{fd_num}.txt')
    rul_file = os.path.join(data_dir, f'RUL_FD00{fd_num}.txt')
    
    train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=COLUMN_NAMES, engine='python')
    test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=COLUMN_NAMES, engine='python')
    rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL_true'], engine='python')
    
    return train_df, test_df, rul_df

def add_rul(df, max_rul=125):
    """
    Adds Remaining Useful Life (RUL) column to the dataframe.
    Values are clipped to max_rul (piecewise linear degradation).
    """
    max_cycles = df.groupby('unit_id')['cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycles']
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    df = df.drop(columns=['max_cycle'])
    return df

def add_failure_labels(df, warning_window=30, critical_window=15):
    """
    Adds classification labels for multi-task learning.
    risk_category: 0 (Healthy), 1 (Warning), 2 (Critical)
    failure_prob_label: 1 if RUL <= warning_window else 0
    """
    df['failure_prob_label'] = np.where(df['RUL'] <= warning_window, 1, 0)
    
    conditions = [
        (df['RUL'] <= critical_window),
        (df['RUL'] > critical_window) & (df['RUL'] <= warning_window)
    ]
    choices = [2, 1] # 2 = Critical, 1 = Warning
    df['risk_category'] = np.select(conditions, choices, default=0) # 0 = Healthy
    
    # Calculate health score (0-100%)
    df['health_score'] = (df['RUL'] / df['RUL'].max()) * 100
    return df

def create_sequences(df, seq_length, feature_cols, target_cols=None):
    """
    Creates sliding window sequences of length `seq_length` for each engine.
    """
    X = []
    y_dict = {col: [] for col in (target_cols or [])}
    
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id]
        
        if len(unit_data) >= seq_length:
            unit_features = unit_data[feature_cols].values
            
            for i in range(len(unit_data) - seq_length + 1):
                X.append(unit_features[i : i + seq_length])
                
                if target_cols:
                    for col in target_cols:
                        y_dict[col].append(unit_data.iloc[i + seq_length - 1][col])
                        
    X = np.array(X)
    if target_cols:
        y_dict = {col: np.array(vals) for col, vals in y_dict.items()}
        return X, y_dict
    
    return X

def create_test_sequences(df, seq_length, feature_cols):
    """
    Creates only the last sequence for each engine for evaluation against true RUL.
    Pads with zeros if the engine data is shorter than seq_length.
    """
    X = []
    
    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id]
        
        if len(unit_data) >= seq_length:
            X.append(unit_data[feature_cols].values[-seq_length:])
        else:
            pad = np.zeros((seq_length - len(unit_data), len(feature_cols)))
            padded_seq = np.vstack([pad, unit_data[feature_cols].values])
            X.append(padded_seq)
            
    return np.array(X)

def preprocess_data(data_dir="data", fd_num=1, max_rul=125):
    """
    Full preprocessing pipeline for a single FD dataset.
    """
    train_df, test_df, rul_df = load_cmapss_data(data_dir, fd_num)
    
    # 1. Add RUL and Labels to train
    train_df = add_rul(train_df, max_rul)
    train_df = add_failure_labels(train_df)
    
    # 2. Drop constant sensors (varies slightly by FD, but generally these are constant)
    # We will identify them dynamically or use the standard list.
    # For robust handling across FD001-FD004, we drop columns with 0 variance in training.
    sensor_cols = [col for col in train_df.columns if col.startswith('s')]
    constant_cols = [col for col in sensor_cols if train_df[col].std() < 1e-4]
    
    train_df = train_df.drop(columns=constant_cols)
    test_df = test_df.drop(columns=constant_cols)
    
    # Remaining features
    feature_cols = [col for col in train_df.columns if col.startswith('s') or col.startswith('os')]
    
    # 3. Scale features
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_df, test_df, rul_df, feature_cols, scaler

if __name__ == "__main__":
    train, test, rul, features, scaler = preprocess_data(fd_num=1)
    print(f"Train shape: {train.shape}")
    print(f"Features: {features}")
