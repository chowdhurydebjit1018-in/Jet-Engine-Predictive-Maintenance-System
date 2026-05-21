import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

def load_cmapss_data(train_path, test_path, rul_path=None):
    cols = ['unit_id', 'cycles', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=cols)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=cols)
    if rul_path:
        rul_true = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
    else:
        rul_true = None
    return train_df, test_df, rul_true

def remove_constant_sensors(df):
    constant_sensors = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
    df.drop(columns=constant_sensors, inplace=True)
    return df

def add_rul_column(df, max_rul_cap=125):
    max_cycles = df.groupby('unit_id')['cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id', how='left')
    df['RUL'] = df['max_cycle'] - df['cycles']
    df['RUL'] = df['RUL'].clip(upper=max_rul_cap)
    df.drop(columns=['max_cycle'], inplace=True)
    return df
