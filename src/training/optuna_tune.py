import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
import os

from src.models.tcn import TCN
from src.models.multi_task import MultiTaskModel
from src.training.evaluate import evaluate_regression, evaluate_classification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def objective(trial, X_train, y_train_dict, X_val, y_val_dict, device='cpu'):
    """
    Optuna objective function for tuning a TCN-based Multi-Task model.
    Constrained to find models with >95% accuracy and <13 RMSE.
    """
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    
    # We test different depth/width configurations
    channel_configs = [
        [32, 64],
        [64, 64],
        [64, 128],
        [64, 128, 128],
        [32, 64, 128]
    ]
    channel_idx = trial.suggest_int('channel_idx', 0, len(channel_configs) - 1)
    num_channels = channel_configs[channel_idx]
    
    # Model Setup
    input_dim = X_train.shape[2]
    base_model = TCN(input_dim=input_dim, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
    model = MultiTaskModel(base_model).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    criterion_rul = nn.MSELoss()
    criterion_failure = nn.BCELoss()
    criterion_health = nn.MSELoss()
    criterion_risk = nn.CrossEntropyLoss()
    
    # Data Setup
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_rul = torch.tensor(y_train_dict['rul'].values, dtype=torch.float32).unsqueeze(1)
    y_train_fail = torch.tensor(y_train_dict['failure_prob'].values, dtype=torch.float32).unsqueeze(1)
    y_train_health = torch.tensor(y_train_dict['health_score'].values, dtype=torch.float32).unsqueeze(1)
    y_train_risk = torch.tensor(y_train_dict['risk_category'].values, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, y_train_rul, y_train_fail, y_train_health, y_train_risk)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_rul = y_val_dict['rul'].values
    y_val_fail = y_val_dict['failure_prob'].values
    
    epochs = 15
    best_rmse = float('inf')
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_rul, batch_fail, batch_health, batch_risk in train_loader:
            batch_x = batch_x.to(device)
            batch_rul, batch_fail = batch_rul.to(device), batch_fail.to(device)
            batch_health, batch_risk = batch_health.to(device), batch_risk.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss_rul = criterion_rul(outputs['rul'], batch_rul)
            loss_fail = criterion_failure(outputs['failure_prob'], batch_fail)
            loss_health = criterion_health(outputs['health_score'], batch_health)
            loss_risk = criterion_risk(outputs['risk_logits'], batch_risk)
            
            total_loss = loss_rul + 100 * loss_fail + loss_health + 10 * loss_risk
            total_loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds_rul = val_outputs['rul'].cpu().numpy().flatten()
            val_preds_fail = val_outputs['failure_prob'].cpu().numpy().flatten()
            
            reg_metrics = evaluate_regression(y_val_rul, val_preds_rul)
            clf_metrics = evaluate_classification(y_val_fail, val_preds_fail)
            
            rmse = reg_metrics['RMSE']
            acc = clf_metrics['Accuracy']
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_acc = acc
                
        trial.report(rmse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    # Constraint penalty: if accuracy is below 95%, penalize the score heavily
    if best_acc < 0.95:
        # Penalize by adding 100 + amount of accuracy missed
        penalty = 100.0 + (0.95 - best_acc) * 1000
        return best_rmse + penalty
        
    return best_rmse

def run_optimization():
    from src.data.preprocess import preprocess_data, create_sequences
    from src.data.features import add_advanced_features
    from sklearn.model_selection import train_test_split
    
    logging.info("Loading real data for Optuna...")
    train_df, _, _, feature_cols, _ = preprocess_data(data_dir="data", fd_num=1)
    train_df, new_features = add_advanced_features(train_df, feature_cols)
    all_features = feature_cols + new_features
    
    seq_len = 30
    target_cols = ['RUL', 'failure_prob_label', 'health_score', 'risk_category']
    X_all, y_all_dict_raw = create_sequences(train_df, seq_len, all_features, target_cols)
    
    y_all_dict = {
        'rul': pd.Series(y_all_dict_raw['RUL']),
        'failure_prob': pd.Series(y_all_dict_raw['failure_prob_label']),
        'health_score': pd.Series(y_all_dict_raw['health_score']),
        'risk_category': pd.Series(y_all_dict_raw['risk_category'])
    }
    
    indices = np.arange(len(X_all))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    y_train_dict = {k: v.iloc[train_idx].reset_index(drop=True) for k, v in y_all_dict.items()}
    y_val_dict = {k: v.iloc[val_idx].reset_index(drop=True) for k, v in y_all_dict.items()}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logging.info("Starting Optuna Study...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train_dict, X_val, y_val_dict, device=device), n_trials=15)
    
    logging.info("Best trial:")
    trial = study.best_trial
    logging.info(f"  Value (RMSE): {trial.value}")
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")

if __name__ == "__main__":
    run_optimization()
