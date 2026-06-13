import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
import os

from src.training.mlops import setup_mlflow, log_params, log_metrics, log_model
from src.training.evaluate import evaluate_regression, evaluate_classification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_pytorch_model(model, X_train, y_train_dict, X_val, y_val_dict, epochs=50, batch_size=64, lr=0.001, device='cpu', run_name="pytorch_run"):
    """
    Trains a PyTorch Multi-Task model.
    y_train_dict should contain: 'rul', 'failure_prob', 'health_score', 'risk_logits'
    """
    setup_mlflow()
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions
    criterion_rul = nn.MSELoss()
    criterion_failure = nn.BCELoss()
    criterion_health = nn.MSELoss()
    criterion_risk = nn.CrossEntropyLoss()
    
    # Create DataLoaders
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_rul = torch.tensor(y_train_dict['rul'].values, dtype=torch.float32).unsqueeze(1)
    y_train_fail = torch.tensor(y_train_dict['failure_prob'].values, dtype=torch.float32).unsqueeze(1)
    y_train_health = torch.tensor(y_train_dict['health_score'].values, dtype=torch.float32).unsqueeze(1)
    y_train_risk = torch.tensor(y_train_dict['risk_category'].values, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_rul, y_train_fail, y_train_health, y_train_risk)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation tensors
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_rul = y_val_dict['rul'].values
    y_val_fail = y_val_dict['failure_prob'].values
    
    import mlflow
    with mlflow.start_run(run_name=run_name):
        log_params({'epochs': epochs, 'batch_size': batch_size, 'lr': lr, 'model_type': model.base_model.__class__.__name__})
        
        best_val_rmse = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
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
                
                # Multi-task loss
                total_loss = loss_rul + 100 * loss_fail + loss_health + 10 * loss_risk
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_preds_rul = val_outputs['rul'].cpu().numpy().flatten()
                val_preds_fail = val_outputs['failure_prob'].cpu().numpy().flatten()
                
                reg_metrics = evaluate_regression(y_val_rul, val_preds_rul)
                clf_metrics = evaluate_classification(y_val_fail, val_preds_fail)
                
                rmse = reg_metrics['RMSE']
                acc = clf_metrics['Accuracy']
                
                log_metrics({'train_loss': train_loss, 'val_rmse': rmse, 'val_acc': acc}, step=epoch)
                
                if epoch % 5 == 0:
                    logging.info(f"Epoch {epoch}/{epochs} - Loss: {train_loss:.4f} - Val RMSE: {rmse:.4f} - Val Acc: {acc:.4f}")
                    
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    # Save best model logic can go here
                    os.makedirs('models', exist_ok=True)
                    torch.save(model.state_dict(), f"models/{run_name}_best.pt")
                    
        # Log the best model
        model.load_state_dict(torch.load(f"models/{run_name}_best.pt"))
        log_model(model, artifact_path="model")
        
        logging.info(f"Training completed. Best Val RMSE: {best_val_rmse:.4f}")
        return model

def train_xgboost(model, X_train, y_train, X_val, y_val, run_name="xgboost_run"):
    """
    Trains an XGBoost model.
    """
    setup_mlflow()
    import mlflow
    
    with mlflow.start_run(run_name=run_name):
        log_params(model.get_params())
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_val)
        reg_metrics = evaluate_regression(y_val, preds)
        
        log_metrics(reg_metrics)
        log_model(model, artifact_path="model")
        
        logging.info(f"XGBoost Training completed. Val RMSE: {reg_metrics['RMSE']:.4f}")
        return model

if __name__ == "__main__":
    import sys
    from src.data.preprocess import preprocess_data
    from src.data.features import add_advanced_features
    from src.models.tcn import TCN
    from src.models.multi_task import MultiTaskModel
    
    logging.info("Starting training pipeline...")
    
    # 1. Load and preprocess data (using FD001 as example)
    try:
        train_df, test_df, rul_df, feature_cols, scaler = preprocess_data(data_dir="data", fd_num=1)
    except Exception as e:
        logging.error(f"Failed to load data. Did you run the download script? Error: {e}")
        sys.exit(1)
        
    # 2. Add advanced features
    train_df, new_features = add_advanced_features(train_df, feature_cols)
    all_features = feature_cols + new_features
    
    # 3. Create Sequences
    from src.data.preprocess import create_sequences
    from sklearn.model_selection import train_test_split
    
    seq_len = 30
    logging.info(f"Generating sequences of length {seq_len}...")
    target_cols = ['RUL', 'failure_prob_label', 'health_score', 'risk_category']
    
    X_all, y_all_dict_raw = create_sequences(train_df, seq_len, all_features, target_cols)
    
    # Convert dict of numpy arrays to dict of pandas Series for the training function
    y_all_dict = {
        'rul': pd.Series(y_all_dict_raw['RUL']),
        'failure_prob': pd.Series(y_all_dict_raw['failure_prob_label']),
        'health_score': pd.Series(y_all_dict_raw['health_score']),
        'risk_category': pd.Series(y_all_dict_raw['risk_category'])
    }
    
    # Split into train and validation sets
    indices = np.arange(len(X_all))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train_real = X_all[train_idx]
    X_val_real = X_all[val_idx]
    
    y_train_dict_real = {k: v.iloc[train_idx].reset_index(drop=True) for k, v in y_all_dict.items()}
    y_val_dict_real = {k: v.iloc[val_idx].reset_index(drop=True) for k, v in y_all_dict.items()}
    
    # 3. Instantiate model
    logging.info("Building model...")
    base_tcn = TCN(input_dim=len(all_features), num_channels=[64, 64], kernel_size=5, dropout=0.137)
    mt_model = MultiTaskModel(base_tcn)
    
    # 4. Train
    logging.info("Starting PyTorch training on real data...")
    trained_model = train_pytorch_model(
        mt_model, X_train_real, y_train_dict_real, X_val_real, y_val_dict_real,
        epochs=15, batch_size=128, lr=0.005
    )
    
    # Save the scaler and features list for inference
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(all_features, 'models/features.pkl')
    logging.info("Training pipeline finished successfully!")
