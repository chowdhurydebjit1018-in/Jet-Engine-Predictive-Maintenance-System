import mlflow
import os

def setup_mlflow(experiment_name="Jet_Engine_Predictive_Maintenance", tracking_uri="sqlite:///mlflow.db"):
    """
    Sets up the MLflow experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

def log_params(params):
    mlflow.log_params(params)

def log_metrics(metrics, step=None):
    mlflow.log_metrics(metrics, step=step)

def log_model(model, artifact_path="model"):
    # Determine type of model
    if hasattr(model, 'save_model'):
        # XGBoost
        mlflow.xgboost.log_model(model, artifact_path)
    else:
        # PyTorch
        mlflow.pytorch.log_model(model, artifact_path)
