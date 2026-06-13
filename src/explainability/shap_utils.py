import shap
import torch
import numpy as np

def explain_dl_model(model, X_train_sample, X_test_sample, device='cpu'):
    """
    Generates SHAP values for a PyTorch Deep Learning model.
    Since SHAP for PyTorch timeseries can be complex, we use DeepExplainer.
    """
    model.eval()
    model.to(device)
    
    # Convert numpy arrays to tensors
    X_train_tensor = torch.tensor(X_train_sample, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_sample, dtype=torch.float32).to(device)
    
    # We wrap the model to only output the RUL for SHAP analysis
    class RULWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)['rul']
            
    wrapped_model = RULWrapper(model)
    
    explainer = shap.DeepExplainer(wrapped_model, X_train_tensor)
    shap_values = explainer.shap_values(X_test_tensor)
    
    return shap_values

def explain_xgb_model(model, X_train, X_test):
    """
    Generates SHAP values for an XGBoost model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values

def get_shap_summary_plot(shap_values, features, feature_names):
    """
    Returns a SHAP summary plot figure.
    """
    # For Streamlit, we typically use matplotlib or just shap built-in plot
    import matplotlib.pyplot as plt
    fig = plt.figure()
    shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
    return fig

def get_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, instance_idx=0):
    """
    Returns a SHAP waterfall plot for a single prediction.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        expected_value, 
        shap_values[instance_idx], 
        feature_names=feature_names,
        show=False
    )
    return fig
