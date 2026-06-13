import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Avoid division by zero for MAPE
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def evaluate_classification(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = 0.5 # Default if only one class present
        
    return {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC_AUC': auc
    }

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted RUL"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y_true)), y=y_true, mode='lines', name='Actual RUL'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_pred)), y=y_pred, mode='lines', name='Predicted RUL'))
    fig.update_layout(title=title, xaxis_title="Sample", yaxis_title="RUL")
    return fig

def plot_error_distribution(y_true, y_pred):
    errors = y_true - y_pred
    fig = go.Figure(data=[go.Histogram(x=errors, nbinsx=50)])
    fig.update_layout(title="Prediction Error Distribution", xaxis_title="Error (Actual - Predicted)", yaxis_title="Count")
    return fig
