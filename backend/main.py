import os
import sys

# Add project root to sys path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_data
from src.data.features import add_advanced_features
from src.models.tcn import TCN
from src.models.multi_task import MultiTaskModel

def get_maintenance_recommendation(risk_category, fail_prob, health_score, pred_rul):
    """
    Rule-based recommendations based on model outputs.
    risk_category: 0 (Healthy), 1 (Warning), 2 (Critical)
    """
    if risk_category == 2 or fail_prob > 0.8:
        return {
            'level': 'CRITICAL',
            'color': 'red',
            'emoji': '🚨',
            'message': f"Immediate inspection recommended. Engine is at critical risk of failure within {int(pred_rul)} cycles.",
            'action': "Ground aircraft and schedule immediate overhaul."
        }
    elif risk_category == 1 or (fail_prob > 0.5 and fail_prob <= 0.8):
        return {
            'level': 'WARNING',
            'color': 'orange',
            'emoji': '⚠️',
            'message': f"Schedule maintenance within next service window. Estimated {int(pred_rul)} cycles remaining.",
            'action': "Plan for maintenance check at next available hub."
        }
    else:
        return {
            'level': 'HEALTHY',
            'color': 'green',
            'emoji': '✅',
            'message': f"Normal operation. Engine health is good ({health_score:.1f}%).",
            'action': "Continue standard monitoring."
        }

app = FastAPI(title="Jet Engine Maintenance API")

# Allow all CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Global variables to hold loaded data and model
test_df_cached = None
all_features_cached = None
model_cached = None

def load_data_and_model():
    global test_df_cached, all_features_cached, model_cached
    if test_df_cached is not None:
        return
        
    try:
        # Load Data
        _, test_df, _, feature_cols, _ = preprocess_data(data_dir=DATA_DIR, fd_num=1)
        test_df, new_features = add_advanced_features(test_df, feature_cols)
        all_features_cached = feature_cols + new_features
        test_df_cached = test_df
        
        # Load Model
        base_tcn = TCN(input_dim=len(all_features_cached), num_channels=[64, 64], kernel_size=5, dropout=0.137)
        model_cached = MultiTaskModel(base_tcn)
        model_path = os.path.join(MODELS_DIR, 'pytorch_run_best.pt')
        model_cached.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model_cached.eval()
    except Exception as e:
        print(f"Error loading model or data: {e}")
        raise e

@app.on_event("startup")
def startup_event():
    load_data_and_model()

@app.get("/api/engines")
def get_engine_ids():
    """Return a list of all available engine IDs."""
    if test_df_cached is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    engines = test_df_cached['unit_id'].unique().tolist()
    return {"engines": engines}

fleet_cache = None

@app.get("/api/fleet")
def get_fleet_status():
    """Return the latest status of all engines."""
    global fleet_cache
    if fleet_cache is not None:
        return {"fleet": fleet_cache}
        
    if test_df_cached is None or model_cached is None:
        raise HTTPException(status_code=500, detail="Model/Data not loaded")
        
    engines = test_df_cached['unit_id'].unique()
    seq_len = 30
    
    # We will build a batch of the latest sequence for each engine
    latest_seqs = []
    engine_infos = []
    
    for engine_id in engines:
        unit_data = test_df_cached[test_df_cached['unit_id'] == engine_id]
        cycles = unit_data['cycles'].values
        current_cycle = int(cycles[-1])
        
        pad_len = seq_len - 1
        padded_features = np.vstack([np.zeros((pad_len, len(all_features_cached))), unit_data[all_features_cached].values])
        
        # Take the last seq_len window
        latest_seq = padded_features[-seq_len:]
        latest_seqs.append(latest_seq)
        engine_infos.append({"id": int(engine_id), "currentCycles": current_cycle})
        
    X_tensor = torch.tensor(np.array(latest_seqs), dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model_cached(X_tensor)
        rul_preds = outputs['rul'].cpu().numpy().flatten()
        health_scores = outputs['health_score'].cpu().numpy().flatten()
        risk_logits = outputs['risk_logits']
        risk_categories = torch.argmax(risk_logits, dim=1).cpu().numpy()
        
    fleet_data = []
    for i, info in enumerate(engine_infos):
        pred_rul = max(0, float(rul_preds[i]))
        health_score = max(0, min(100, float(health_scores[i])))
        risk_cat = int(risk_categories[i])
        
        status = 'Healthy'
        if risk_cat == 2 or pred_rul <= 15:
            status = 'Critical'
        elif risk_cat == 1 or pred_rul <= 30:
            status = 'Warning'
            
        fleet_data.append({
            "id": info["id"],
            "currentCycles": info["currentCycles"],
            "predictedRUL": int(pred_rul),
            "healthScore": round(health_score, 1),
            "status": status
        })
        
    fleet_cache = fleet_data
    return {"fleet": fleet_data}

@app.get("/api/engine/{engine_id}")
def get_engine_predictions(engine_id: int):
    """Return live predictions and history for a specific engine."""
    if test_df_cached is None or model_cached is None:
        raise HTTPException(status_code=500, detail="Model/Data not loaded")
        
    unit_data = test_df_cached[test_df_cached['unit_id'] == engine_id].copy()
    if len(unit_data) == 0:
        raise HTTPException(status_code=404, detail="Engine not found")
        
    seq_len = 30
    cycles = unit_data['cycles'].values
    
    pad_len = seq_len - 1
    padded_features = np.vstack([np.zeros((pad_len, len(all_features_cached))), unit_data[all_features_cached].values])
    
    X_seqs = []
    for i in range(len(unit_data)):
        seq = padded_features[i : i + seq_len]
        X_seqs.append(seq)
        
    X_tensor = torch.tensor(np.array(X_seqs), dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model_cached(X_tensor)
        rul_preds = outputs['rul'].cpu().numpy().flatten()
        fail_probs = outputs['failure_prob'].cpu().numpy().flatten()
        health_scores = outputs['health_score'].cpu().numpy().flatten()
        risk_logits = outputs['risk_logits']
        risk_categories = torch.argmax(risk_logits, dim=1).cpu().numpy()
        
    # Get latest prediction
    current_cycle = int(cycles[-1])
    pred_rul = float(rul_preds[-1])
    health_score = float(health_scores[-1])
    fail_prob = float(fail_probs[-1])
    risk_category = int(risk_categories[-1])
    
    recommendation = get_maintenance_recommendation(risk_category, fail_prob, health_score, pred_rul)
    
    # Format history for charting
    history = []
    # unit_data is a DataFrame, cycles is an array. They should align by index.
    # To safely get sensor data, we can use iloc.
    s2_vals = unit_data['s2'].values if 's2' in unit_data.columns else np.zeros(len(cycles))
    s3_vals = unit_data['s3'].values if 's3' in unit_data.columns else np.zeros(len(cycles))
    
    for i in range(len(cycles)):
        history.append({
            "cycle": int(cycles[i]),
            "rul": max(0, float(rul_preds[i])),
            "health_score": float(health_scores[i]),
            "s2": float(s2_vals[i]),
            "s3": float(s3_vals[i])
        })
        
    return {
        "engine_id": engine_id,
        "current_cycle": current_cycle,
        "prediction": {
            "rul": max(0, pred_rul),
            "health_score": min(100, max(0, health_score)),
            "failure_probability": min(100, max(0, fail_prob * 100)),
            "risk_category": risk_category
        },
        "recommendation": recommendation,
        "history": history
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
