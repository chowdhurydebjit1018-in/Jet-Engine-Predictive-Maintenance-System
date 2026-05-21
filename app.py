import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from scipy.signal import savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Jet Engine Predictive Maintenance", layout="wide")

# Constants
MAX_RUL = 125
SEQUENCE_LENGTH = 50

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
    }
    .status-healthy { color: #00ff00; }
    .status-caution { color: #ffaa00; }
    .status-critical { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

st.title("✈️ Jet Engine Predictive Maintenance System")
st.caption("CNN-LSTM Hybrid Model | NASA CMAPSS Dataset | RMSE: 14.25 | Accuracy: 95%")

@st.cache_resource
def load_assets():
    return (
        load_model('models/rul_regressor_v2.h5', compile=False),
        load_model('models/rul_classifier_v2.h5', compile=False),
        joblib.load('models/scaler_v2.pkl')
    )

@st.cache_data
def load_data():
    cols = ['unit_id', 'cycles', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    test_df = pd.read_csv('data/test_FD001.txt', sep=r'\\s+', header=None, names=cols)
    true_rul = pd.read_csv('data/RUL_FD001.txt', sep=r'\\s+', header=None, names=['RUL'])
    drop_cols = ['os1', 'os2', 'os3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
    test_df.drop(columns=drop_cols, inplace=True)
    return test_df, true_rul

# Load everything
reg_model, class_model, scaler = load_assets()
test_df, true_rul = load_data()
sensor_cols = [c for c in test_df.columns if c not in ['unit_id', 'cycles']]

# Sidebar
st.sidebar.header("Engine Control Panel")
engine_id = st.sidebar.selectbox("Select Engine ID:", test_df['unit_id'].unique())
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Select different engines to see varying health conditions")

# Process engine data
engine_data = test_df[test_df['unit_id'] == engine_id].copy()
actual_rul = true_rul['RUL'].iloc[engine_id - 1]

# Apply Savitzky-Golay smoothing
for col in sensor_cols:
    if len(engine_data[col]) > 11:
        engine_data[col] = savgol_filter(engine_data[col], 11, 2)

# Scale
engine_data[sensor_cols] = scaler.transform(engine_data[sensor_cols])

# Create sequence
if len(engine_data) >= SEQUENCE_LENGTH:
    seq = engine_data[sensor_cols].values[-SEQUENCE_LENGTH:]
else:
    pad = np.zeros((SEQUENCE_LENGTH - len(engine_data), len(sensor_cols)))
    seq = np.vstack([pad, engine_data[sensor_cols].values])

input_seq = seq.reshape(1, SEQUENCE_LENGTH, len(sensor_cols))

# Predictions
pred_rul = max(0, min(MAX_RUL, float(reg_model.predict(input_seq, verbose=0)[0][0])))
fail_prob = float(class_model.predict(input_seq, verbose=0)[0][0])
health_score = (pred_rul / MAX_RUL) * 100

# Determine status
if health_score >= 70:
    status, color, emoji = "HEALTHY", "green", "🟢"
elif health_score >= 30:
    status, color, emoji = "CAUTION", "orange", "🟡"
else:
    status, color, emoji = "CRITICAL", "red", "🔴"

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Predicted RUL", f"{pred_rul:.0f} cycles", f"Actual: {actual_rul}")
with col2:
    st.metric("🎯 Actual RUL", f"{actual_rul} cycles")
with col3:
    st.metric("💚 Health Score", f"{health_score:.0f}%")
with col4:
    risk_text = "⚠️ High Risk" if fail_prob > 0.5 else "✅ Low Risk"
    st.metric("📈 Failure Risk (30d)", f"{fail_prob*100:.0f}%", risk_text)

# Status Box
st.markdown(f"""
<div style="padding: 25px; border-radius: 15px; background-color: {color}20; text-align: center; margin: 20px 0">
    <h1 style="color: {color}; margin: 0">{emoji} Engine Status: {status}</h1>
    <p style="color: {color}; margin-top: 10px">Health Score: {health_score:.1f}% | Predicted RUL: {pred_rul:.0f} cycles</p>
</div>
""", unsafe_allow_html=True)

# Maintenance Recommendations
st.subheader("📋 Maintenance Recommendations")
if health_score >= 70:
    st.success(f"✅ **Normal Operation** - Next inspection in 30 cycles. Continue monitoring.")
elif health_score >= 40:
    st.warning(f"⚠️ **Schedule Maintenance** - Plan service within {int(pred_rul)} cycles.")
elif health_score >= 15:
    st.error(f"🔧 **Urgent Maintenance** - Schedule service within {int(pred_rul)} cycles.")
else:
    st.error(f"🚨 **CRITICAL** - Immediate maintenance required within {int(pred_rul)} cycles!")

# Health Timeline Visualization
st.subheader("📈 Health Degradation Timeline")

# Generate predictions for the engine's lifecycle
all_predictions = []
for i in range(SEQUENCE_LENGTH, len(engine_data) + 1):
    window = engine_data[sensor_cols].values[i-SEQUENCE_LENGTH:i]
    if len(window) == SEQUENCE_LENGTH:
        pred = float(reg_model.predict(window.reshape(1, SEQUENCE_LENGTH, len(sensor_cols)), verbose=0)[0][0])
        all_predictions.append(max(0, min(MAX_RUL, pred)))

if all_predictions:
    cycles = engine_data['cycles'].values[SEQUENCE_LENGTH-1:len(all_predictions)+SEQUENCE_LENGTH-1]
    health_scores = [p/MAX_RUL*100 for p in all_predictions]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Remaining Useful Life (RUL) Over Time", "Health Score Trend"),
        vertical_spacing=0.12
    )
    
    # RUL plot
    fig.add_trace(
        go.Scatter(x=cycles, y=all_predictions, mode='lines', name='Predicted RUL',
                  line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )
    fig.add_hline(y=30, line_dash="dash", line_color="orange", 
                 annotation_text="Caution Zone", row=1, col=1)
    fig.add_hline(y=MAX_RUL, line_dash="dot", line_color="green", 
                 annotation_text="Max RUL", row=1, col=1)
    
    # Health score plot with colored zones
    fig.add_trace(
        go.Scatter(x=cycles, y=health_scores, mode='lines', name='Health Score',
                  line=dict(color='#2ca02c', width=2), fill='tozeroy'),
        row=2, col=1
    )
    
    # Add colored zones
    fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="red", opacity=0.2, row=2, col=1)
    fig.add_hrect(y0=30, y1=70, line_width=0, fillcolor="orange", opacity=0.2, row=2, col=1)
    fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="green", opacity=0.2, row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Time Cycles", row=2, col=1)
    fig.update_yaxes(title_text="RUL (cycles)", row=1, col=1)
    fig.update_yaxes(title_text="Health Score (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# Sensor Analysis Section
with st.expander("🔍 View Individual Sensor Trends"):
    sensor_to_plot = st.selectbox("Select Sensor to Analyze", sensor_cols)
    fig_sensor = go.Figure()
    fig_sensor.add_trace(go.Scatter(
        x=engine_data['cycles'], 
        y=engine_data[sensor_to_plot],
        mode='lines',
        name=sensor_to_plot,
        line=dict(color='#1f77b4', width=2)
    ))
    fig_sensor.update_layout(
        title=f"{sensor_to_plot} over Time",
        xaxis_title="Time Cycle",
        yaxis_title="Normalized Value",
        height=400
    )
    st.plotly_chart(fig_sensor, use_container_width=True)

# Model Performance Section
with st.expander("📊 Model Performance Details"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Root Mean Square Error (RMSE)", "14.25 cycles", delta="Better than LSTM baseline")
        st.metric("Mean Absolute Error (MAE)", "~11.7 cycles")
    with col2:
        st.metric("Binary Classification Accuracy", "95.00%")
        st.metric("Precision (Failure Detection)", "92%")
        st.metric("Recall (Failure Detection)", "88%")
    st.caption("**Note**: Our CNN-LSTM model outperforms the standard LSTM baseline (14.93 RMSE)")
    
    st.markdown("---")
    st.subheader("Model Architecture")
    st.code("""
    CNN-LSTM Hybrid Network:
    ├── Conv1D(64, kernel=3) + ReLU
    ├── Conv1D(32, kernel=3) + ReLU  
    ├── MaxPooling1D(pool=2)
    ├── LSTM(64, return_sequences=True)
    ├── Dropout(0.2)
    ├── LSTM(32, return_sequences=False)
    ├── Dropout(0.2)
    ├── Dense(32, activation='relu')
    └── Dense(1, activation='linear')
    """, language="text")
