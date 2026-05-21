import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Jet Engine Predictive Maintenance", layout="wide")

MAX_RUL = 125
SEQ_LEN = 50

st.title("Jet Engine Predictive Maintenance System")
st.caption("CNN-LSTM Model | RMSE: 14.25 | Accuracy: 95%")

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
    test_df = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=cols)
    true_rul = pd.read_csv('data/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
    drop_cols = ['os1', 'os2', 'os3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
    test_df.drop(columns=drop_cols, inplace=True)
    return test_df, true_rul

reg_model, class_model, scaler = load_assets()
test_df, true_rul = load_data()
sensor_cols = [c for c in test_df.columns if c not in ['unit_id', 'cycles']]

st.sidebar.header("Engine Control")
engine_id = st.sidebar.selectbox("Select Engine ID:", test_df['unit_id'].unique())

engine_data = test_df[test_df['unit_id'] == engine_id].copy()
actual_rul = true_rul['RUL'].iloc[engine_id - 1]

for col in sensor_cols:
    if len(engine_data[col]) > 11:
        engine_data[col] = savgol_filter(engine_data[col], 11, 2)

engine_data[sensor_cols] = scaler.transform(engine_data[sensor_cols])

if len(engine_data) >= SEQ_LEN:
    seq = engine_data[sensor_cols].values[-SEQ_LEN:]
else:
    pad = np.zeros((SEQ_LEN - len(engine_data), len(sensor_cols)))
    seq = np.vstack([pad, engine_data[sensor_cols].values])

input_seq = seq.reshape(1, SEQ_LEN, len(sensor_cols))
pred_rul = max(0, min(MAX_RUL, float(reg_model.predict(input_seq, verbose=0)[0][0])))
fail_prob = float(class_model.predict(input_seq, verbose=0)[0][0])
health = (pred_rul / MAX_RUL) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted RUL", f"{pred_rul:.0f} cycles", f"Actual: {actual_rul}")
col2.metric("Actual RUL", f"{actual_rul} cycles")
col3.metric("Health Score", f"{health:.0f}%")
col4.metric("Failure Risk", f"{fail_prob*100:.0f}%")

if health >= 70:
    status, color = "HEALTHY", "green"
elif health >= 30:
    status, color = "CAUTION", "orange"
else:
    status, color = "CRITICAL", "red"

st.markdown(f'<div style="padding:20px;border-radius:10px;background-color:{color}20;text-align:center"><h2 style="color:{color}">{status}</h2></div>', unsafe_allow_html=True)

if health >= 70:
    st.success("Normal operation. Next inspection in 30 cycles.")
elif health >= 40:
    st.warning(f"Plan maintenance within {int(pred_rul)} cycles.")
else:
    st.error(f"IMMEDIATE maintenance required within {int(pred_rul)} cycles!")

with st.expander("Model Performance"):
    st.metric("RMSE", "14.25 cycles")
    st.metric("Binary Accuracy", "95.00%")
    st.caption("CNN-LSTM outperforms standard LSTM baseline (14.93 RMSE)")
