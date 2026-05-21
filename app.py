import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Jet Engine Predictive Maintenance", layout="wide")

st.title("✈️ Jet Engine Predictive Maintenance System")
st.caption("CNN-LSTM Model | RMSE: 14.25 | Accuracy: 95%")

# Cache loading to avoid reloading
@st.cache_resource
def load_models():
    import joblib
    import numpy as np
    from tensorflow.keras.models import load_model
    
    # Load with reduced memory
    reg_model = load_model('models/rul_regressor_v2.h5', compile=False)
    class_model = load_model('models/rul_classifier_v2.h5', compile=False)
    scaler = joblib.load('models/scaler_v2.pkl')
    return reg_model, class_model, scaler

@st.cache_data
def load_data():
    cols = ['unit_id', 'cycles', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    test_df = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=cols)
    true_rul = pd.read_csv('data/RUL_FD001.txt', sep=r'\s+', header=None, names=['RUL'])
    drop_cols = ['os1', 'os2', 'os3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
    test_df.drop(columns=drop_cols, inplace=True)
    return test_df, true_rul

# Load everything
reg_model, class_model, scaler = load_models()
test_df, true_rul = load_data()
sensor_cols = [c for c in test_df.columns if c not in ['unit_id', 'cycles']]

st.sidebar.header("Engine Control")
engine_id = st.sidebar.selectbox("Select Engine ID:", test_df['unit_id'].unique())

# Simple prediction without sequence processing
engine_data = test_df[test_df['unit_id'] == engine_id].copy()
actual_rul = true_rul['RUL'].iloc[engine_id - 1]

# Take last 50 cycles or pad
SEQ_LEN = 50
if len(engine_data) >= SEQ_LEN:
    seq = engine_data[sensor_cols].values[-SEQ_LEN:]
else:
    pad = np.zeros((SEQ_LEN - len(engine_data), len(sensor_cols)))
    seq = np.vstack([pad, engine_data[sensor_cols].values])

# Scale
seq_scaled = scaler.transform(seq)
input_seq = seq_scaled.reshape(1, SEQ_LEN, len(sensor_cols))

# Predict
pred_rul = max(0, min(125, float(reg_model.predict(input_seq, verbose=0)[0][0])))
fail_prob = float(class_model.predict(input_seq, verbose=0)[0][0])
health = (pred_rul / 125) * 100

# Display metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted RUL", f"{pred_rul:.0f} cycles", f"Actual: {actual_rul}")
c2.metric("Actual RUL", f"{actual_rul} cycles")
c3.metric("Health Score", f"{health:.0f}%")
c4.metric("Failure Risk", f"{fail_prob*100:.0f}%")

# Status
if health >= 70: status, color = "HEALTHY 🟢", "green"
elif health >= 30: status, color = "CAUTION 🟡", "orange"
else: status, color = "CRITICAL 🔴", "red"

st.markdown(f'<div style="padding:20px;border-radius:10px;background-color:{color}20;text-align:center"><h2 style="color:{color}">{status}</h2></div>', unsafe_allow_html=True)

# Recommendation
if health >= 70:
    st.success("✅ Normal operation. Next inspection in 30 cycles.")
elif health >= 40:
    st.warning(f"⚠️ Plan maintenance within {int(pred_rul)} cycles.")
else:
    st.error(f"🔧 IMMEDIATE maintenance required!")

# Model performance
with st.expander("📊 Model Performance"):
    st.metric("RMSE", "14.25 cycles")
    st.metric("Binary Accuracy", "95.00%")
    st.caption("CNN-LSTM outperforms standard LSTM baseline (14.93 RMSE)")
