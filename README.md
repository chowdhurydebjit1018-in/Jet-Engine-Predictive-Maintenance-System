# ✈️ Jet Engine Predictive Maintenance System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)

## 🎯 Project Overview

This project implements a **Predictive Maintenance System** for aircraft jet engines using the NASA CMAPSS dataset.

### 📊 Key Results
- **RMSE**: 14.25 cycles (beats LSTM baseline of 14.93)
- **Binary Accuracy**: 95% for 30-day failure prediction
- **Model**: CNN-LSTM Hybrid Network

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/Jet-Engine-Predictive-Maintenance.git
cd Jet-Engine-Predictive-Maintenance
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Performance Metrics

| Metric | Value | Benchmark |
|--------|-------|-----------|
| RMSE | 14.25 cycles | LSTM: 14.93 |
| Binary Accuracy | 95.00% | - |
| Precision (Failure) | 92% | - |
| Recall (Failure) | 88% | - |

## 🏗️ Model Architecture

```
CNN-LSTM Network:
├── Conv1D(64, kernel=3) + ReLU
├── Conv1D(32, kernel=3) + ReLU  
├── MaxPooling1D(pool=2)
├── LSTM(64, return_sequences=True)
├── Dropout(0.2)
├── LSTM(32, return_sequences=False)
├── Dropout(0.2)
├── Dense(32, activation=relu)
└── Dense(1, activation=linear)
```

## 📁 Project Structure

```
Jet-Engine-Predictive-Maintenance/
├── README.md              # Documentation
├── requirements.txt       # Dependencies
├── app.py                 # Streamlit dashboard
├── train.py              # Training script
├── predict.py            # Prediction script
├── models/               # Trained models
│   ├── rul_regressor_v2.h5
│   ├── rul_classifier_v2.h5
│   └── scaler_v2.pkl
├── data/                 # Dataset
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
└── utils/                # Utilities
    └── preprocess.py
```

## 🎮 Dashboard Features

- **Engine Selection**: Choose any engine (1-100)
- **Real-time Metrics**: RUL, health score, failure risk
- **Color-coded Status**: Green/Yellow/Red alerts
- **Maintenance Alerts**: Actionable recommendations

## 📈 Health Score Calculation

```
Health Score = (Predicted RUL / 125) × 100%

Status Zones:
- 🟢 70-100%: Healthy - Normal operation
- 🟡 30-70%: Caution - Schedule maintenance
- 🔴 0-30%: Critical - Immediate action required
```

## 🛠️ Technologies

- **TensorFlow 2.13** - Deep learning
- **Streamlit 1.28** - Interactive dashboard
- **Scikit-learn** - Preprocessing and metrics
- **Plotly** - Visualizations
- **NumPy/Pandas** - Data manipulation

## 📝 License

MIT License

## 👤 Author

Debjit Chowdhury

---
⭐ Star this repository if you find it useful!
