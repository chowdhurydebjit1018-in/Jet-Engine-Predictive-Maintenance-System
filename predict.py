import argparse
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

def predict_engine(engine_id):
    print(f"Predicting for Engine #{engine_id}")
    reg_model = load_model('models/rul_regressor_v2.h5', compile=False)
    print("Model loaded successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_id', type=int, required=True)
    args = parser.parse_args()
    predict_engine(args.engine_id)
