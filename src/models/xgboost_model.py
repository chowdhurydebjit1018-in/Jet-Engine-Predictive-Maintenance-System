import xgboost as xgb

def build_xgboost_model(params=None):
    """
    Builds an XGBoost regressor for predicting RUL based on tabular engineered features.
    """
    if params is None:
        params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'reg:squarederror'
        }
    
    model = xgb.XGBRegressor(**params)
    return model
