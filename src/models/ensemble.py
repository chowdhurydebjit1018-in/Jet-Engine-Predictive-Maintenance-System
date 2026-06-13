import numpy as np

class WeightedEnsemble:
    def __init__(self, weights=None):
        """
        weights: list or array of weights for each model.
        If None, equal weighting is used.
        """
        self.weights = weights
        
    def predict(self, predictions_list):
        """
        predictions_list: List of numpy arrays, each array is the prediction from one model.
        Returns the weighted average.
        """
        preds = np.array(predictions_list) # Shape: (num_models, num_samples)
        
        if self.weights is None:
            self.weights = np.ones(len(predictions_list)) / len(predictions_list)
            
        weights = np.array(self.weights).reshape(-1, 1)
        weighted_preds = np.sum(preds * weights, axis=0)
        return weighted_preds

def optimize_ensemble_weights(val_predictions, val_targets):
    """
    Simple grid search to find best weights on validation set for RUL.
    """
    from itertools import product
    from sklearn.metrics import mean_squared_error
    
    num_models = len(val_predictions)
    best_weights = None
    best_rmse = float('inf')
    
    # Try combinations of weights (e.g., 0.1, 0.2, ... 0.9)
    # To keep it simple and fast, we use random search
    for _ in range(100):
        w = np.random.dirichlet(np.ones(num_models), size=1)[0]
        ensemble = WeightedEnsemble(weights=w)
        preds = ensemble.predict(val_predictions)
        rmse = np.sqrt(mean_squared_error(val_targets, preds))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = w
            
    return best_weights, best_rmse
