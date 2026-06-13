import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    """
    Wraps a base feature extraction model to output multiple predictions:
    1. RUL (Regression)
    2. Failure Probability (Binary Classification)
    3. Health Score (Regression 0-100)
    4. Risk Category (Classification 0: Healthy, 1: Warning, 2: Critical)
    """
    def __init__(self, base_model):
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        feature_dim = base_model.feature_dim
        
        # Task 1: RUL Prediction
        self.rul_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Task 2: Failure Probability (Binary)
        self.failure_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Task 3: Health Score (Regression)
        self.health_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Will be scaled to 0-100 later
        )
        
        # Task 4: Risk Category (3 classes)
        self.risk_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        features = self.base_model(x)
        
        rul_pred = self.rul_head(features)
        fail_prob = self.failure_head(features)
        health_pred = self.health_head(features) * 100.0
        risk_logits = self.risk_head(features)
        
        return {
            'rul': rul_pred,
            'failure_prob': fail_prob,
            'health_score': health_pred,
            'risk_logits': risk_logits
        }
