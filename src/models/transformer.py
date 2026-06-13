import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model by taking care of the last element
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.2):
        super(TransformerEncoderModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.feature_dim = d_model

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_linear(x) # (batch_size, seq_len, d_model)
        
        # Transformer expects (seq_len, batch_size, d_model) if batch_first=False
        # But we used batch_first=True, so input is (batch_size, seq_len, d_model)
        
        # Need to transpose for positional encoding which expects (seq_len, batch_size, d_model)
        x_pe = x.transpose(0, 1)
        x_pe = self.pos_encoder(x_pe)
        x = x_pe.transpose(0, 1)
        
        output = self.transformer_encoder(x) # (batch_size, seq_len, d_model)
        
        # Aggregate across sequence length (e.g., mean pooling)
        pooled_output = torch.mean(output, dim=1) # (batch_size, d_model)
        return pooled_output
