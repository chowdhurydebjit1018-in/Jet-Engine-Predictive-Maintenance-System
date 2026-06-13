import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(self.attention(lstm_output), dim=1) # (batch_size, seq_len, 1)
        context_vector = torch.sum(attn_weights * lstm_output, dim=1) # (batch_size, hidden_dim)
        return context_vector, attn_weights

class CNNLSTMAttention(nn.Module):
    def __init__(self, input_dim, cnn_filters=64, lstm_units=64, dropout=0.2):
        super(CNNLSTMAttention, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.conv2 = nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layers (Bidirectional)
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_units, 
                            batch_first=True, bidirectional=True)
        
        # Attention
        self.attention = Attention(lstm_units * 2) # *2 for bidirectional
        
        # Feature extraction output dimension
        self.feature_dim = lstm_units * 2
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Conv1d expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Back to (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        context_vector, attn_weights = self.attention(lstm_out)
        
        # context_vector is the extracted feature representation for downstream tasks
        return context_vector
