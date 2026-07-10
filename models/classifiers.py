import torch
import torch.nn as nn

class DeepSeekLatentClassifier(nn.Module):
    """LSTM-based classifier for DeepSeek trajectories."""
    def __init__(self, hidden_dim=1536, num_layers=29, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, trajectory_tensor):
        x = self.layer_norm(trajectory_tensor)
        lstm_out, _ = self.lstm(x)
        lstm_pooled = lstm_out.mean(dim=1)
        logits = self.classifier(lstm_pooled)
        return logits.squeeze(1)

class QwenLatentClassifier(nn.Module):
    """Simpler MLP classifier with Global Average Pooling for Qwen."""
    def __init__(self, hidden_dim=2048, num_layers=37, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, trajectory_tensor):
        x = self.layer_norm(trajectory_tensor)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits.squeeze(1)
