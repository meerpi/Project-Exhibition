# early experiment with LSTM for sequence classification
# this is messy, just testing if temporal approach works at all

import numpy as np
import torch
import torch.nn as nn

class QuickLSTM(nn.Module):
    """throwaway prototype - will rewrite properly"""
    def __init__(self, input_dim=126, hidden_dim=64, num_classes=20):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out)

if __name__ == "__main__":
    # quick test with random data
    model = QuickLSTM()
    dummy = torch.randn(4, 30, 126)
    out = model(dummy)
    print(f"output shape: {out.shape}")  # should be (4, 20)
    print("looks like it works, need to add conv stem tho")
