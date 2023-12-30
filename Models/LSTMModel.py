from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 4:
            x = x[:, -1, :, :]
        x = x.transpose(-2, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
