import torch
import torch.nn as nn


class SepsisLSTM(nn.Module):
    def __init__(self, input_size=22, hidden_size=64,
                 num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)

        return self.classifier(context).squeeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)


if __name__ == '__main__':
    model = SepsisLSTM(input_size=22)
    dummy = torch.randn(32, 12, 22)
    out = model(dummy)
    print(f"Input:       {dummy.shape}")
    print(f"Output:      {out.shape}")
    print(f"Parameters:  {count_parameters(model):,}")
    print("Architecture OK")