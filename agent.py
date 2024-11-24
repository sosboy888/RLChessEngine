import torch
import torch.nn as nn
import torch.optim as optim

class ZeroNet(nn.Module):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(ZeroNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4672),  # All possible chess moves
            nn.Softmax(dim=-1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        features = self.conv_layers(x)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value
