import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim=8, out_dim=2, nb_hidden=3, hidden_dim=30):
        super().__init__()
        self.nb_hidden = nb_hidden
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_hid = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # Model input
        x = F.relu(self.fc_in(x))
        # Hidden layers
        for i in range(self.nb_hidden):
          x = F.relu(self.fc_hid(x))
        # Model output
        x = self.fc_out(x)

        return x