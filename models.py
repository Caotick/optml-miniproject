import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim=8, out_dim=2, nb_hidden=3, hidden_dim=30):
        super().__init__()
        self.nb_hidden = nb_hidden
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_hids = [nn.Linear(nb_hidden, nb_hidden) for i in range(nb_hidden)]
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # Model input
        x = F.relu(self.fc_in(x))
        # Hidden layers
        for fc_hid in self.fc_hids:
          x = F.relu(fc_hid(x))
        # Model output
        x = self.fc_out(x)

        return x

## From https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out