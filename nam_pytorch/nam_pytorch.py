import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        out = self.l6(x)

        return out

class NAM(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.networks = nn.ModuleList([
            FeatureNetwork() for _ in range(num_features)
        ])
        self.num_features = num_features
        self.bias = nn.Parameter(torch.rand(1)) # extra beta term

    def forward(self, x):
        B, dim = x.shape
        outs = torch.Tensor(B, dim)
        
        for i in range(B):
            temp = torch.Tensor(dim)
            for j in range(self.num_features): # for all dim
                net = self.networks[j]
                xi = x[i, j].unsqueeze(dim=0)
                temp[j] = net(xi)
            outs[i] = temp

        summed = outs.sum(axis=1) + self.bias
        res = torch.sigmoid(summed).view(B, 1)

        return res