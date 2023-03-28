import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data

from torch.nn.parameter import Parameter
from torch.nn import init

class Seq_32_32(nn.Module):
    def __init__(self, output_dim=1):
        super(Seq_32_32, self).__init__()
        self.output_dim = output_dim
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(800, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32,self.output_dim)
        self.dropout=nn.Dropout(0.5)
    
    def forward(self,x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x