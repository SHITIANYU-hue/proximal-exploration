import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data

from torch.nn.parameter import Parameter
from torch.nn import init

class Seq32x1_16(nn.Module):
    def __init__(self, output_dim=1):
        super(Seq32x1_16,self).__init__()
        #self.length = length
        #in, out, kernel, stride, padding
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(20, 32, 5, 1, 2)
        self.pool = nn.MaxPool2d((1,2), stride=(1,2))
        self.flatten = nn.Flatten()
        
        #40 = 80/2 * math.ceil(math.ceil(self.length/2)/2)
        self.fc1 = nn.Linear(640,16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.flatten(x)
        #x = x.view(-1, 64 * math.ceil(math.ceil(20/2)/2))
        x = self.dropout(F.relu(self.fc1(x)))
        return x

class Seq32x2_16(nn.Module):
    def __init__(self, output_dim=1):
        super(Seq32x2_16, self).__init__()
        #self.length = length
        #in, out, kernel, stride, padding
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(20, 32, 5, 1, 2)
        self.pool = nn.MaxPool2d((1,2), stride=(1,2))
        self.conv2 = nn.Conv1d(32, 64, 5, 1, 2)
        self.flatten = nn.Flatten()
        
        #40 = 80/2 * math.ceil(math.ceil(self.length/2)/2)
        self.fc1 = nn.Linear(640,16)
        self.fc2 = nn.Linear(16, self.output_dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        #x = x.view(-1, 64 * math.ceil(math.ceil(20/2)/2))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class Seq64x1_16(nn.Module):
    def __init__(self, output_dim=1):
        super(Seq64x1_16,self).__init__()
        #self.length = length
        #in, out, kernel, stride, padding
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(20, 64, 5, 1)
        self.pool = nn.MaxPool2d((1,2), stride=(1,2))
        self.flatten = nn.Flatten()
        
        #40 = 80/2 * math.ceil(math.ceil(self.length/2)/2)
        self.fc1 = nn.Linear(1152,16)
        self.fc2 = nn.Linear(16, self.output_dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.flatten(x)

        #x = x.view(-1, 64 * math.ceil(math.ceil(20/2)/2))
        x = self.dropout(F.relu(self.fc1(x)))
        return x

class Seq_emb_32x1_16(nn.Module):
    def __init__(self, output_dim=1):
        super(Seq_emb_32x1_16, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(20, 8, 1, 1, 1)
        self.pool = nn.MaxPool2d((1,2), stride=(1,2))
        self.conv2 = nn.Conv1d(8, 64, 5, 1, 2)
        #self.pool = nn.MaxPool2d((1,2), stride=(1,2))
        self.flatten = nn.Flatten()
        
        #40 = 80/2 * math.ceil(math.ceil(self.length/2)/2)
        self.fc1 = nn.Linear(640,16)
        self.fc2 = nn.Linear(16, self.output_dim)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        #x = x.view(-1, 64 * math.ceil(math.ceil(20/2)/2))
        x = self.dropout(F.relu(self.fc1(x)))
        return x

class Seq32x1_16_filt3(nn.Module):
    def __init__(self, output_dim=1):
        super(Seq32x1_16_filt3, self).__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(20, 32, 3, 1, 2)
        self.pool = nn.MaxPool2d((1,2), stride=(1,2))
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(1344, 16)
        self.fc2 = nn.Linear(16, self.output_dim)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.flatten(x)
        #x = x.view(-1, 64 * math.ceil(math.ceil(20/2)/2))
        x = self.dropout(F.relu(self.fc1(x)))
        return x
    

class WeightClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-3,3)
            module.weight.data = w