import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SatFormer_Block import device

class TransferModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(TransferModule, self).__init__()
        self.to(device)
        self.query = nn.Linear(in_features, out_features)
        self.key = nn.Linear(in_features, out_features)
        self.value = nn.Linear(in_features, out_features)
        self.scale = torch.sqrt(torch.FloatTensor([out_features]))
        self.output_linear = nn.Linear(out_features, out_features)

        nn.init.xavier_uniform_(self.query.weight.data)
        nn.init.xavier_uniform_(self.key.weight.data)
        nn.init.xavier_uniform_(self.value.weight.data)
        nn.init.xavier_uniform_(self.output_linear.weight.data)
        nn.init.zeros_(self.output_linear.bias.data)

    def forward(self, x):
        self.to(device)
        x=torch.transpose(x,0,1)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention = F.softmax(attention_scores, dim=-1)
        x = torch.matmul(attention, V)
        x = self.output_linear(x)
        
        return x
    
class MLPModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLPModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight.data)
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, x):
        x=torch.transpose(x,0,1)
        return self.linear(x)
    