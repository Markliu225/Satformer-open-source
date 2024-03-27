import torch
import torch.nn as nn
import torch.nn.functional as F
from SatFormer_Block import device

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.to(device)
        self.fc = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.fc.weight.data)
        nn.init.zeros_(self.fc.bias.data)

    def forward(self, x):
        self.to(device)
        output = self.fc(x)
        return F.relu(output)
    
    
class SpatiotemporalGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(SpatiotemporalGraphConvolution, self).__init__()
        self.to(device)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.zeros_(self.bias.data)

    def forward(self, x, adjacency_matrix):
        self.to(device)
        support = torch.matmul(x, self.weight)
        
        output = torch.matmul(adjacency_matrix, support) + self.bias
        return F.relu(output)

class SpatiotemporalGraphDeConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(SpatiotemporalGraphDeConvolution, self).__init__()
        self.to(device)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features,out_features))
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.zeros_(self.bias.data)

    def forward(self, x, adjacency_matrix):
        self.to(device)
        support = torch.matmul(x, self.weight)
        support=torch.transpose(support,0,1)
        output = torch.matmul(adjacency_matrix, support) 
        return F.relu(output)
    

class Norm_SpatiotemporalGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(Norm_SpatiotemporalGraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.zeros_(self.bias.data)

    def norm(self,adjacency_matrix):
        I = torch.eye(adjacency_matrix.size(0)).to(device)
        A_hat = adjacency_matrix + I 
        A_hat=A_hat.to(device)
        D_hat = torch.diag_embed(torch.pow(A_hat.sum(1), -0.5)) 
        A_norm = torch.matmul(torch.matmul(D_hat, A_hat), D_hat)
        return A_norm
    
    def forward(self, x, adjacency_matrix):
        A_norm = self.norm(adjacency_matrix)
        support = torch.matmul(x, self.weight)
        
        output = torch.matmul(A_norm, support) + self.bias
        return F.relu(output)

class Norm_SpatiotemporalGraphDeConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(Norm_SpatiotemporalGraphDeConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features,out_features))
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.zeros_(self.bias.data)
    def norm(self,adjacency_matrix):
        I = torch.eye(adjacency_matrix.size(0)).to(device)
        A_hat = adjacency_matrix + I  
        A_hat = A_hat.to(device)
        D_hat = torch.diag_embed(torch.pow(A_hat.sum(1), -0.5))  
        A_norm = torch.matmul(torch.matmul(D_hat, A_hat), D_hat) 
        return A_norm

    def forward(self, x, adjacency_matrix):
        A_norm = self.norm(adjacency_matrix)
        support = torch.matmul(x, self.weight)
        support=torch.transpose(support,0,1)
        output = torch.matmul(A_norm, support) 
        return F.relu(output)