import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN import Norm_SpatiotemporalGraphDeConvolution
from SatFormer_Block import SatFormer_Block, DeSatFormer_Block
from ResNet import ResNetBlock, DeResNetBlock
from SatFormer_Block import device
n=66
t=10
class DecodeModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes, num_time_steps):
        super(DecodeModule, self).__init__()
        self.to(device)
        self.st_gcn1 = Norm_SpatiotemporalGraphDeConvolution(hidden_size, input_size)
        self.attention1 = DeSatFormer_Block(input_size, input_size)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.resnet = DeResNetBlock(input_size ,input_size)
        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps

    def forward(self, x, adjacency_matrices):
        self.to(device)
        output = torch.randn((n, n, t))
        
        for time_step in range(self.num_time_steps):
            x_resnet = self.resnet(x.unsqueeze(-1)).squeeze(-1)
            x_gcn1 = self.st_gcn1(x, adjacency_matrices[:, :, time_step])
            x_attention1 = self.attention1(x_gcn1)
            x_resnet = torch.matmul(x_resnet, x_resnet.t())
            x_res_att = x_attention1 + x_resnet
            output[:,:,time_step]=self.mlp(x_res_att)
        return output