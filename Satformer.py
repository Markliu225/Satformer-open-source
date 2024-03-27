import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from GCN import Norm_SpatiotemporalGraphConvolution
from SatFormer_Block import SatFormer_Block, process_local_regions
from ResNet import ResNetBlock
from Decode import n,t
from Decode import DecodeModule
from Transfer import TransferModule
import time
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

from SatFormer_Block import device

class SatFormer(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes, num_time_steps, transfer_size, region_size):
        super(SatFormer, self).__init__()
        self.to(device)
        self.st_gcn = Norm_SpatiotemporalGraphConvolution(transfer_size, hidden_size)
        self.attention = SatFormer_Block(region_size, heads)
        self.transfer = TransferModule(hidden_size, transfer_size)
        self.decode = DecodeModule(transfer_size, hidden_size, num_nodes, num_time_steps)
        self.resnet = ResNetBlock(transfer_size, hidden_size)
        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps

    def forward(self, x, adjacency_matrices):
        outputs = []
        self.to(device)
        for time_step in range(self.num_time_steps):
            adjacency_matrix = adjacency_matrices[:, :, time_step]
            x_slice = x[:, :, time_step]
            x_resnet = self.resnet(x_slice.unsqueeze(-1)).squeeze(-2)
            x_gcn = self.st_gcn(x_slice, adjacency_matrix)
            x_attention = process_local_regions(x_gcn, self.attention, heads)
            x_resnet = torch.reshape(x_resnet, (x_resnet.shape[0], -1))
            x_resnet = torch.mean(x_resnet, dim=1)
            x_attention = torch.mean(x_attention, dim=(1))
            x_res_att = x_attention + x_resnet
            outputs.append(x_res_att.unsqueeze(-1))

        encoded_features = torch.cat(outputs, dim=-1)
        transferred_features = self.transfer(encoded_features)
        decoded_features = self.decode(transferred_features, adjacency_matrices)

        return decoded_features
tensor_list = torch.empty(n, n, t)

for i in range(1, t+1):
    file_path = f'D:/Myarticle/IMC/dataset/iri_inter_1000/inter_1000/{i}.xlsx'#iri
    #file_path = f'D:/STK_Files/traffic_generator/traffic_generator/telesat/inter_1000/Useful/{i}.xlsx'#tele
    #file_path = f'D:/BaiduNetdiskDownload/inter_starlink/'+f'{i}.xlsx'#starlink
    df = pd.read_excel(file_path, header=None)
    tensor_list[:,:,i-1]=torch.tensor(df.values)
    print("Reading Done: ",i)

def custom_normalization(x, xmax):
    return torch.tensor((np.log(x + 1) / np.log(xmax.item())), dtype=torch.float32)

xmax = torch.max(tensor_list)
data_tensor_normalized = custom_normalization(tensor_list, xmax)
print("Normalization finished")

data_tensor_normalized = data_tensor_normalized.to(device)

origin_tensor = data_tensor_normalized.to(device)
input_tensor = origin_tensor.clone().to(device)
test_tensor = origin_tensor.clone().to(device)

torch.manual_seed(3047)
np.random.seed(3047)
mask_percentage = 0.98   
num_elements_to_mask = int(input_tensor.numel() * mask_percentage)
mask_indices = torch.randperm(input_tensor.numel())[:num_elements_to_mask]
input_tensor.view(-1)[mask_indices] = 0
adjacency_matrices = torch.zeros(n, n, t)
for i in range(t):
    data_at_time = input_tensor[:, :, i]
    adj_matrix = (data_at_time > 0).float()
    adjacency_matrices[:, :, i] = adj_matrix
adjacency_matrices = adjacency_matrices.to(device)


num_elements_to_mask = int(test_tensor.numel() * mask_percentage)
mask_indices = torch.randperm(test_tensor.numel())[:num_elements_to_mask]
test_tensor.view(-1)[mask_indices] = 0
test_adjacency_matrices = torch.zeros(n, n, t)
for i in range(t):
    data_at_time = test_tensor[:, :, i]
    adj_matrix = (data_at_time > 0).float()
    test_adjacency_matrices[:, :, i] = adj_matrix
test_adjacency_matrices = test_adjacency_matrices.to(device)


heads = 16
region_size = 16
input_size = t
hidden_size = 128
num_nodes = n
num_time_steps = t
transfer_size = n

input_tensor = input_tensor.float()
adjacency_matrices = adjacency_matrices.float()

stgcn_extractor = SatFormer(input_size, hidden_size, num_nodes, num_time_steps, transfer_size, region_size).to(device)
lr=0.001
criterion = nn.MSELoss()
criterion = criterion.to(device)
optimizer = optim.Adam(stgcn_extractor.parameters(), lr)

num_epochs = 2000
input_tensor_cpu = input_tensor.to('cpu').numpy()
origin_tensor_cpu = origin_tensor.to('cpu').numpy()
for epoch in range(num_epochs):
    start_time = time.time()

    optimizer.zero_grad()
    output = stgcn_extractor(input_tensor, adjacency_matrices).to(device)
    loss = criterion(output, origin_tensor).to(device)
    loss.backward()
    optimizer.step()
    end_time = time.time()
    output_cpu = output.detach().to('cpu').numpy()
    
    mae = mean_absolute_error(origin_tensor_cpu.reshape(-1), output_cpu.reshape(-1))
    rmse = calculate_rmse(origin_tensor_cpu, output_cpu)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, MAE: {mae:.4f}, RMSE:{rmse:.4f}')
    
    Epoch_time = end_time - start_time
    print("time: ", Epoch_time)
stgcn_extractor.eval()
with torch.no_grad():
    test_output = stgcn_extractor(test_tensor, test_adjacency_matrices).to(device)
    test_output_cpu = test_output.to('cpu').numpy()
    origin_tensor_cpu = origin_tensor.to('cpu').numpy()
    input_tensor_cpu = input_tensor.to('cpu').numpy()
    
    test_loss = criterion(test_output, origin_tensor)

test_mae = mean_absolute_error(origin_tensor_cpu.reshape(-1), test_output_cpu.reshape(-1))
test_rmse = calculate_rmse(origin_tensor_cpu, test_output_cpu)

print(f'Test Loss: {test_loss.item():.4f}, Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}')
