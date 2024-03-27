import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
class MLP_Block(nn.Module):
    def __init__(self,hidden_size):
        super(MLP_Block, self).__init__()
        self.to(device)
        self.layer_norm = nn.LayerNorm(1584)
        self.mlp = nn.Sequential(
            nn.Linear(1584, 1584),
            nn.ReLU(),
            nn.Linear(1584, 1)
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        self.to(device)
        x=torch.transpose(x,0,1)
        x_normalized = self.layer_norm(x)
        output = self.mlp(x_normalized)
        return output.squeeze(-1) 
    
class SatFormer_Block(nn.Module):
    def __init__(self, hidden_size, heads):
        super(SatFormer_Block, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        assert (
            self.head_dim * heads == hidden_size
        ), "hidden_size must be divisible by heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.sparse_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.kaiming_uniform_(self.sparse_weights, a=math.sqrt(5))
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        self.scaling = self.head_dim ** -0.5
        nn.init.xavier_uniform_(self.query.weight.data)
        nn.init.zeros_(self.query.bias.data)
        nn.init.xavier_uniform_(self.key.weight.data)
        nn.init.zeros_(self.key.bias.data)
        nn.init.xavier_uniform_(self.value.weight.data)
        nn.init.zeros_(self.value.bias.data)
        nn.init.xavier_uniform_(self.output_linear.weight.data)
        nn.init.zeros_(self.output_linear.bias.data)

    def forward(self, x, heads):
        batch_size, seq_len, _ = x.unsqueeze(0).size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        sparse_mask = self.create_sparse_mask(seq_len, self.heads, device=x.device)
        attention_scores = attention_scores * sparse_mask #- 1e10 * (1 - sparse_mask)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(heads, seq_len, 16)
        y = F.adaptive_avg_pool2d(attention_output, (1, 128))
        vector_from_third_dimension = torch.squeeze(y)
        final_vector = vector_from_third_dimension.mean(dim=0)
        return final_vector

    def create_sparse_mask(self, seq_len, heads, device="cpu"):
        mask = torch.zeros((heads, seq_len, seq_len), device=device)
        for i in range(seq_len):
            for j in range(max(0, i-1), min(seq_len, i+2)):
                mask[:, i, j] = 1
        return mask


def process_local_regions(x, block, num_heads, region_size=(16, 16)):
    height, width = x.size()
    num_rows = height // region_size[0]
    num_cols = width // region_size[1]
    region_outputs = []
    for i in range(num_rows):
        for j in range(num_cols):
            region = x[ i * region_size[0]:(i + 1) * region_size[0], 
                      j * region_size[1]:(j + 1) * region_size[1]]
            attention_output = block(region, num_heads)
            region_outputs.append(attention_output)
    final_output = torch.stack(region_outputs, dim=1)

    return final_output

class DeSatFormer_Block(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DeSatFormer_Block, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(hidden_size, input_size))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x):
        x = torch.transpose(x,0,1)
        x_normalized = self.layer_norm(x)
        attention_scores = F.softmax(torch.matmul(x_normalized, self.weight.squeeze(-1)), dim=-1)
        attention_scores=torch.transpose(attention_scores,0,1)
        attended_features = torch.matmul(attention_scores, x)
        output = self.mlp(attended_features.squeeze(-2))
        return output

