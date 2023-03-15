import torch as torch
import torch.nn as nn
import torch.nn.functional as F

# assume 1 head and 1 batch
batch_size = 2
n_heads = 2
seq_len = 2
head_size = 4
hidden_size = head_size*n_heads  

dropout_rate = 0

class MHA(nn.Module):

    def __init__(self):
        super(MHA, self).__init__()
        self.q_w = nn.Linear(hidden_size, hidden_size)
        self.k_w = nn.Linear(hidden_size, hidden_size)
        self.v_w = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax  = nn.Softmax(dim=-1)
        self.o_w = nn.Linear(hidden_size, hidden_size)  
        torch.manual_seed(2023)
    def forward(self, Q_in, K_in, V_in, mask):
        Q = self.q_w(Q_in).view(batch_size, seq_len, n_heads, head_size).permute(0, 2, 1, 3)
        K = self.k_w(K_in).view(batch_size, seq_len, n_heads, head_size).permute(0, 2, 1, 3)
        V = self.v_w(V_in).view(batch_size, seq_len, n_heads, head_size).permute(0, 2, 1, 3)
        S = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(head_size)) - (1.0 - mask.unsqueeze(1).unsqueeze(2).float()) * 10000.0
        P = self.softmax(S)
        P = self.dropout(P)
        O = torch.matmul(P, V).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, hidden_size)
        O_out = self.o_w(O).view(batch_size, seq_len, hidden_size)
        return O_out

# Input Data
Q = torch.randn([batch_size, seq_len, hidden_size],dtype=torch.float32, requires_grad=True)
K = torch.randn([batch_size, seq_len, hidden_size],dtype=torch.float32, requires_grad=True)
V = torch.randn([batch_size, seq_len, hidden_size],dtype=torch.float32, requires_grad=True)
mask = torch.zeros([batch_size, seq_len],dtype=torch.float32, requires_grad=False)

# Model
mha = MHA()
# init parameters
# for name, param in mha.named_parameters():
#     if param.requires_grad:
#         # print(name, param.data)
#         param.data.fill_(2)

torch.onnx.export(mha, (Q, K, V, mask), 'mha.onnx')
output = mha(Q, K, V, mask)

# print("Initial Model Weights:")
# for name, param in mha.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# Loss
criterion = nn.MSELoss() # nn.L1Loss(reduction='sum')
target = 0.5*torch.ones(output.shape,dtype=torch.float32, requires_grad=False)
loss = criterion(output, target)
loss.backward()

print("Gradients of Model Weights:")
for name, param in mha.named_parameters():
    if param.requires_grad:
        print(name, param.grad)
        
# BP
lr = 1
optimizer = torch.optim.SGD(mha.parameters(), lr=lr)
optimizer.zero_grad()
optimizer.step()

# print("Updated Model Weights:")
# for name, param in mha.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

print("param name:",[name for (name, param) in mha.named_parameters()])
print("param is_leaf:",[param.is_leaf for (name, param) in mha.named_parameters()])
print("param grad_fn:",[param.grad_fn for (name, param) in mha.named_parameters()])

