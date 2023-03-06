import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# assume 1 head and 1 batch
batch_size = 1
n_heads = 1
seq_len = 1
head_size = 64
hidden_size = head_size*n_heads  

dropout_rate = 0.1

class BERTAttention(nn.Module):

    def __init__(self):
        super(BERTAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.q_w = nn.Linear(hidden_size, hidden_size)
        self.k_w = nn.Linear(hidden_size, hidden_size)
        self.v_w = nn.Linear(hidden_size, hidden_size)

    def forward(self, embeddings, mask):
        Q = self.q_w(embeddings).view(batch_size, seq_len, n_heads, head_size).permute(0, 2, 1, 3)
        K = self.k_w(embeddings).view(batch_size, seq_len, n_heads, head_size).permute(0, 2, 1, 3)
        V = self.v_w(embeddings).view(batch_size, seq_len, n_heads, head_size).permute(0, 2, 1, 3)
        S = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(head_size)) + mask
        print(S.requires_grad)
        P = F.softmax(S, dim=-1)
        print(P.requires_grad)
        P = self.dropout(P)
        O = torch.matmul(P, V).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        return O

# Input Data
embeddings = torch.ones([batch_size,seq_len,hidden_size],dtype=torch.float32, requires_grad=False)
mask = torch.zeros([batch_size,seq_len,seq_len],dtype=torch.float32, requires_grad=False)

# Model
BERT_Attention = BERTAttention( )
output = BERT_Attention(embeddings, mask)
# print("output:",output)
# print("output.shape:",output.shape)

# print("Initial Model Weights:")
# for name, param in BERT_Attention.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# Loss
criterion = nn.L1Loss(reduction='sum')
loss = criterion(output, torch.zeros(output.shape,dtype=torch.float32, requires_grad=False))
loss.backward()

print("Gradients of Model Weights:")
for name, param in BERT_Attention.named_parameters():
    if param.requires_grad:
        print(name, param.grad)
        
# BP
lr = 1
optimizer = torch.optim.SGD(BERT_Attention.parameters(), lr=lr)
optimizer.zero_grad()
optimizer.step()

# print("Updated Model Weights:")
# for name, param in BERT_Attention.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)


print([param.is_leaf for (name, param) in BERT_Attention.named_parameters()])
print([param.grad_fn for (name, param) in BERT_Attention.named_parameters()])