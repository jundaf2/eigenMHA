# this script is for further understanding the workspace size returned by cudnnGetMultiHeadAttnBuffers()
import torch
import numpy as np

class Memory_Net:
    def __init__(self):
        self.p00 = torch.tensor(np.array(0), dtype=torch.float32, requires_grad=True) 
        self.p01 = torch.tensor(np.array(0), dtype=torch.float32, requires_grad=True)  
        self.p10 = torch.tensor(np.array(0), dtype=torch.float32, requires_grad=True)  
        self.p11 = torch.tensor(np.array(0), dtype=torch.float32, requires_grad=True)  
        
        self.__parameters = dict(p00=self.p00, p01=self.p01, p10=self.p10, p11=self.p11)

    def forward(self, batch_size, head_num): # p00 + p01*batch_size + p10*head_num + p11*batch_size*head_num
        return self.p00 + batch_size*self.p01 + head_num*self.p10 + batch_size*head_num*self.p11
    
    def parameters(self):
        for name, param in self.__parameters.items():
            yield param
            
    def cuda(self):
        pass
    
    def cpu(self):
        pass

# Data
x, y = torch.meshgrid(torch.tensor([1,2,3,4,5,6,7,8]),torch.tensor([1,2,3,4,5,6,7,8]))

base = torch.tensor([[24,48,72,96,120,144,168,192],
    [48,96,144,192,240,288,336,384],
    [72,144,216,288,360,432,504,576],
    [96,192,288,384,480,576,672,768],
    [120,240,360,480,600,720,840,960],
    [144,288,432,576,720,864,1008,1152],
    [168,336,504,672,840,1008,1176,1344],
    [192,384,576,768,960,1152,1344,1536]]) / 4

data = torch.tensor([[544,608,672,736,832,896,960,1024],
    [608,736,896,1024,1376,1504,1664,1792],
    [672,896,1312,1504,1728,2144,2368,2560],
    [736,1024,1504,1792,2272,2560,3040,3328],
    [832,1376,1728,2272,2848,3200,3744,4096],
    [896,1504,2144,2560,3200,3808,4448,4864],
    [960,1664,2368,3040,3744,4448,5152,5632],
    [1024,1792,2560,3328,4096,4864,5632,6400]]) / 4

data1 = data - base

# Net
mem_net = Memory_Net()

optimizer = torch.optim.Adam(mem_net.parameters(), lr=0.1, weight_decay=0.000)

mse_loss = torch.nn.MSELoss(reduction='sum')

for i in range(len(x)):
  for j in range(len(y)):
    out = mem_net.forward(x[i][j],y[i][j])
    optimizer.zero_grad()
    loss = mse_loss(base[i][j], out)
    loss.backward()
    optimizer.step()
    loss_numpy = loss.detach().numpy()
    print(x[i][j],y[i][j], loss_numpy, mem_net.p00, mem_net.p01, mem_net.p10, mem_net.p11)

