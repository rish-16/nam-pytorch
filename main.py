import torch 
from nam_pytorch import NAM

nam = NAM(784)
x = torch.rand(32, 784)

y = nam(x)

print (y.shape)