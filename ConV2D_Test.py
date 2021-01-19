from modules.Layer import Conv2D
import numpy as np


# Testing Forward Path for Conv2D layer

'''
import torch

 

input1 = torch.ones(1,1,6,6)
print(input1)

net = torch.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 2, padding = 1)
print("net = ",net)

print("Weight = ",net.weight)

print("bias = ",net.bias)

out = net(input1)

print("output = ",out)
    
'''


#net = torch.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 2, padding = 1)

input1 = np.ones((1,1,6,6))
print(input1)
net = Conv2D(1,1,1,2,3)

out = net(input1)

print(out)



