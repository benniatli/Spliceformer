# +
import torch
import torch.nn as nn
import numpy as np

def keras_init(m):
    if isinstance(m, nn.Conv1d): 
        fin, fout = nn.init._calculate_fan_in_and_fan_out(m.weight)
        a = np.sqrt(6/(m.in_channels*(fin+fout)))
        torch.nn.init.uniform_(m.weight, a=-a, b=a)
        #print(m,nn.init._calculate_fan_in_and_fan_out(m.weight))
        #nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        #nn.init.kaiming_uniform_(m.weight,mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)
        #if m.bias is not None:
        #    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        #    bound = 1 / np.sqrt(fan_in)
        #    nn.init.uniform_(m.bias, -bound, bound)
