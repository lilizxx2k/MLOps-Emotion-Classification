import torch
import torch.nn as nn

class baseline(nn.Module):
    def __init__(self, input_dim, hid_dim_1 = 256, hid_dim_2= 64, output_dim = 8):
        super().__init__()
        self.lin_1 = nn.Linear(-1, hid_dim_1, hid_dim_2)
        self.relu = nn.ReLU()
        self.lin_2 = nn.Linear(hid_dim_2, output_dim)
        
    def forward(self, x):
        x = nn.Flatten(x)
        x = self.lin_1(x)
        x = self.relu(x)
        x = self.lin_2(x)
        return x # note that we do not use softmax or somethink likewise because this is calculated in the loss