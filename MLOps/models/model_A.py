import torch
import torch.nn as nn

class model_A(torch.nn.Module): # a model using convelutional layers, max pooling, linear net and drop out
    def __init__(self, input_dim, hid_dim_1 = 256, hid_dim_2= 64, output_dim = 8):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3)
        self.lin_1 = nn.Linear(128 * 20 * 20, 64)
        self.lin_2 = nn.Linear(64, 8)

        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        # first we do the convelution layer and max pooling
        x = nn.F.relu(self.conv_1(x))
        x = self.pool_1(x)

        x = nn.F.relu(self.conv_2(x))
        x = self.pool_2(x)

        x = nn.F.relu(self.conv_3(x))

        # Then we flatten the output so we can use a simple linear neraul net on it.
        # Note we use dropout here
        x = torch.flatten(x, 1)
        x = nn.F.relu(self.lin_1(x))
        x = self.dropout(x)
        x = self.lin_2(x)

        return x # note we do not use softmax since this will be taken care of in the loss function.