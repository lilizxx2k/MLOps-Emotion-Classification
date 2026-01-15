import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()

        # convolutional backbone 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # fixed feature map size with adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

        # classifier head 
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Force fixed spatial dimensions
        x = self.adaptive_pool(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = Model(output_dim=8)  #8 emotional classes
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
