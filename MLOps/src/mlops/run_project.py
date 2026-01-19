"""
Minimal real project execution for Docker (M10)
- Uses real dataset class
- Uses real model
- Runs a real forward pass (correct shape)
"""
# Cleaned entry point (no module references)


import torch
from pathlib import Path

from mlops.data import AffectNetDataset
from mlops.model import Model

# Dataset path (can be empty, this is OK)
images_dir = Path("/app/data/raw/affectnet/YOLO_format/train/images")

# 1. Dataset (REAL project code)
dataset = AffectNetDataset(images_dir)
print(f"Dataset initialized. Number of samples: {len(dataset)}")

# 2. Model (REAL project code)
model = Model()
model.eval()

# 3. CORRECT dummy input for your model (Linear(1,1))
dummy_input = torch.rand(1, 1)  # batch_size=1, features=1

with torch.no_grad():
    output = model(dummy_input)

print("Model forward pass successful")
print("Output:", output)
print("Output shape:", output.shape)

print("Real project code executed successfully")
