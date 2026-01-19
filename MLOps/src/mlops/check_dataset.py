"""
M10 Docker sanity check for AffectNet dataset.
Uses existing data.py exactly as-is.
"""

from mlops.data import AffectNetDataset
from pathlib import Path

images_dir = Path("/app/data/raw/affectnet/YOLO_format/train/images")

dataset = AffectNetDataset(images_dir)

print("AffectNetDataset initialized successfully")
print("Images directory:", images_dir)
print("Number of images detected:", len(dataset))
print("Dataset sanity check completed ")
