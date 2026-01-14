# data.py
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import zipfile
import subprocess

class AffectNetDataset(Dataset):
    """
    PyTorch Dataset for AffectNet YOLO-format dataset.
    
    Expects the following folder structure:
    YOLO_format/
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
    """
    def __init__(self, images_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Collect all image files (jpg + png)
        self.image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))

        if len(self.image_files) == 0:
            print(f"Warning: No images found in {self.images_dir}")
        else:
            print(f"Found {len(self.image_files)} images in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def download_affectnet(base_path: Path):
    """
    Download AffectNet YOLO-format dataset from Kaggle if it doesn't exist.
    Requires Kaggle API credentials (~/.kaggle/kaggle.json).
    """
    dataset_url = "fatihkgg/affectnet-yolo-format"  

    zip_path = base_path / "affectnet.zip"
    if not zip_path.exists():
        print("Downloading AffectNet dataset from Kaggle...")
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_url,
            "-p", str(base_path)
        ], check=True)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(base_path)

        print("Dataset downloaded and extracted.")
    else:
        print("AffectNet dataset already downloaded.")


def get_dataloaders(base_path: str, batch_size: int = 32, img_size: int = 224):
    """
    Returns train, validation, and test dataloaders for AffectNet YOLO-format dataset.
    
    Downloads dataset from Kaggle if missing.
    """
    base_path = Path(base_path)

    # Ensure dataset exists
    if not (base_path / "train/images").exists():
        download_affectnet(base_path)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Define datasets
    train_dataset = AffectNetDataset(base_path / "train/images", transform=transform)
    val_dataset   = AffectNetDataset(base_path / "valid/images", transform=transform)
    test_dataset  = AffectNetDataset(base_path / "test/images", transform=transform)

    # Define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    BASE_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/affectnet/YOLO_format")

    train_loader, val_loader, test_loader = get_dataloaders(BASE_PATH)

    # Iterate over a batch
    for imgs in train_loader:
        print(f"Batch shape: {imgs.shape}")
        break
