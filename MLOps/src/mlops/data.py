# data.py
import kaggle
from loguru import logger
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
    def __init__(self, images_dir: str, labels_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
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

        # Label file has same name as image
        label_path = self.labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {img_path.name}")

        # Read YOLO label (class_id x y w h)
        with open(label_path, "r") as f:
            line = f.readline().strip()
            class_id = int(line.split()[0])

        if self.transform:
            img = self.transform(img)

        return img, class_id  # Return both image and label


def download_affectnet(base_path: Path):
    """
    Download AffectNet YOLO-format dataset using the Kaggle Python API.
    """
    dataset_url = "fatihkgg/affectnet-yolo-format"
    zip_path = base_path / "affectnet-yolo-format.zip"

    # Check if images already exist
    if not (base_path / "train/images").exists():
        if not zip_path.exists():
            logger.info(f"Downloading {dataset_url} via Kaggle API...")
            
            # Using the official API instead of subprocess
            kaggle.api.dataset_download_files(
                dataset_url, 
                path=str(base_path), 
                unzip=True  # This handles the extraction for you!
            )
            
            logger.success("Download and extraction complete.")
        else:
            logger.info("Zip file found, skipping download.")
    else:
        logger.info("Dataset already exists.")

def resolve_dataset_root(base_path: Path) -> Path:
    """
    Handle nested YOLO_format/YOLO_format structure from Kaggle.
    Returns the actual dataset root containing train/valid/test.
    """
    if (base_path / "train").exists():
        return base_path

    nested = base_path / "YOLO_format"
    if (nested / "train").exists():
        print(f"Detected nested dataset directory: {nested}")
        return nested

    raise FileNotFoundError(
        f"Could not find train/valid/test folders in {base_path}"
    )


def get_dataloaders(base_path: str, batch_size: int = 32, img_size: int = 224):
    """
    Returns train, validation, and test dataloaders for AffectNet YOLO-format dataset.
    Automatically downloads and prepares dataset if missing.
    """
    base_path = Path(base_path)

    # Ensure dataset exists
    download_affectnet(base_path)
    base_path = resolve_dataset_root(base_path)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Define datasets
    train_dataset = AffectNetDataset(
        images_dir=base_path / "train/images",
        labels_dir=base_path / "train/labels",
        transform=transform
    )
    val_dataset = AffectNetDataset(
        images_dir=base_path / "valid/images",
        labels_dir=base_path / "valid/labels",
        transform=transform
    )
    test_dataset = AffectNetDataset(
        images_dir=base_path / "test/images",
        labels_dir=base_path / "test/labels",
        transform=transform
    )

    # Define dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Dataloaders initialized with batch_size={batch_size}")
    logger.success(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    BASE_PATH = os.path.join(os.path.expanduser("~"), "dtu/data/raw/affectnet/YOLO_format")

    train_loader, val_loader, test_loader = get_dataloaders(BASE_PATH)

    # Iterate over a batch to check
    for imgs, labels in train_loader:
        logger.debug(f"Batch Check - Images: {imgs.shape}, Labels: {labels}")
        break
