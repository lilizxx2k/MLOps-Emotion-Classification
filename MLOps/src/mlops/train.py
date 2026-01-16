import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import wandb  # Added wandb import

from data import get_dataloaders
from model import Model
from labels import NUM_CLASSES
from loguru import logger
from dotenv import load_dotenv

# Load API keys from local .env files
load_dotenv()  

# Remove the default logger
logger.remove()

# Only WARNING and higher are shown on console
logger.add(sys.stderr, level="WARNING")

# Save DEBUG and higher with 100MB rotation
logger.add(
    "my_log.log", 
    level="DEBUG", 
    rotation="100 MB"
)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return total_loss / len(dataloader), acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Added logic to capture some images for wandb logging
    example_images = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Capture first batch of images for visual logging
            if not example_images:
                for i in range(min(5, imgs.size(0))):
                    example_images.append(wandb.Image(imgs[i], caption=f"Pred: {preds[i].item()}, GT: {labels[i].item()}"))

    acc = correct / total
    return total_loss / len(dataloader), acc, example_images

def main():
    # Using 'mps' 
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Initialize wandb
     wandb.init(project="affectnet-classification", job_type="training")

    BASE_PATH = os.path.join(
        os.path.dirname(__file__),
        "data/raw/affectnet/YOLO_format"
    )

    # Extract Sweep values 
    LR = wandb.config.get("learning_rate", 1e-3)
    BATCH_SIZE = wandb.config.get("batch_size", 32)
    OPT_TYPE = wandb.config.get("optimizer", "adam")
    EPOCHS = 10 

    # Data Loading 
    train_loader, val_loader, test_loader = get_dataloaders(
        BASE_PATH,
        batch_size=BATCH_SIZE,
        img_size=224
    )

    #  Model Setup
    model = Model(output_dim=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer Setup 
    if OPT_TYPE == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LR)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    logger.info(f"Starting training on {DEVICE} for {EPOCHS} epochs (LR: {LR}, Batch: {BATCH_SIZE})")

    # Training Loop
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc, val_images = evaluate(
            model, val_loader, criterion, DEVICE
        )

        # Log metrics and images to wandb (M14 requirement)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_examples": val_images
        })

        logger.success(
            f"Epoch [{epoch+1}/{EPOCHS}] complete. "
            f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        logger.info(f"Loss - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    # Final test accuracy
    test_loss, test_acc, _ = evaluate(
        model, test_loader, criterion, DEVICE
    )
    
    wandb.log({"test_acc": test_acc})
    logger.warning(f"FINAL TEST ACCURACY: {test_acc:.4f}")

    # Save and Log Artifact
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/TrainedModel.pth"
    torch.save(model.state_dict(), model_save_path)
    
    artifact = wandb.Artifact("affectnet-model", type="model")
    artifact.add_file(model_save_path)
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()