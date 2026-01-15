import os
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloaders
from model import Model
from labels import NUM_CLASSES


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

    acc = correct / total
    return total_loss / len(dataloader), acc


def main():
    torch.manual_seed(42)

    BASE_PATH = os.path.join(
        os.path.dirname(__file__),
        "data/raw/affectnet/YOLO_format"
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 1e-3

    train_loader, val_loader, test_loader = get_dataloaders(
        BASE_PATH,
        batch_size=BATCH_SIZE,
        img_size=224
    )

    model = Model(output_dim=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

    # Final test accuracy
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\nTest Accuracy: {test_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/TrainedModel.pth")


if __name__ == "__main__":
    main()

