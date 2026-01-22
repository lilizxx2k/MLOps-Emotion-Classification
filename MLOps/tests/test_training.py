import torch
import torch.nn as nn
from model import Model
from train import train, evaluate


def test_single_training_step():
    model = Model(output_dim=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    x = torch.randn(2, 3, 94, 94)
    y = torch.randint(0, 8, (4,))

    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be positive"

def test_train_function_reduces_loss():
    model = Model(output_dim=8)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=le-3)
    # Fake dataloader: 2 batches of 2 images each
    batch_1 = (torch.randn(2, 3, 94, 94), torch.randint(0, 8, (2,)))
    batch_2 = (torch.randn(2, 3, 94, 94), torch.randint(0, 8, (2,)))
    fake_loader = [batch_1, batch_2]

    avg_loss, acc = train(model, fake_loader, optimizer, criterion, device="cpu")
    assert avg_loss > 0, "Average loss should be positive"
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"


def test_evaluate_function_returns_loss_and_accuracy():
    model = Model(output_dim=8)
    criterion = nn.CrossEntropyLoss()

    batch = (torch.randn(2, 3, 94, 94), torch.randint(0, 8, (2,)))
    fake_loader = [batch]

    avg_loss, acc = evaluate(model, fake_loader, criterion, device="cpu")
    assert avg_loss > 0, "Average loss should be positive"
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
