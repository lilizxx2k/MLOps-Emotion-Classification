import torch
from mlops.model import Model

def test_model_output_shape():
    model = Model(output_dim=8)
    model.eval()

    # use 94x94 images to match current model's linear layer
    x = torch.randn(4, 3, 94, 94)
    y = model(x)

    assert y.shape == (4, 8), f"Expected (4, 8), got {y.shape}"

def test_model_forward_training():
    model = Model(output_dim=8)
    model.train()  #enable dropout
    x = torch.randn(2, 3, 94, 94)
    y = model(x)
    assert y.shape == (2, 8), f"Expected (2,8), got {y.shape}"

def test_model_gradients():
    model = Model(output_dim=8)
    model.train()
    x = torch.randn(2, 3, 94, 94, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None, "Parameter gradients should not be None"


