from torch.utils.data import Dataset

from mlops.data import AffectNetDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = AffectNetDataset("data/raw")
    assert isinstance(dataset, Dataset)
