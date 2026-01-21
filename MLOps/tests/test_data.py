from torch.utils.data import Dataset

from mlops.data import AffectNetDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = AffectNetDataset(images_dir="data/raw", labels_dir="data/raw")
    assert isinstance(dataset, Dataset)
