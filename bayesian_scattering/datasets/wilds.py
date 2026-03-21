import torch
from torch.utils.data import DataLoader, Subset
from wilds import get_dataset
import torchvision.transforms as transforms

from bayesian_scattering.datasets.abstract_dataset import AbstractDataset


class WILDS(AbstractDataset):
    def __init__(
            self,
            dataset_name,
            split,
            dataset_path=None,
            **kwargs
    ):
        assert dataset_name in ['iwildcam', 'poverty'], "WILDS dataset not available"
        if dataset_name == 'iwildcam':
            self.dataset = get_dataset(
                dataset=dataset_name,
                root_dir=dataset_path if dataset_path is not None else "./",
                download=True
            ).get_subset(
                split,
                transform=transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
            )
        elif dataset_name == 'poverty':
            self.dataset = get_dataset(
                dataset=dataset_name,
                root_dir=dataset_path if dataset_path is not None else "./",
                download=True
            ).get_subset(
                split,
            )
            # if 'channels' in kwargs:
            #     self.ch = torch.tensor(kwargs.get('channels'), dtype=torch.int)

        self.device = torch.device("cpu")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y, _ = self.dataset[idx]
        # if hasattr(self, 'ch'):
        #     x = torch.index_select(x, 0, self.ch)
        # return x.to(self.device), y.to(self.device)
        return x, y.sub(self.labels_norm[0]).div(self.labels_norm[1]) if hasattr(self, "labels_norm") else y

    def normalized_labels(self, idx=None):
        labels = None
        if hasattr(self.dataset, "y_array"):
            labels = self.dataset.y_array
            if idx is not None:
                labels = labels[idx]
            labels = labels.float()
        if labels is None:
            db = Subset(self, idx) if idx is not None else self
            _, labels = next(iter(DataLoader(db, len(db))))
        self.labels_norm = (labels.mean(), labels.std())

    def generate_sample(self, idx):
        NotImplementedError("This should not be called for WILDS class")

    @property
    def shape(self) -> torch.Size:
        return self.dataset[0][0].shape[-2:]

    @property
    def channels(self) -> int:
        return 8

    def to(self, device):
        self.device = device
        return self
