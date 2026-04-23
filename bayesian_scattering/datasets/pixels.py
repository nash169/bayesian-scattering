import torch
import torchvision.transforms.functional as F
import pickle
from importlib.resources import files
from bayesian_scattering.datasets.abstract_dataset import AbstractDataset


class Pixels(AbstractDataset):
    def __init__(
            self,
            dataset_name,
            split,
            dataset_path=None,
            **kwargs
    ):
        if dataset_path is None:
            dataset_path = files(".")

        if split == "train":
            with open(dataset_path.joinpath(f'{dataset_name}/images_train.pkl'), 'rb') as f_x:
                self.samples = torch.from_numpy(pickle.load(f_x)).float().permute(0, 3, 1, 2).contiguous() / 255.
            with open(dataset_path.joinpath(f'{dataset_name}/labels_train.pkl'), 'rb') as f_y:
                self.labels = torch.from_numpy(pickle.load(f_y)).float().contiguous()
        elif split == "test":
            with open(dataset_path.joinpath(f'{dataset_name}/images_test.pkl'), 'rb') as f_x:
                self.samples = torch.from_numpy(pickle.load(f_x)).float().permute(0, 3, 1, 2).contiguous() / 255.
            with open(dataset_path.joinpath(f'{dataset_name}/labels_test.pkl'), 'rb') as f_y:
                self.labels = torch.from_numpy(pickle.load(f_y)).float().contiguous()

        self.dataset_name = dataset_name
        self.device = torch.device("cpu")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    def generate_sample(self, idx):
        NotImplementedError("This should not be called for Pixels class")

    @property
    def shape(self) -> torch.Size:
        return self.samples[0].shape[-2:]

    @property
    def channels(self) -> int:
        return 3

    def to(self, device):
        self.device = device
        return self
