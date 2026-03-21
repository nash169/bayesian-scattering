import os
import json
import torch
import numpy as np

from abc import ABC, abstractmethod
from filelock import FileLock
from torch.utils.data import Dataset
from typing import Optional, Self


class AbstractDataset(Dataset, ABC):
    @abstractmethod
    def __init__(
        self,
        cache_root: Optional[str] = None,
        dataset_id: str = "dataset_it",
        shard_size: int = 10_000,
        dtype=np.float32
    ):
        if cache_root is not None:
            self.shard_size = int(shard_size)
            self.dtype = np.dtype(dtype)
            self.dataset_id = dataset_id

            self.cache_dir = os.path.join(cache_root, dataset_id)
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(os.path.join(self.cache_dir, "locks"), exist_ok=True)

            self.meta_path = os.path.join(self.cache_dir, "meta.json")

            # lazily opened per worker
            self._x = None
            self._meta = None
            self._data_mms = {}     # shard_id -> memmap
            self._present_mms = {}  # shard_id -> memmap

            self._init_lock = FileLock(os.path.join(self.cache_dir, "init.lock"))

    @abstractmethod
    def generate_sample(self, idx) -> torch.Tensor: ...

    @property
    @abstractmethod
    def shape(self) -> torch.Size: ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @abstractmethod
    def to(self, device) -> Self: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        if self.cache_dir is not None:
            if self._x is None:
                self._ensure_initialized(probe_idx=idx)

            shard_id = idx // self.shard_size
            off = idx % self.shard_size
            self._open_shard(shard_id)

            if self._present_mms[shard_id][off] == 1:
                if self.channels > 1:
                    return torch.from_numpy(self._data_mms[shard_id][off]).reshape(self.channels, *self.shape), self.labels[idx]
                else:
                    return torch.from_numpy(self._data_mms[shard_id][off]).reshape(*self.shape), self.labels[idx]

        sample = self.generate_sample(idx) \
            .detach() \
            .to(dtype=torch.float32, device="cpu") \
            .contiguous()

        if self.cache_dir is not None:
            if sample.numel() != self._x:
                raise ValueError(f"Output length changed: got {sample.numel()} expected {self._x} for this config")

            self._write_one(idx, sample)

        return sample, self.labels[idx]

    # def __getitem__(self, idx):
    #     label = self.labels[idx]
    #
    #     if hasattr(self, "store_path") and idx in self.store_indices:
    #         print("load sample")
    #         return sample_loader(self.store_path, f"{self.store_prefix}_{idx}"), label
    #     else:
    #         print("generate sample")
    #         sample = self.generate_sample(idx)
    #
    #     if hasattr(self, "store_path"):
    #         print("store sample")
    #         self.store_indices += [idx]
    #         torch.save(sample, self.store_path.joinpath(f"{self.store_prefix}_{idx}.pt"))
    #
    #     return sample, label

    def normalized_labels(self, idx=None):
        if not hasattr(self, "_labels_raw"):
            if torch.is_tensor(self.labels):
                self._labels_raw = self.labels.clone()
            else:
                self._labels_raw = self.labels.copy()

        labels_source = self._labels_raw
        labels_subset = labels_source[idx] if idx is not None else labels_source

        if torch.is_tensor(labels_subset):
            mu_y = labels_subset.mean()
            std_y = labels_subset.std()
            self.labels = labels_source.clone()
            self.labels.sub_(mu_y).div_(std_y)
        else:
            mu_y = labels_subset.mean()
            std_y = labels_subset.std()
            self.labels = labels_source.copy()
            self.labels = (self.labels - mu_y) / std_y

        self.labels_norm = (mu_y, std_y)

    def _read_meta(self):
        if not os.path.exists(self.meta_path):
            return None
        with open(self.meta_path, "r") as f:
            return json.load(f)

    def _write_meta(self, x: int):
        meta = {
            "id": self.dataset_id,
            "dtype": "float32",
            "N": self.__len__(),
            "x": int(x),
            "shard_size": self.shard_size,
        }
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, self.meta_path)

    def _num_shards(self):
        return (self.__len__() + self.shard_size - 1) // self.shard_size

    def _shard_paths(self, shard_id: int):
        data_path = os.path.join(self.cache_dir, f"data_shard_{shard_id:05d}.dat")
        present_path = os.path.join(self.cache_dir, f"present_shard_{shard_id:05d}.dat")
        lock_path = os.path.join(self.cache_dir, "locks", f"{shard_id:05d}.lock")
        return data_path, present_path, lock_path

    def _ensure_initialized(self, probe_idx: int):
        meta = self._read_meta()
        if meta is not None:
            self._meta = meta
            self._x = int(meta["x"])
            return

        with self._init_lock:
            meta = self._read_meta()
            if meta is not None:
                self._meta = meta
                self._x = int(meta["x"])
                return

            # probe once to learn x
            y = self.generate_sample(probe_idx)
            if not isinstance(y, torch.Tensor):
                raise TypeError("transform_fn must return a torch.Tensor")
            y = y.detach().to(dtype=torch.float32, device="cpu").contiguous()
            x = y.numel()
            self._x = x

            self._write_meta(x)
            self._meta = self._read_meta()

            # create shard files (allocate)
            for sid in range(self._num_shards()):
                data_path, present_path, _ = self._shard_paths(sid)
                # last shard may be smaller
                shard_len = min(self.shard_size, self.__len__() - sid * self.shard_size)

                if not os.path.exists(data_path):
                    mm = np.memmap(data_path, mode="w+", dtype=self.dtype, shape=(shard_len, x))
                    mm.flush()
                    del mm

                if not os.path.exists(present_path):
                    pm = np.memmap(present_path, mode="w+", dtype=np.uint8, shape=(shard_len,))
                    pm[:] = 0
                    pm.flush()
                    del pm

            # store probed item
            self._write_one(probe_idx, y)

    def _open_shard(self, shard_id: int):
        if shard_id in self._data_mms:
            return
        data_path, present_path, _ = self._shard_paths(shard_id)
        shard_len = min(self.shard_size, self.__len__() - shard_id * self.shard_size)

        self._data_mms[shard_id] = np.memmap(
            data_path,
            mode="r+",
            dtype=self.dtype,
            shape=(shard_len, self._x)
        )
        self._present_mms[shard_id] = np.memmap(
            present_path,
            mode="r+",
            dtype=np.uint8,
            shape=(shard_len,)
        )

    def _write_one(self, idx: int, y: torch.Tensor):
        shard_id = idx // self.shard_size
        off = idx % self.shard_size
        self._open_shard(shard_id)
        _, _, lock_path = self._shard_paths(shard_id)

        with FileLock(lock_path):
            if self._present_mms[shard_id][off] == 0:
                self._data_mms[shard_id][off] = y.flatten().numpy()
                self._present_mms[shard_id][off] = 1
                self._data_mms[shard_id].flush()
                self._present_mms[shard_id].flush()
