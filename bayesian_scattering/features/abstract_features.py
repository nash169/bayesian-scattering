import os
import json
import numpy as np
import torch

import logging
logging.getLogger("filelock").setLevel(logging.INFO)

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from filelock import FileLock
from typing import Self, Optional, Union, cast

from bayesian_scattering.datasets.abstract_dataset import AbstractDataset


class AbstractFeatures(Dataset, ABC):
    def __init__(
        self,
        dataset: Optional[AbstractDataset] = None,
        cache_root: Optional[str] = None,
        transform_id: str = "features_id",
        shard_size: int = 5_000,
        dtype=np.float32,
    ):
        if dataset is not None:
            self.base = dataset
            self.N = len(dataset)

        if cache_root is not None:
            self.shard_size = int(shard_size)
            self.dtype = np.dtype(dtype)
            self.transform_id = transform_id

            self.cache_dir = os.path.join(cache_root, transform_id)
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
    def features_from_sample(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def to(self, device) -> Self:
        ...

    def __len__(self):
        assert hasattr(self, "base"), "Method not available without a base dataset."
        return self.N

    def __call__(
        self,
        x
    ):
        return self.features_from_sample(x)

    def __getitem__(self, idx: Union[int, torch.Tensor, np.ndarray]):
        assert hasattr(self, "base"), "Method not available without a base dataset."
        if isinstance(idx, torch.Tensor):
            idx_tensor = cast(torch.Tensor, idx)
            if idx_tensor.numel() != 1:
                raise ValueError("Expected scalar index tensor")
            idx = int(idx_tensor.item())
        elif isinstance(idx, np.ndarray):
            idx_array = cast(np.ndarray, idx)
            if idx_array.size != 1:
                raise ValueError("Expected scalar index array")
            idx = int(idx_array.item())
        if self.cache_dir is not None:
            if self._x is None:
                self._ensure_initialized(probe_idx=idx)

            shard_id = idx // self.shard_size
            off = idx % self.shard_size
            self._open_shard(shard_id)

            if self._present_mms[shard_id][off] == 1:
                return torch.from_numpy(self._data_mms[shard_id][off]), self.base[idx][1]

        sample, label = self.base[idx]
        y = self.features_from_sample(sample).detach().to(dtype=torch.float32, device="cpu").flatten().contiguous()

        if self.cache_dir is not None:
            if y.numel() != self._x:
                raise ValueError(f"Output length changed: got {y.numel()} expected {self._x} for this config")

            self._write_one(idx, y)

        return y, label

    def _read_meta(self):
        if not os.path.exists(self.meta_path):
            return None
        with open(self.meta_path, "r") as f:
            return json.load(f)

    def _write_meta(self, x: int):
        meta = {
            "id": self.transform_id,
            "dtype": "float32",
            "N": self.N,
            "x": int(x),
            "shard_size": self.shard_size,
        }
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, self.meta_path)

    def _num_shards(self):
        return (self.N + self.shard_size - 1) // self.shard_size

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
            y = self.features_from_sample(self.base[probe_idx][0])
            if not isinstance(y, torch.Tensor):
                raise TypeError("transform_fn must return a torch.Tensor")
            y = y.detach().to(dtype=torch.float32, device="cpu").flatten().contiguous()
            x = y.numel()
            self._x = x

            self._write_meta(x)
            self._meta = self._read_meta()

            # create shard files (allocate)
            for sid in range(self._num_shards()):
                data_path, present_path, _ = self._shard_paths(sid)
                # last shard may be smaller
                shard_len = min(self.shard_size, self.N - sid * self.shard_size)

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
        shard_len = min(self.shard_size, self.N - shard_id * self.shard_size)

        self._data_mms[shard_id] = np.memmap(
            data_path, mode="r+", dtype=self.dtype, shape=(shard_len, self._x)
        )
        self._present_mms[shard_id] = np.memmap(
            present_path, mode="r+", dtype=np.uint8, shape=(shard_len,)
        )

    def _write_one(self, idx: int, y: torch.Tensor):
        shard_id = idx // self.shard_size
        off = idx % self.shard_size
        self._open_shard(shard_id)
        _, _, lock_path = self._shard_paths(shard_id)

        with FileLock(lock_path):
            if self._present_mms[shard_id][off] == 0:
                self._data_mms[shard_id][off] = y.numpy()
                self._present_mms[shard_id][off] = 1
                self._data_mms[shard_id].flush()
                self._present_mms[shard_id].flush()
