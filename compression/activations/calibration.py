"""Shared CIFAR-10 calibration sample loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


_CALIB_CACHE: dict[tuple[str, int, int, int], tuple[DataLoader, list[int]]] = {}


@dataclass(frozen=True)
class CalibrationData:
    loader: DataLoader
    indices: list[int]

    @property
    def indices_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(",".join(str(i) for i in self.indices).encode("utf-8"))
        return h.hexdigest()


class RandomCIFARLikeDataset(Dataset):
    """Fallback synthetic dataset when CIFAR-10 is unavailable."""

    def __init__(self, num_items: int, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(num_items, 3, 32, 32, generator=gen)
        self.y = torch.randint(0, 10, (num_items,), generator=gen)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.x[idx], int(self.y[idx])


def _build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )


def get_calibration_data(
    data_dir: str,
    num_images: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> CalibrationData:
    key = (data_dir, int(num_images), int(batch_size), int(seed))
    if key in _CALIB_CACHE:
        loader, indices = _CALIB_CACHE[key]
        return CalibrationData(loader=loader, indices=indices)

    rng = np.random.default_rng(seed)
    try:
        train = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=_build_eval_transform(),
        )
        indices = rng.choice(len(train), size=num_images, replace=False).tolist()
        subset: Dataset = Subset(train, indices)
    except Exception:
        indices = list(range(num_images))
        subset = RandomCIFARLikeDataset(num_images, seed)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    _CALIB_CACHE[key] = (loader, indices)
    return CalibrationData(loader=loader, indices=indices)
