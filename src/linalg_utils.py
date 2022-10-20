import numpy as np
import torch
import torch.nn.functional as F


def normalize_segments(array: np.ndarray, size: int = 768) -> np.ndarray:
    num_segments = array.shape[1] // size
    for i in range(num_segments):
        array[:, (i - 1) * size : i * size] = F.normalize(
            array[:, (i - 1) * size : i * size]
        )
    return array


def normalize_segments_(array: np.ndarray) -> np.ndarray:
    num_segments = array.shape[1] // 1024
    for i in range(num_segments):
        array[:, (i - 1) * 1024 : i * 1024] = F.normalize(
            array[:, (i - 1) * 1024 : i * 1024]
        )
    return array


def f_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize vectors"""
    return F.normalize(x, dim=0)
