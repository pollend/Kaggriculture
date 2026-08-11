"""Torch device selection shared by persistent and standalone DQN trainers."""

from __future__ import annotations

import os

import torch

DEVICE_ENV = "KAGGRICULTURE_TORCH_DEVICE"


def resolve_device(configured: str = "auto") -> torch.device:
    requested = os.environ.get(DEVICE_ENV, configured).strip().lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this PyTorch environment")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
    try:
        return torch.device(requested)
    except RuntimeError as exc:
        raise ValueError(f"unsupported torch device: {requested!r}") from exc


def move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move tensors restored from a CPU-mapped optimizer checkpoint."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
