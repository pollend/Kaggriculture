import pytest
import torch

from agents.dqn_job_scorer.training.device import DEVICE_ENV, resolve_device


def test_auto_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_environment_overrides_config(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV, "cpu")
    assert resolve_device("mps").type == "cpu"


def test_unavailable_mps_is_an_explicit_error(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS was requested"):
        resolve_device("mps")
