from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION = None


def source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native" / "temporal_response_cuda.cu"


def load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        digest = hashlib.sha256(source_path().read_bytes()).hexdigest()[:12]
        _EXTENSION = load(
            name=f"information_field_temporal_response_{digest}",
            sources=[str(source_path())],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _EXTENSION


class TemporalCuda:
    def __init__(self) -> None:
        self.extension = load_extension()

    def regular(self, transitions: torch.Tensor, lengths: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        return self.extension.regular_forward(transitions, lengths, initial)

    def event(self, powers: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        return self.extension.event_forward(powers, initial)
