from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from .relation import RelationNativeSource


_EXTENSION = None


def source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "native" / "quotient_response_cuda.cu"


def load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        digest = hashlib.sha256(source_path().read_bytes()).hexdigest()[:12]
        _EXTENSION = load(
            name=f"information_field_quotient_response_{digest}",
            sources=[str(source_path())],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _EXTENSION


class CudaQuotientResponse:
    def __init__(self) -> None:
        self.extension = load_extension()

    def __call__(self, source: RelationNativeSource) -> torch.Tensor:
        return self.extension.forward(
            source.observed_columns,
            source.incidents.indices,
            source.incidents.amplitudes,
            source.incidents.valid,
        )
