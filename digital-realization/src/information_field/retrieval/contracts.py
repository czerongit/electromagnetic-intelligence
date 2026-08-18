"""Shared typed contract for substitutable attention implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AttentionInput:
    prefix_values: Tensor
    incident_values: Tensor
    prefix_positions: Tensor
    incident_positions: Tensor
    prefix_valid: Tensor
    incident_valid: Tensor


@dataclass(frozen=True)
class AttentionAdapterState:
    adapter_name: str
    payload: Any


@dataclass(frozen=True)
class AttentionDiagnostics:
    adapter_name: str
    internal_family: str
    backend: str
    context_frozen: bool
    prefix_only_context: bool
    incident_independent: bool
    geometric_boundary: bool
    condition_numbers: Tensor | None


@dataclass(frozen=True)
class AttentionResult:
    responses: Tensor
    state: AttentionAdapterState
    diagnostics: AttentionDiagnostics


@dataclass(frozen=True)
class ValidatedAttentionInput:
    batch_size: int
    prefix_length: int
    incident_length: int
    fiber_width: int


def validate_attention_input(
    value: AttentionInput,
) -> ValidatedAttentionInput:
    tensors = {
        "prefix values": value.prefix_values,
        "incident values": value.incident_values,
        "prefix positions": value.prefix_positions,
        "incident positions": value.incident_positions,
        "prefix validity mask": value.prefix_valid,
        "incident validity mask": value.incident_valid,
    }
    if value.prefix_values.ndim != 3:
        raise ValueError(
            "prefix values must have shape (batch, prefix, fiber)"
        )
    if value.incident_values.ndim != 3:
        raise ValueError(
            "incident values must have shape "
            "(batch, incident, fiber)"
        )
    batch, prefix, fiber = value.prefix_values.shape
    incident_batch, incident, incident_fiber = (
        value.incident_values.shape
    )
    if (
        batch != incident_batch
        or fiber != incident_fiber
        or batch < 1
        or prefix < 1
        or incident < 1
        or fiber < 1
    ):
        raise ValueError(
            "prefix and incident batches must have compatible "
            "positive dimensions"
        )
    expected_shapes = {
        "prefix positions": (batch, prefix),
        "incident positions": (batch, incident),
        "prefix validity mask": (batch, prefix),
        "incident validity mask": (batch, incident),
    }
    for name, expected in expected_shapes.items():
        if tensors[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if (
        value.prefix_valid.dtype != torch.bool
        or value.incident_valid.dtype != torch.bool
    ):
        raise ValueError("validity masks must be boolean")
    if not value.prefix_values.is_floating_point():
        raise ValueError("attention values must be floating point")
    if (
        value.incident_values.dtype != value.prefix_values.dtype
        or value.prefix_positions.dtype
        != value.prefix_values.dtype
        or value.incident_positions.dtype
        != value.prefix_values.dtype
    ):
        raise ValueError(
            "values and positions must share one floating dtype"
        )
    device = value.prefix_values.device
    if any(tensor.device != device for tensor in tensors.values()):
        raise ValueError(
            "all attention inputs must share one device"
        )
    if any(
        not bool(torch.all(torch.isfinite(tensor)))
        for tensor in (
            value.prefix_values,
            value.incident_values,
            value.prefix_positions,
            value.incident_positions,
        )
    ):
        raise ValueError("attention inputs must be finite")
    if not bool(torch.all(torch.any(value.prefix_valid, dim=1))):
        raise ValueError(
            "every batch item requires one valid prefix sample"
        )
    return ValidatedAttentionInput(
        batch_size=batch,
        prefix_length=prefix,
        incident_length=incident,
        fiber_width=fiber,
    )


def validate_result(
    batch: AttentionInput,
    result: AttentionResult,
) -> None:
    expected = batch.incident_values.shape
    if result.responses.shape != expected:
        raise RuntimeError(
            f"adapter response must have shape {expected}"
        )
    if (
        result.responses.dtype != batch.incident_values.dtype
        or result.responses.device != batch.incident_values.device
    ):
        raise RuntimeError(
            "adapter response must preserve dtype and device"
        )
    invalid = ~batch.incident_valid
    if bool(torch.any(result.responses[invalid] != 0)):
        raise RuntimeError(
            "invalid incident positions must emit zero"
        )


class AttentionAdapter(nn.Module, ABC):
    """One external slot implemented by distinct attention families."""

    adapter_name: str

    def initialize_state(
        self, batch: AttentionInput
    ) -> AttentionAdapterState:
        """Freeze prefix state without consuming a valid incident."""

        empty = AttentionInput(
            prefix_values=batch.prefix_values,
            incident_values=batch.incident_values,
            prefix_positions=batch.prefix_positions,
            incident_positions=batch.incident_positions,
            prefix_valid=batch.prefix_valid,
            incident_valid=torch.zeros_like(batch.incident_valid),
        )
        return self.forward(empty).state

    @abstractmethod
    def forward(
        self,
        batch: AttentionInput,
        state: AttentionAdapterState | None = None,
    ) -> AttentionResult:
        raise NotImplementedError
