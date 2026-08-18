from __future__ import annotations

import math
from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class RelationIncidents:
    indices: Tensor
    amplitudes: Tensor
    valid: Tensor

    def __post_init__(self) -> None:
        if self.indices.ndim != 3:
            raise ValueError("relation incidents require batch by query by support tensors")
        if self.indices.shape != self.amplitudes.shape or self.indices.shape != self.valid.shape:
            raise ValueError("relation incident tensors must have one shape")
        if self.indices.dtype != torch.int64:
            raise ValueError("relation incident indices must be int64")
        if self.valid.dtype != torch.bool:
            raise ValueError("relation incident validity must be boolean")
        if not bool(torch.isfinite(self.amplitudes).all()):
            raise ValueError("relation incident amplitudes must be finite")

    def first(self) -> "RelationIncidents":
        return RelationIncidents(
            self.indices[:, :1].contiguous(),
            self.amplitudes[:, :1].contiguous(),
            self.valid[:, :1].contiguous(),
        )

    def to(self, device: str | torch.device, dtype: torch.dtype) -> "RelationIncidents":
        return RelationIncidents(
            self.indices.to(device=device),
            self.amplitudes.to(device=device, dtype=dtype),
            self.valid.to(device=device),
        )


@dataclass(frozen=True)
class RelationNativeSource:
    """Finite source operator represented by its observed relation columns."""

    observed_columns: Tensor
    incidents: RelationIncidents

    def __post_init__(self) -> None:
        if self.observed_columns.ndim != 3:
            raise ValueError("observed columns require batch by relation by output tensors")
        if self.observed_columns.shape[0] != self.incidents.indices.shape[0]:
            raise ValueError("source and incidents must have one batch dimension")
        active = self.incidents.indices[self.incidents.valid]
        if active.numel() and (
            int(active.min()) < 0 or int(active.max()) >= self.observed_columns.shape[1]
        ):
            raise ValueError("incident index lies outside the relation carrier")

    def to(self, device: str | torch.device, dtype: torch.dtype) -> "RelationNativeSource":
        return RelationNativeSource(
            self.observed_columns.to(device=device, dtype=dtype).contiguous(),
            self.incidents.to(device, dtype),
        )

    def first(self) -> "RelationNativeSource":
        return RelationNativeSource(self.observed_columns, self.incidents.first())


def _design(positions: Tensor, binding_count: int) -> Tensor:
    components = []
    for frequency in range(1, binding_count + 1):
        components.extend(
            (
                torch.cos(float(frequency) * positions) / math.sqrt(math.pi),
                torch.sin(float(frequency) * positions) / math.sqrt(math.pi),
            )
        )
    return torch.stack(components, dim=1)


def _determine_coefficients(design: Tensor, values: Tensor) -> Tensor:
    """Use the source measure directly when its localization basis is orthogonal."""

    gram = design.transpose(0, 1) @ design
    diagonal = torch.diagonal(gram)
    residual = gram - torch.diag_embed(diagonal)
    tolerance = 64.0 * torch.finfo(design.dtype).eps * max(
        1.0, float(torch.max(torch.abs(diagonal)).item())
    )
    if bool(torch.max(torch.abs(residual)) <= tolerance) and bool(
        torch.all(diagonal > 0)
    ):
        return (design.transpose(0, 1) @ values) / diagonal[:, None]
    return torch.linalg.lstsq(design, values).solution


def canonical_relation_incidents(attention_input, binding_count: int) -> RelationIncidents:
    """Represent the fixture's answer-free queries in native relation coordinates."""

    batch = attention_input.prefix_values.shape[0]
    query_count = attention_input.incident_values.shape[1]
    if query_count > binding_count:
        raise ValueError("query coordinate exceeds the relation carrier")
    indices = torch.arange(
        query_count,
        dtype=torch.int64,
        device=attention_input.prefix_values.device,
    ).reshape(1, query_count, 1).expand(batch, -1, -1).contiguous()
    amplitudes = torch.ones(
        (batch, query_count, 1),
        dtype=attention_input.prefix_values.dtype,
        device=attention_input.prefix_values.device,
    )
    return RelationIncidents(
        indices,
        amplitudes,
        attention_input.incident_valid.unsqueeze(-1).contiguous(),
    )


def determine_relation_source(
    attention_input,
    binding_count: int,
    incidents: RelationIncidents,
) -> RelationNativeSource:
    """Determine D from prefix observations before applying native incidents.

    No evaluator-held payload basis, permutation, or target enters this construction.
    Relation coordinate i is the declared frequency-labelled query coordinate. Prefix
    projection determines D e_i; a query supplies only e_i.
    """

    columns = []
    for item in range(attention_input.prefix_values.shape[0]):
        admitted = attention_input.prefix_valid[item]
        positions = attention_input.prefix_positions[item, admitted]
        values = attention_input.prefix_values[item, admitted]
        design = _design(positions, binding_count)
        if design.shape[0] < design.shape[1]:
            raise ValueError("prefix observations do not determine the relation source")
        coefficients = _determine_coefficients(design, values)
        columns.append(
            coefficients.reshape(binding_count, 2, values.shape[-1])[:, 0]
        )

    return RelationNativeSource(torch.stack(columns).contiguous(), incidents)


def determine_canonical_fixture(attention_input, binding_count: int) -> RelationNativeSource:
    incidents = canonical_relation_incidents(attention_input, binding_count)
    return determine_relation_source(attention_input, binding_count, incidents)


def dense_quotient_response(source: RelationNativeSource) -> Tensor:
    batch, queries, support = source.incidents.indices.shape
    width = source.observed_columns.shape[-1]
    result = torch.zeros(
        (batch, queries, width),
        dtype=source.observed_columns.dtype,
        device=source.observed_columns.device,
    )
    for slot in range(support):
        index = source.incidents.indices[:, :, slot]
        selected = torch.gather(
            source.observed_columns,
            1,
            index[..., None].expand(-1, -1, width),
        )
        amplitude = torch.where(
            source.incidents.valid[:, :, slot],
            source.incidents.amplitudes[:, :, slot],
            torch.zeros_like(source.incidents.amplitudes[:, :, slot]),
        )
        result = result + amplitude[..., None] * selected
    return result


def relation_native_attention_result(responses: Tensor):
    from .contracts import (
        AttentionAdapterState,
        AttentionDiagnostics,
        AttentionResult,
    )

    return AttentionResult(
        responses,
        AttentionAdapterState("relation-native-field", None),
        AttentionDiagnostics(
            "relation-native-field",
            "quotient response",
            responses.device.type,
            True,
            True,
            True,
            True,
            None,
        ),
    )
