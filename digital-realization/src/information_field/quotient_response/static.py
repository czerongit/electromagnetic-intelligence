from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from .source import SparseIncidentBatch, SparseRelationSource, Tensor, _digest_tensor


@dataclass(frozen=True)
class StaticAccounting:
    compiled_features: int
    compiled_operator_nonzeros: int
    retained_bytes: int
    projector_applications: int
    ambient_source_materializations: int
    dense_observation_operator_materialized: bool


@dataclass(frozen=True)
class PreparedStaticIncidents:
    local_indices: Tensor
    amplitudes: Tensor
    valid: Tensor

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "PreparedStaticIncidents":
        return PreparedStaticIncidents(
            self.local_indices.to(device=device),
            self.amplitudes.to(
                device=device, dtype=dtype or self.amplitudes.dtype
            ),
            self.valid.to(device=device),
        )


@dataclass(frozen=True)
class CompiledStaticResponse:
    source_digest: str
    observation_digest: str
    relation_dim: int
    selected_features: Tensor
    feature_lookup: Tensor
    observed_columns: Tensor
    accounting: StaticAccounting

    @property
    def device(self) -> torch.device:
        return self.observed_columns.device

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "CompiledStaticResponse":
        target_dtype = dtype or self.observed_columns.dtype
        return CompiledStaticResponse(
            self.source_digest,
            self.observation_digest,
            self.relation_dim,
            self.selected_features.to(device=device),
            self.feature_lookup.to(device=device),
            self.observed_columns.to(device=device, dtype=target_dtype),
            self.accounting,
        )

    def prepare(self, incidents: SparseIncidentBatch) -> PreparedStaticIncidents:
        if incidents.indices.device != self.device:
            raise ValueError("compiled response and incidents must share a device")
        active_global = incidents.indices[incidents.valid]
        if active_global.numel() and (
            int(active_global.min()) < 0 or int(active_global.max()) >= self.relation_dim
        ):
            raise ValueError("incident index is outside the relation carrier")
        safe_global = torch.where(
            incidents.valid, incidents.indices, torch.zeros_like(incidents.indices)
        )
        local = self.feature_lookup[safe_global]
        missing = incidents.valid & (local < 0)
        if bool(missing.any()):
            raise ValueError("incident uses a feature outside the compiled quotient path")
        return PreparedStaticIncidents(
            torch.where(incidents.valid, local, torch.zeros_like(local)),
            torch.where(
            incidents.valid, incidents.amplitudes, torch.zeros_like(incidents.amplitudes)
            ),
            incidents.valid,
        )

    def run_prepared(self, incidents: PreparedStaticIncidents) -> Tensor:
        columns = self.observed_columns[incidents.local_indices]
        return torch.sum(columns * incidents.amplitudes[..., None], dim=1)

    def run(self, incidents: SparseIncidentBatch) -> Tensor:
        return self.run_prepared(self.prepare(incidents))


def observation_digest(observation: Tensor) -> str:
    digest = hashlib.sha256()
    _digest_tensor(digest, observation)
    return digest.hexdigest()


def compile_static_response(
    source: SparseRelationSource,
    observation: Tensor,
    admitted_features: Tensor,
) -> CompiledStaticResponse:
    if admitted_features.device != source.device:
        admitted_features = admitted_features.to(source.device)
    selected = torch.unique(admitted_features.to(torch.int64), sorted=True)
    if selected.numel() and (int(selected.min()) < 0 or int(selected.max()) >= source.relation_dim):
        raise ValueError("compiled feature is outside the relation carrier")
    observation = observation.to(device=source.device, dtype=source.dtype)
    columns = source.observed_columns(observation, selected)
    lookup = torch.full(
        (source.relation_dim,), -1, dtype=torch.int64, device=source.device
    )
    if selected.numel():
        lookup[selected] = torch.arange(selected.numel(), device=source.device)
    selected_nnz = int(torch.isin(source.columns, selected).sum().item()) if selected.numel() else 0
    retained = sum(
        value.numel() * value.element_size()
        for value in (selected, lookup, columns)
    )
    return CompiledStaticResponse(
        source.digest,
        observation_digest(observation),
        source.relation_dim,
        selected,
        lookup,
        columns,
        StaticAccounting(
            int(selected.numel()),
            selected_nnz,
            retained,
            0,
            0,
            False,
        ),
    )


def dense_static_oracle(
    source: SparseRelationSource,
    observation: Tensor,
    incidents: SparseIncidentBatch,
) -> Tensor:
    dense = incidents.dense(source.relation_dim)
    return (observation @ source.apply(dense).T).T


def ambient_projector_oracle(
    source: SparseRelationSource,
    observation: Tensor,
    incidents: SparseIncidentBatch,
) -> Tensor:
    dense = incidents.dense(source.relation_dim)
    induced = source.apply(dense)
    admitted = (source.projector() @ induced.T).T
    return (observation @ admitted.T).T
