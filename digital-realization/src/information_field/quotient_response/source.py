from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


Tensor = torch.Tensor


def _digest_tensor(digest: "hashlib._Hash", value: Tensor) -> None:
    cpu = value.detach().contiguous().cpu()
    metadata = f"{cpu.dtype}:{tuple(cpu.shape)}".encode()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    payload = cpu.numpy().tobytes()
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


@dataclass(frozen=True)
class SparseIncident:
    indices: Tensor
    amplitudes: Tensor

    def __post_init__(self) -> None:
        if self.indices.ndim != 1 or self.amplitudes.ndim != 1:
            raise ValueError("incident indices and amplitudes must be vectors")
        if self.indices.shape != self.amplitudes.shape:
            raise ValueError("incident indices and amplitudes must have one shape")
        if self.indices.dtype != torch.int64:
            raise ValueError("incident indices must be int64")
        if not torch.isfinite(self.amplitudes).all():
            raise ValueError("incident amplitudes must be finite")

    def dense(self, relation_dim: int) -> Tensor:
        if self.indices.numel() and (
            int(self.indices.min()) < 0 or int(self.indices.max()) >= relation_dim
        ):
            raise ValueError("incident index is outside the relation carrier")
        result = torch.zeros(
            relation_dim,
            dtype=self.amplitudes.dtype,
            device=self.amplitudes.device,
        )
        if self.indices.numel():
            result.scatter_add_(0, self.indices, self.amplitudes)
        return result

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "SparseIncident":
        return SparseIncident(
            self.indices.to(device=device),
            self.amplitudes.to(device=device, dtype=dtype or self.amplitudes.dtype),
        )


@dataclass(frozen=True)
class SparseIncidentBatch:
    indices: Tensor
    amplitudes: Tensor
    valid: Tensor

    def __post_init__(self) -> None:
        if self.indices.ndim != 2 or self.amplitudes.shape != self.indices.shape:
            raise ValueError("batched incidents require matching batch-by-support tensors")
        if self.valid.shape != self.indices.shape or self.valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean batch-by-support tensor")
        if self.indices.dtype != torch.int64:
            raise ValueError("incident indices must be int64")
        if not torch.isfinite(self.amplitudes).all():
            raise ValueError("incident amplitudes must be finite")

    @property
    def batch_size(self) -> int:
        return self.indices.shape[0]

    @property
    def support_width(self) -> int:
        return self.indices.shape[1]

    def dense(self, relation_dim: int) -> Tensor:
        active = self.indices[self.valid]
        if active.numel() and (int(active.min()) < 0 or int(active.max()) >= relation_dim):
            raise ValueError("incident index is outside the relation carrier")
        result = torch.zeros(
            self.batch_size,
            relation_dim,
            dtype=self.amplitudes.dtype,
            device=self.amplitudes.device,
        )
        safe_indices = torch.where(self.valid, self.indices, torch.zeros_like(self.indices))
        values = torch.where(self.valid, self.amplitudes, torch.zeros_like(self.amplitudes))
        result.scatter_add_(1, safe_indices, values)
        return result

    def admitted_features(self) -> Tensor:
        return torch.unique(self.indices[self.valid], sorted=True)

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "SparseIncidentBatch":
        return SparseIncidentBatch(
            self.indices.to(device=device),
            self.amplitudes.to(device=device, dtype=dtype or self.amplitudes.dtype),
            self.valid.to(device=device),
        )


@dataclass(frozen=True)
class SparseRelationSource:
    quantity_dim: int
    relation_dim: int
    rows: Tensor
    columns: Tensor
    values: Tensor
    quantity_metric: Tensor
    relation_metric: Tensor

    def __post_init__(self) -> None:
        if self.quantity_dim < 1 or self.relation_dim < 1:
            raise ValueError("carrier dimensions must be positive")
        if self.rows.dtype != torch.int64 or self.columns.dtype != torch.int64:
            raise ValueError("sparse coordinates must be int64")
        if not (self.rows.ndim == self.columns.ndim == self.values.ndim == 1):
            raise ValueError("sparse coordinates and values must be vectors")
        if not (self.rows.shape == self.columns.shape == self.values.shape):
            raise ValueError("sparse coordinates and values must have one shape")
        if self.rows.numel() and (
            int(self.rows.min()) < 0
            or int(self.rows.max()) >= self.quantity_dim
            or int(self.columns.min()) < 0
            or int(self.columns.max()) >= self.relation_dim
        ):
            raise ValueError("sparse coordinate is outside its carrier")
        if self.quantity_metric.shape != (self.quantity_dim,):
            raise ValueError("quantity metric must be positive diagonal data")
        if self.relation_metric.shape != (self.relation_dim,):
            raise ValueError("relation metric must be positive diagonal data")
        if not torch.isfinite(self.values).all():
            raise ValueError("operator values must be finite")
        if not torch.isfinite(self.quantity_metric).all() or not torch.all(self.quantity_metric > 0):
            raise ValueError("quantity metric must be finite and positive")
        if not torch.isfinite(self.relation_metric).all() or not torch.all(self.relation_metric > 0):
            raise ValueError("relation metric must be finite and positive")

    @classmethod
    def from_dense(
        cls,
        operator: Tensor,
        *,
        quantity_metric: Tensor | None = None,
        relation_metric: Tensor | None = None,
        zero_tolerance: float = 0.0,
    ) -> "SparseRelationSource":
        if operator.ndim != 2:
            raise ValueError("relation operator must be a matrix")
        mask = torch.abs(operator) > zero_tolerance
        coordinates = torch.nonzero(mask, as_tuple=False)
        rows = coordinates[:, 0].to(torch.int64)
        columns = coordinates[:, 1].to(torch.int64)
        values = operator[rows, columns]
        h, g = operator.shape
        return cls(
            h,
            g,
            rows,
            columns,
            values,
            torch.ones(h, dtype=operator.dtype, device=operator.device)
            if quantity_metric is None
            else quantity_metric,
            torch.ones(g, dtype=operator.dtype, device=operator.device)
            if relation_metric is None
            else relation_metric,
        )

    @property
    def nnz(self) -> int:
        return self.values.numel()

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"{self.quantity_dim}:{self.relation_dim}".encode())
        for value in (
            self.rows,
            self.columns,
            self.values,
            self.quantity_metric,
            self.relation_metric,
        ):
            _digest_tensor(digest, value)
        return digest.hexdigest()

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "SparseRelationSource":
        target_dtype = dtype or self.dtype
        return SparseRelationSource(
            self.quantity_dim,
            self.relation_dim,
            self.rows.to(device=device),
            self.columns.to(device=device),
            self.values.to(device=device, dtype=target_dtype),
            self.quantity_metric.to(device=device, dtype=target_dtype),
            self.relation_metric.to(device=device, dtype=target_dtype),
        )

    def dense_operator(self) -> Tensor:
        result = torch.zeros(
            self.quantity_dim,
            self.relation_dim,
            dtype=self.dtype,
            device=self.device,
        )
        if self.nnz:
            flat = self.rows * self.relation_dim + self.columns
            result.view(-1).scatter_add_(0, flat, self.values)
        return result

    def apply(self, relation_values: Tensor) -> Tensor:
        squeeze = relation_values.ndim == 1
        values = relation_values.unsqueeze(0) if squeeze else relation_values
        if values.ndim != 2 or values.shape[1] != self.relation_dim:
            raise ValueError("relation values have the wrong carrier dimension")
        result = torch.zeros(
            values.shape[0], self.quantity_dim, dtype=values.dtype, device=values.device
        )
        if self.nnz:
            contribution = values[:, self.columns] * self.values
            result.scatter_add_(1, self.rows.expand(values.shape[0], -1), contribution)
        return result[0] if squeeze else result

    def adjoint_apply(self, quantity_values: Tensor) -> Tensor:
        squeeze = quantity_values.ndim == 1
        values = quantity_values.unsqueeze(0) if squeeze else quantity_values
        if values.ndim != 2 or values.shape[1] != self.quantity_dim:
            raise ValueError("quantity values have the wrong carrier dimension")
        result = torch.zeros(
            values.shape[0], self.relation_dim, dtype=values.dtype, device=values.device
        )
        if self.nnz:
            weighted = values[:, self.rows] * self.quantity_metric[self.rows]
            contribution = weighted * self.values / self.relation_metric[self.columns]
            result.scatter_add_(1, self.columns.expand(values.shape[0], -1), contribution)
        return result[0] if squeeze else result

    def whitened_apply(self, relation_values: Tensor) -> Tensor:
        unweighted = relation_values / torch.sqrt(self.relation_metric)
        return torch.sqrt(self.quantity_metric) * self.apply(unweighted)

    def whitened_adjoint(self, quantity_values: Tensor) -> Tensor:
        unweighted = quantity_values / torch.sqrt(self.quantity_metric)
        return torch.sqrt(self.relation_metric) * self.adjoint_apply(unweighted)

    def whitened_dense(self) -> Tensor:
        return (
            torch.sqrt(self.quantity_metric)[:, None]
            * self.dense_operator()
            / torch.sqrt(self.relation_metric)[None, :]
        )

    def projector(self) -> Tensor:
        whitened_range = torch.sqrt(self.quantity_metric)[:, None] * self.dense_operator()
        euclidean = whitened_range @ torch.linalg.pinv(whitened_range)
        return (
            euclidean
            * torch.rsqrt(self.quantity_metric)[:, None]
            * torch.sqrt(self.quantity_metric)[None, :]
        )

    def observed_columns(self, observation: Tensor, features: Tensor) -> Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.quantity_dim:
            raise ValueError("observation map has the wrong quantity dimension")
        if features.ndim != 1 or features.dtype != torch.int64:
            raise ValueError("features must be an int64 vector")
        result = torch.zeros(
            features.numel(), observation.shape[0],
            dtype=self.dtype, device=self.device,
        )
        if not features.numel() or not self.nnz:
            return result
        lookup = torch.full(
            (self.relation_dim,), -1, dtype=torch.int64, device=self.device
        )
        lookup[features] = torch.arange(features.numel(), device=self.device)
        local = lookup[self.columns]
        selected = local >= 0
        if bool(selected.any()):
            selected_local = local[selected]
            contribution = (
                observation[:, self.rows[selected]].T * self.values[selected, None]
            )
            result.scatter_add_(
                0,
                selected_local[:, None].expand(-1, observation.shape[0]),
                contribution,
            )
        return result

    def changed_features(self, other: "SparseRelationSource", tolerance: float = 0.0) -> Tensor:
        if (self.quantity_dim, self.relation_dim) != (other.quantity_dim, other.relation_dim):
            return torch.arange(max(self.relation_dim, other.relation_dim), dtype=torch.int64)
        if (
            torch.equal(self.rows.cpu(), other.rows.cpu())
            and torch.equal(self.columns.cpu(), other.columns.cpu())
        ):
            changed_entries = torch.abs(self.values.cpu() - other.values.cpu()) > tolerance
            return torch.unique(self.columns.cpu()[changed_entries], sorted=True)
        left = self.dense_operator().cpu()
        right = other.dense_operator().cpu()
        return torch.nonzero(torch.any(torch.abs(left - right) > tolerance, dim=0), as_tuple=False).flatten()
