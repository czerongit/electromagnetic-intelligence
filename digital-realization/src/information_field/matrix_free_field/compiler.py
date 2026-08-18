from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from information_field.causal_minimal import (
    CausalMinimalRealization,
    ReductionCertificate,
    block_krylov_basis,
)
from information_field.quotient_response import SparseRelationSource


Tensor = torch.Tensor


def _digest_tensor(digest, value: Tensor) -> None:
    cpu = value.detach().contiguous().cpu()
    digest.update(str(tuple(cpu.shape)).encode())
    digest.update(str(cpu.dtype).encode())
    digest.update(cpu.numpy().tobytes())


def _structural_digest(
    source: SparseRelationSource,
    relation_port: Tensor,
    observation: Tensor,
    calibration: float,
    initial_position_port: Tensor,
    initial_velocity_port: Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"matrix-free-relation-field-v1")
    digest.update(source.digest.encode())
    digest.update(repr(float(calibration)).encode())
    for value in (
        relation_port,
        observation,
        initial_position_port,
        initial_velocity_port,
    ):
        _digest_tensor(digest, value)
    return digest.hexdigest()


def _relative_residual(actual: Tensor, expected: Tensor) -> float:
    difference = float(torch.linalg.matrix_norm(actual - expected).item())
    scale = max(1.0, float(torch.linalg.matrix_norm(expected).item()))
    return difference / scale


def _orthogonal_block(candidate: Tensor, basis: Tensor, tolerance: float) -> Tensor:
    if candidate.shape[1] == 0:
        return candidate
    residual = candidate
    if basis.shape[1]:
        residual = residual - basis @ (basis.T @ residual)
        residual = residual - basis @ (basis.T @ residual)
    left, singular, _ = torch.linalg.svd(residual, full_matrices=False)
    if singular.numel() == 0:
        return residual[:, :0]
    return left[:, singular > tolerance]


@dataclass
class OperatorCounter:
    applications: int = 0
    maximum_block_width: int = 0


class FactorizedIntrinsicOperator:
    def __init__(
        self,
        source: SparseRelationSource,
        calibration: float,
        counter: OperatorCounter,
    ) -> None:
        self.source = source
        self.calibration = calibration
        self.counter = counter

    def __call__(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[0] != self.source.quantity_dim:
            raise ValueError("operator block has the wrong quantity dimension")
        self.counter.applications += 1
        self.counter.maximum_block_width = max(
            self.counter.maximum_block_width, values.shape[1]
        )
        relation = self.source.whitened_adjoint(values.T)
        return self.calibration * self.source.whitened_apply(relation).T


def matrix_free_block_krylov(
    apply_operator,
    seed: Tensor,
    *,
    ambient_dimension: int,
    tolerance: float,
) -> Tensor:
    basis = seed[:, :0]
    frontier = _orthogonal_block(seed, basis, tolerance)
    for _ in range(ambient_dimension):
        if frontier.shape[1] == 0:
            break
        basis = torch.cat((basis, frontier), dim=1)
        if basis.shape[1] == ambient_dimension:
            break
        frontier = _orthogonal_block(apply_operator(frontier), basis, tolerance)
    return basis


@dataclass(frozen=True)
class MatrixFreeAccounting:
    ambient_dimension: int
    source_nonzeros: int
    reachable_dimension: int
    minimal_dimension: int
    factorized_operator_applications: int
    d_applications: int
    adjoint_applications: int
    maximum_block_width: int
    source_bytes: int
    retained_execution_bytes: int
    dense_relation_operator_materialized: bool
    dense_intrinsic_operator_materialized: bool


@dataclass(frozen=True)
class MatrixFreeCompilation:
    realization: CausalMinimalRealization
    accounting: MatrixFreeAccounting
    reachable_basis: Tensor

    def is_valid_for(
        self,
        source: SparseRelationSource,
        relation_port: Tensor,
        observation: Tensor,
        *,
        calibration: float = 1.0,
        initial_position_port: Tensor | None = None,
        initial_velocity_port: Tensor | None = None,
    ) -> bool:
        n = source.quantity_dim
        position = (
            torch.empty((n, 0), dtype=source.dtype, device=source.device)
            if initial_position_port is None
            else initial_position_port
        )
        velocity = (
            torch.empty((n, 0), dtype=source.dtype, device=source.device)
            if initial_velocity_port is None
            else initial_velocity_port
        )
        return self.realization.certificate.structural_digest == _structural_digest(
            source, relation_port, observation, calibration, position, velocity
        )


def _default_tolerance(source: SparseRelationSource, calibration: float) -> float:
    eps = torch.finfo(source.dtype).eps
    scale = max(
        1.0,
        float(torch.linalg.vector_norm(source.values).item()),
        abs(calibration),
    )
    return 64.0 * max(source.quantity_dim, source.relation_dim) * eps * scale


def _source_bytes(source: SparseRelationSource) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in (
            source.rows,
            source.columns,
            source.values,
            source.quantity_metric,
            source.relation_metric,
        )
    )


def compile_matrix_free_relation_field(
    source: SparseRelationSource,
    relation_port: Tensor,
    observation: Tensor,
    *,
    calibration: float = 1.0,
    initial_position_port: Tensor | None = None,
    initial_velocity_port: Tensor | None = None,
    tolerance: float | None = None,
) -> MatrixFreeCompilation:
    if calibration <= 0:
        raise ValueError("calibration must be positive")
    if relation_port.ndim != 2 or relation_port.shape[0] != source.relation_dim:
        raise ValueError("relation port has the wrong relation dimension")
    if observation.ndim != 2 or observation.shape[1] != source.quantity_dim:
        raise ValueError("observation has the wrong quantity dimension")
    relation_port = relation_port.to(device=source.device, dtype=source.dtype)
    observation = observation.to(device=source.device, dtype=source.dtype)
    n = source.quantity_dim
    position_port = (
        torch.empty((n, 0), dtype=source.dtype, device=source.device)
        if initial_position_port is None
        else initial_position_port.to(device=source.device, dtype=source.dtype)
    )
    velocity_port = (
        torch.empty((n, 0), dtype=source.dtype, device=source.device)
        if initial_velocity_port is None
        else initial_velocity_port.to(device=source.device, dtype=source.dtype)
    )
    if position_port.ndim != 2 or position_port.shape[0] != n:
        raise ValueError("initial position port has the wrong quantity dimension")
    if velocity_port.ndim != 2 or velocity_port.shape[0] != n:
        raise ValueError("initial velocity port has the wrong quantity dimension")
    tolerance = (
        _default_tolerance(source, calibration)
        if tolerance is None
        else float(tolerance)
    )
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    counter = OperatorCounter()
    apply_operator = FactorizedIntrinsicOperator(source, calibration, counter)
    whitened_relation_port = (
        torch.sqrt(source.relation_metric)[:, None] * relation_port
    )
    incident = source.whitened_apply(whitened_relation_port.T).T
    whitened_observation = observation / torch.sqrt(source.quantity_metric)[None, :]
    whitened_position = torch.sqrt(source.quantity_metric)[:, None] * position_port
    whitened_velocity = torch.sqrt(source.quantity_metric)[:, None] * velocity_port
    seed = torch.cat((incident, whitened_position, whitened_velocity), dim=1)

    reachable = matrix_free_block_krylov(
        apply_operator,
        seed,
        ambient_dimension=n,
        tolerance=tolerance,
    )
    applied_reachable = apply_operator(reachable) if reachable.shape[1] else reachable
    reachable_operator = reachable.T @ applied_reachable
    reachable_operator = 0.5 * (reachable_operator + reachable_operator.T)
    reachable_incident = reachable.T @ incident
    reachable_position = reachable.T @ whitened_position
    reachable_velocity = reachable.T @ whitened_velocity
    reachable_observation = whitened_observation @ reachable

    observable = block_krylov_basis(
        reachable_operator,
        reachable_observation.T,
        tolerance=tolerance,
    )
    lift = reachable @ observable
    reduced_operator = observable.T @ reachable_operator @ observable
    reduced_operator = 0.5 * (reduced_operator + reduced_operator.T)
    reduced_incident = observable.T @ reachable_incident
    reduced_position = observable.T @ reachable_position
    reduced_velocity = observable.T @ reachable_velocity
    reduced_observation = reachable_observation @ observable

    reachable_residual = (
        _relative_residual(reachable @ reachable_operator, applied_reachable)
        if reachable.shape[1]
        else 0.0
    )
    observable_residual = (
        _relative_residual(
            observable @ reduced_operator,
            reachable_operator @ observable,
        )
        if observable.shape[1]
        else 0.0
    )
    markov_residual = 0.0
    port_pairs = tuple(
        (full, reduced)
        for full, reduced in (
            (incident, reduced_incident),
            (whitened_position, reduced_position),
            (whitened_velocity, reduced_velocity),
        )
        if full.shape[1]
    )
    full_powers = [full for full, _ in port_pairs]
    reduced_powers = [reduced for _, reduced in port_pairs]
    scale = max(1.0, float(torch.linalg.matrix_norm(reachable_operator).item()))
    for _ in range(max(1, reachable.shape[1])):
        for full, reduced in zip(full_powers, reduced_powers):
            markov_residual = max(
                markov_residual,
                _relative_residual(
                    reduced_observation @ reduced,
                    whitened_observation @ full,
                ),
            )
        full_powers = [apply_operator(value) / scale for value in full_powers]
        reduced_powers = [reduced_operator @ value / scale for value in reduced_powers]

    eigenvalues, eigenmodes = torch.linalg.eigh(reduced_operator)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    modal_incident = eigenmodes.T @ reduced_incident
    modal_observation = reduced_observation @ eigenmodes
    modal_position = eigenmodes.T @ reduced_position
    modal_velocity = eigenmodes.T @ reduced_velocity
    structural = _structural_digest(
        source,
        relation_port,
        observation,
        calibration,
        position_port,
        velocity_port,
    )
    symmetry_residual = _relative_residual(reduced_operator, reduced_operator.T)
    positivity_floor = float(eigenvalues.min().item()) if eigenvalues.numel() else 0.0
    certificate = ReductionCertificate(
        ambient_dimension=n,
        reachable_dimension=reachable.shape[1],
        minimal_dimension=observable.shape[1],
        incident_width=relation_port.shape[1],
        observation_width=observation.shape[0],
        initial_position_width=position_port.shape[1],
        initial_velocity_width=velocity_port.shape[1],
        tolerance=tolerance,
        symmetry_residual=symmetry_residual,
        positivity_floor=positivity_floor,
        reachable_invariance_residual=reachable_residual,
        observable_invariance_residual=observable_residual,
        maximum_markov_residual=markov_residual,
        execution_digest=structural,
        structural_digest=structural,
    )
    realization = CausalMinimalRealization(
        reduced_operator,
        reduced_incident,
        reduced_observation,
        lift,
        reduced_position,
        reduced_velocity,
        eigenvalues,
        eigenmodes,
        modal_incident,
        modal_observation,
        modal_position,
        modal_velocity,
        certificate,
    )
    accounting = MatrixFreeAccounting(
        ambient_dimension=n,
        source_nonzeros=source.nnz,
        reachable_dimension=reachable.shape[1],
        minimal_dimension=observable.shape[1],
        factorized_operator_applications=counter.applications,
        d_applications=counter.applications + 1,
        adjoint_applications=counter.applications,
        maximum_block_width=counter.maximum_block_width,
        source_bytes=_source_bytes(source),
        retained_execution_bytes=realization.execution_bytes,
        dense_relation_operator_materialized=False,
        dense_intrinsic_operator_materialized=False,
    )
    return MatrixFreeCompilation(realization, accounting, reachable)
