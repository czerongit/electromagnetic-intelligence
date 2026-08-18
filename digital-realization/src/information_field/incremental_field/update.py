from __future__ import annotations

from dataclasses import dataclass

import torch

from information_field.causal_minimal import CausalMinimalRealization, ReductionCertificate, block_krylov_basis
from information_field.matrix_free_field import MatrixFreeCompilation, compile_matrix_free_relation_field
from information_field.matrix_free_field.compiler import (
    FactorizedIntrinsicOperator,
    MatrixFreeAccounting,
    OperatorCounter,
    _source_bytes,
    _structural_digest,
)
from information_field.quotient_response import SparseRelationSource


Tensor = torch.Tensor


def _residual(value: Tensor, basis: Tensor) -> float:
    difference = value - basis @ (basis.T @ value)
    return float(torch.linalg.matrix_norm(difference).item()) / max(
        1.0, float(torch.linalg.matrix_norm(value).item())
    )


def _relative(actual: Tensor, expected: Tensor) -> float:
    return float(torch.linalg.matrix_norm(actual - expected).item()) / max(
        1.0, float(torch.linalg.matrix_norm(expected).item())
    )


@dataclass(frozen=True)
class IncrementalDecision:
    status: str
    reason: str
    incident_outside_residual: float
    invariance_outside_residual: float
    compilation: MatrixFreeCompilation | None
    factorized_operator_applications: int


def try_reduced_update(
    previous: MatrixFreeCompilation,
    old_source: SparseRelationSource,
    new_source: SparseRelationSource,
    relation_port: Tensor,
    observation: Tensor,
    *,
    calibration: float = 1.0,
    initial_position_port: Tensor | None = None,
    initial_velocity_port: Tensor | None = None,
    tolerance: float | None = None,
) -> IncrementalDecision:
    if (old_source.quantity_dim, old_source.relation_dim) != (
        new_source.quantity_dim,
        new_source.relation_dim,
    ):
        return IncrementalDecision(
            "global-recompile-required",
            "carrier dimension changed",
            float("inf"),
            float("inf"),
            None,
            0,
        )
    if not (
        torch.equal(old_source.quantity_metric.cpu(), new_source.quantity_metric.cpu())
        and torch.equal(old_source.relation_metric.cpu(), new_source.relation_metric.cpu())
    ):
        return IncrementalDecision(
            "global-recompile-required",
            "carrier metric changed",
            float("inf"),
            float("inf"),
            None,
            0,
        )
    if calibration <= 0:
        raise ValueError("calibration must be positive")
    relation_port = relation_port.to(new_source.device, new_source.dtype)
    observation = observation.to(new_source.device, new_source.dtype)
    n = new_source.quantity_dim
    position = (
        torch.empty((n, 0), dtype=new_source.dtype, device=new_source.device)
        if initial_position_port is None
        else initial_position_port.to(new_source.device, new_source.dtype)
    )
    velocity = (
        torch.empty((n, 0), dtype=new_source.dtype, device=new_source.device)
        if initial_velocity_port is None
        else initial_velocity_port.to(new_source.device, new_source.dtype)
    )
    if relation_port.ndim != 2 or relation_port.shape[0] != new_source.relation_dim:
        raise ValueError("relation port has the wrong relation dimension")
    if observation.ndim != 2 or observation.shape[1] != n:
        raise ValueError("observation has the wrong quantity dimension")
    if position.ndim != 2 or position.shape[0] != n:
        raise ValueError("initial position port has the wrong quantity dimension")
    if velocity.ndim != 2 or velocity.shape[0] != n:
        raise ValueError("initial velocity port has the wrong quantity dimension")
    tolerance = (
        previous.realization.certificate.tolerance
        if tolerance is None
        else float(tolerance)
    )
    basis = previous.reachable_basis.to(new_source.device, new_source.dtype)
    if basis.shape[0] != n:
        return IncrementalDecision(
            "global-recompile-required",
            "previous reachable carrier has another ambient dimension",
            float("inf"),
            float("inf"),
            None,
            0,
        )

    counter = OperatorCounter()
    apply_operator = FactorizedIntrinsicOperator(new_source, calibration, counter)
    incident = new_source.whitened_apply(
        (torch.sqrt(new_source.relation_metric)[:, None] * relation_port).T
    ).T
    whitened_position = torch.sqrt(new_source.quantity_metric)[:, None] * position
    whitened_velocity = torch.sqrt(new_source.quantity_metric)[:, None] * velocity
    seed = torch.cat((incident, whitened_position, whitened_velocity), dim=1)
    incident_residual = _residual(seed, basis)
    applied_basis = apply_operator(basis)
    invariance_residual = _residual(applied_basis, basis)
    if incident_residual > tolerance:
        return IncrementalDecision(
            "global-recompile-required",
            "new incident or initial-state port leaves the previous reachable carrier",
            incident_residual,
            invariance_residual,
            None,
            counter.applications,
        )
    if invariance_residual > tolerance:
        return IncrementalDecision(
            "global-recompile-required",
            "new intrinsic operator leaves the previous reachable carrier",
            incident_residual,
            invariance_residual,
            None,
            counter.applications,
        )

    carrier_operator = basis.T @ applied_basis
    carrier_operator = 0.5 * (carrier_operator + carrier_operator.T)
    carrier_seed = basis.T @ seed
    inner_reachable = block_krylov_basis(
        carrier_operator, carrier_seed, tolerance=tolerance
    )
    reachable = basis @ inner_reachable
    reachable_operator = inner_reachable.T @ carrier_operator @ inner_reachable
    reachable_operator = 0.5 * (reachable_operator + reachable_operator.T)
    widths = (incident.shape[1], position.shape[1], velocity.shape[1])
    restricted_seed = inner_reachable.T @ carrier_seed
    restricted_incident, restricted_position, restricted_velocity = torch.split(
        restricted_seed, widths, dim=1
    )
    whitened_observation = observation / torch.sqrt(new_source.quantity_metric)[None, :]
    reachable_observation = whitened_observation @ reachable
    observable = block_krylov_basis(
        reachable_operator, reachable_observation.T, tolerance=tolerance
    )
    lift = reachable @ observable
    reduced_operator = observable.T @ reachable_operator @ observable
    reduced_operator = 0.5 * (reduced_operator + reduced_operator.T)
    reduced_incident = observable.T @ restricted_incident
    reduced_position = observable.T @ restricted_position
    reduced_velocity = observable.T @ restricted_velocity
    reduced_observation = reachable_observation @ observable

    observable_residual = (
        _relative(
            observable @ reduced_operator,
            reachable_operator @ observable,
        )
        if observable.shape[1]
        else 0.0
    )
    markov = 0.0
    port_pairs = tuple(
        (full, reduced)
        for full, reduced in (
            (restricted_incident, reduced_incident),
            (restricted_position, reduced_position),
            (restricted_velocity, reduced_velocity),
        )
        if full.shape[1]
    )
    full_powers = [full for full, _ in port_pairs]
    reduced_powers = [small for _, small in port_pairs]
    scale = max(1.0, float(torch.linalg.matrix_norm(reachable_operator).item()))
    for _ in range(max(1, reachable.shape[1])):
        for full, small in zip(full_powers, reduced_powers):
            markov = max(
                markov,
                _relative(
                    reduced_observation @ small,
                    reachable_observation @ full,
                ),
            )
        full_powers = [reachable_operator @ value / scale for value in full_powers]
        reduced_powers = [reduced_operator @ value / scale for value in reduced_powers]

    eigenvalues, eigenmodes = torch.linalg.eigh(reduced_operator)
    eigenvalues = torch.clamp(eigenvalues, min=0)
    modal_incident = eigenmodes.T @ reduced_incident
    modal_observation = reduced_observation @ eigenmodes
    modal_position = eigenmodes.T @ reduced_position
    modal_velocity = eigenmodes.T @ reduced_velocity
    structural = _structural_digest(
        new_source,
        relation_port,
        observation,
        calibration,
        position,
        velocity,
    )
    certificate = ReductionCertificate(
        ambient_dimension=n,
        reachable_dimension=reachable.shape[1],
        minimal_dimension=observable.shape[1],
        incident_width=relation_port.shape[1],
        observation_width=observation.shape[0],
        initial_position_width=position.shape[1],
        initial_velocity_width=velocity.shape[1],
        tolerance=tolerance,
        symmetry_residual=_relative(reduced_operator, reduced_operator.T),
        positivity_floor=float(eigenvalues.min().item()) if eigenvalues.numel() else 0.0,
        reachable_invariance_residual=invariance_residual,
        observable_invariance_residual=observable_residual,
        maximum_markov_residual=markov,
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
        source_nonzeros=new_source.nnz,
        reachable_dimension=reachable.shape[1],
        minimal_dimension=observable.shape[1],
        factorized_operator_applications=counter.applications,
        d_applications=counter.applications + 1,
        adjoint_applications=counter.applications,
        maximum_block_width=basis.shape[1],
        source_bytes=_source_bytes(new_source),
        retained_execution_bytes=realization.execution_bytes,
        dense_relation_operator_materialized=False,
        dense_intrinsic_operator_materialized=False,
    )
    compilation = MatrixFreeCompilation(realization, accounting, reachable)
    return IncrementalDecision(
        "reduced-exact",
        "new ports remain in and new intrinsic action preserves the previous reachable carrier",
        incident_residual,
        invariance_residual,
        compilation,
        counter.applications,
    )


def update_or_recompile(
    previous: MatrixFreeCompilation,
    old_source: SparseRelationSource,
    new_source: SparseRelationSource,
    relation_port: Tensor,
    observation: Tensor,
    **kwargs,
) -> IncrementalDecision:
    decision = try_reduced_update(
        previous,
        old_source,
        new_source,
        relation_port,
        observation,
        **kwargs,
    )
    if decision.compilation is not None:
        return decision
    compilation = compile_matrix_free_relation_field(
        new_source,
        relation_port,
        observation,
        calibration=kwargs.get("calibration", 1.0),
        initial_position_port=kwargs.get("initial_position_port"),
        initial_velocity_port=kwargs.get("initial_velocity_port"),
        tolerance=kwargs.get("tolerance"),
    )
    return IncrementalDecision(
        "global-recompiled",
        decision.reason,
        decision.incident_outside_residual,
        decision.invariance_outside_residual,
        compilation,
        decision.factorized_operator_applications
        + compilation.accounting.factorized_operator_applications,
    )
