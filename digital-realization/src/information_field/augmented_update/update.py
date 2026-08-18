from __future__ import annotations

from dataclasses import dataclass

import torch

from information_field.incremental_field import try_reduced_update
from information_field.matrix_free_field import MatrixFreeCompilation, compile_matrix_free_relation_field
from information_field.matrix_free_field.compiler import (
    FactorizedIntrinsicOperator,
    OperatorCounter,
    matrix_free_block_krylov,
)
from information_field.quotient_response import SparseRelationSource


Tensor = torch.Tensor


@dataclass(frozen=True)
class AugmentedUpdateDecision:
    status: str
    reason: str
    old_reachable_dimension: int
    augmented_dimension: int
    final_reachable_dimension: int | None
    added_dimensions: int
    augmentation_operator_applications: int
    total_operator_applications: int
    compilation: MatrixFreeCompilation | None


def _global_required(reason: str, old_dimension: int) -> AugmentedUpdateDecision:
    return AugmentedUpdateDecision(
        "global-recompile-required",
        reason,
        old_dimension,
        old_dimension,
        None,
        0,
        0,
        0,
        None,
    )


def try_augmented_update(
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
    maximum_added_dimensions: int | None = None,
) -> AugmentedUpdateDecision:
    old_dimension = previous.reachable_basis.shape[1]
    if (old_source.quantity_dim, old_source.relation_dim) != (
        new_source.quantity_dim,
        new_source.relation_dim,
    ):
        return _global_required("carrier dimension changed", old_dimension)
    if not (
        torch.equal(old_source.quantity_metric.cpu(), new_source.quantity_metric.cpu())
        and torch.equal(old_source.relation_metric.cpu(), new_source.relation_metric.cpu())
    ):
        return _global_required("carrier metric changed", old_dimension)
    if maximum_added_dimensions is not None and maximum_added_dimensions < 0:
        raise ValueError("maximum added dimensions must be nonnegative")
    if calibration <= 0:
        raise ValueError("calibration must be positive")
    n = new_source.quantity_dim
    relation_port = relation_port.to(new_source.device, new_source.dtype)
    observation = observation.to(new_source.device, new_source.dtype)
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

    incident = new_source.whitened_apply(
        (torch.sqrt(new_source.relation_metric)[:, None] * relation_port).T
    ).T
    whitened_position = torch.sqrt(new_source.quantity_metric)[:, None] * position
    whitened_velocity = torch.sqrt(new_source.quantity_metric)[:, None] * velocity
    seed = torch.cat((incident, whitened_position, whitened_velocity), dim=1)
    old_basis = previous.reachable_basis.to(new_source.device, new_source.dtype)
    counter = OperatorCounter()
    apply_operator = FactorizedIntrinsicOperator(new_source, calibration, counter)
    augmented = matrix_free_block_krylov(
        apply_operator,
        torch.cat((old_basis, seed), dim=1),
        ambient_dimension=n,
        tolerance=tolerance,
    )
    added = max(0, augmented.shape[1] - old_dimension)
    if maximum_added_dimensions is not None and added > maximum_added_dimensions:
        return AugmentedUpdateDecision(
            "global-recompile-required",
            "augmented carrier exceeds the declared update budget",
            old_dimension,
            augmented.shape[1],
            None,
            added,
            counter.applications,
            counter.applications,
            None,
        )

    carrier = MatrixFreeCompilation(
        previous.realization,
        previous.accounting,
        augmented,
    )
    reduced = try_reduced_update(
        carrier,
        new_source,
        new_source,
        relation_port,
        observation,
        calibration=calibration,
        initial_position_port=position,
        initial_velocity_port=velocity,
        tolerance=tolerance,
    )
    if reduced.compilation is None:
        return AugmentedUpdateDecision(
            "global-recompile-required",
            "augmented carrier failed its reduced invariance certificate",
            old_dimension,
            augmented.shape[1],
            None,
            added,
            counter.applications,
            counter.applications + reduced.factorized_operator_applications,
            None,
        )
    return AugmentedUpdateDecision(
        "augmented-exact",
        "old carrier and new port seeds close to an invariant augmented carrier",
        old_dimension,
        augmented.shape[1],
        reduced.compilation.accounting.reachable_dimension,
        added,
        counter.applications,
        counter.applications + reduced.factorized_operator_applications,
        reduced.compilation,
    )


def update_with_augmentation_or_recompile(
    previous: MatrixFreeCompilation,
    old_source: SparseRelationSource,
    new_source: SparseRelationSource,
    relation_port: Tensor,
    observation: Tensor,
    **kwargs,
) -> AugmentedUpdateDecision:
    decision = try_augmented_update(
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
    return AugmentedUpdateDecision(
        "global-recompiled",
        decision.reason,
        decision.old_reachable_dimension,
        decision.augmented_dimension,
        compilation.accounting.reachable_dimension,
        decision.added_dimensions,
        decision.augmentation_operator_applications,
        decision.total_operator_applications
        + compilation.accounting.factorized_operator_applications,
        compilation,
    )
