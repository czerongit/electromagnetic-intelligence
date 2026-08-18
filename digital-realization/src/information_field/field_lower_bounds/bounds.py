from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


def _rank(value: Tensor, tolerance: float) -> int:
    if value.ndim != 2:
        raise ValueError("rank requires a matrix")
    if value.numel() == 0:
        return 0
    return int(torch.count_nonzero(torch.linalg.svdvals(value) > tolerance).item())


def _active_columns(value: Tensor, tolerance: float) -> tuple[int, ...]:
    active = torch.any(torch.abs(value) > tolerance, dim=0)
    return tuple(torch.nonzero(active, as_tuple=False).flatten().cpu().tolist())


def _active_rows(value: Tensor, tolerance: float) -> tuple[int, ...]:
    active = torch.any(torch.abs(value) > tolerance, dim=1)
    return tuple(torch.nonzero(active, as_tuple=False).flatten().cpu().tolist())


@dataclass(frozen=True)
class LinearMapBounds:
    input_dimension: int
    output_dimension: int
    rank: int
    active_input_coordinates: tuple[int, ...]
    active_output_coordinates: tuple[int, ...]
    minimum_linear_factor_dimension: int
    minimum_worst_case_coordinate_reads: int
    minimum_variable_output_writes: int
    full_materialized_output_entries: int
    arithmetic_circuit_optimality_claimed: bool


def linear_map_bounds(
    response_map: Tensor, *, tolerance: float = 1e-10
) -> LinearMapBounds:
    if response_map.ndim != 2 or not response_map.is_floating_point():
        raise ValueError("response map must be a floating matrix")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    active_inputs = _active_columns(response_map, tolerance)
    active_outputs = _active_rows(response_map, tolerance)
    rank = _rank(response_map, tolerance)
    return LinearMapBounds(
        response_map.shape[1],
        response_map.shape[0],
        rank,
        active_inputs,
        active_outputs,
        rank,
        len(active_inputs),
        len(active_outputs),
        response_map.shape[0],
        False,
    )


@dataclass(frozen=True)
class FactorizationCertificate:
    exact: bool
    relative_residual: float
    factor_dimension: int
    response_rank: int
    meets_rank_lower_bound: bool


def certify_linear_factorization(
    response_map: Tensor,
    output_factor: Tensor,
    input_factor: Tensor,
    *,
    tolerance: float = 1e-10,
) -> FactorizationCertificate:
    if response_map.ndim != 2 or output_factor.ndim != 2 or input_factor.ndim != 2:
        raise ValueError("factorization data must be matrices")
    if output_factor.shape[0] != response_map.shape[0]:
        raise ValueError("output factor has the wrong output dimension")
    if input_factor.shape[1] != response_map.shape[1]:
        raise ValueError("input factor has the wrong input dimension")
    if output_factor.shape[1] != input_factor.shape[0]:
        raise ValueError("factor dimensions do not agree")
    reconstructed = output_factor @ input_factor
    difference = float(torch.linalg.matrix_norm(reconstructed - response_map).item())
    scale = max(1.0, float(torch.linalg.matrix_norm(response_map).item()))
    residual = difference / scale
    rank = _rank(response_map, tolerance)
    width = output_factor.shape[1]
    return FactorizationCertificate(
        residual <= tolerance,
        residual,
        width,
        rank,
        width >= rank,
    )


@dataclass(frozen=True)
class BasisLookupBound:
    column: int
    variable_output_entries: int
    dense_materialized_output_entries: int
    arithmetic_operation_lower_bound: int
    applies_only_to_materialized_output: bool


def basis_lookup_bound(
    response_map: Tensor, column: int, *, tolerance: float = 1e-10
) -> BasisLookupBound:
    if response_map.ndim != 2:
        raise ValueError("response map must be a matrix")
    if column < 0 or column >= response_map.shape[1]:
        raise ValueError("basis column is outside the input carrier")
    variable = int(torch.count_nonzero(torch.abs(response_map[:, column]) > tolerance).item())
    return BasisLookupBound(column, variable, response_map.shape[0], 0, True)


@dataclass(frozen=True)
class SpectralDegreeBound:
    distinct_visible_eigenvalues: tuple[float, ...]
    residue_ranks: tuple[int, ...]
    minimum_second_order_carrier_dimension: int
    minimum_continuous_first_order_state_dimension: int
    executor_class: str
    sampled_grid_degree_claimed: bool
    arithmetic_circuit_optimality_claimed: bool


def continuous_spectral_degree_bound(
    eigenvalues: Tensor,
    residues: Tensor,
    *,
    tolerance: float = 1e-10,
) -> SpectralDegreeBound:
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a vector")
    if residues.ndim != 3 or residues.shape[0] != eigenvalues.numel():
        raise ValueError("residues must have shape frequency by output by input")
    if bool(torch.any(eigenvalues < -tolerance)):
        raise ValueError("intrinsic eigenvalues must be nonnegative")
    order = torch.argsort(eigenvalues)
    groups: list[tuple[float, Tensor]] = []
    for index in order.tolist():
        value = float(eigenvalues[index].item())
        residue = residues[index]
        if groups and abs(value - groups[-1][0]) <= tolerance:
            previous_value, previous_residue = groups[-1]
            groups[-1] = (previous_value, previous_residue + residue)
        else:
            groups.append((value, residue.clone()))
    visible_values = []
    ranks = []
    for value, residue in groups:
        rank = _rank(residue, tolerance)
        if rank:
            visible_values.append(value)
            ranks.append(rank)
    second_order = sum(ranks)
    return SpectralDegreeBound(
        tuple(visible_values),
        tuple(ranks),
        second_order,
        2 * second_order,
        "finite-dimensional continuous-time LTI force-to-position realizations",
        False,
        False,
    )


@dataclass(frozen=True)
class DiscreteHankelBound:
    supplied_state_dimension: int
    block_rows: int
    block_columns: int
    hankel_rank: int
    minimum_discrete_lti_state_dimension: int
    minimality_certified: bool
    arithmetic_circuit_optimality_claimed: bool


def discrete_hankel_degree_bound(
    state_operator: Tensor,
    incident_port: Tensor,
    observation: Tensor,
    *,
    block_rows: int | None = None,
    block_columns: int | None = None,
    tolerance: float = 1e-10,
) -> DiscreteHankelBound:
    if state_operator.ndim != 2 or state_operator.shape[0] != state_operator.shape[1]:
        raise ValueError("state operator must be square")
    n = state_operator.shape[0]
    if incident_port.ndim != 2 or incident_port.shape[0] != n:
        raise ValueError("incident port has the wrong state dimension")
    if observation.ndim != 2 or observation.shape[1] != n:
        raise ValueError("observation has the wrong state dimension")
    rows = n if block_rows is None else int(block_rows)
    columns = n if block_columns is None else int(block_columns)
    if rows < 1 or columns < 1 or tolerance <= 0:
        raise ValueError("Hankel dimensions and tolerance must be positive")
    powers = [incident_port]
    for _ in range(rows + columns - 2):
        powers.append(state_operator @ powers[-1])
    blocks = [
        torch.cat(
            [observation @ powers[row + column] for column in range(columns)],
            dim=1,
        )
        for row in range(rows)
    ]
    hankel = torch.cat(blocks, dim=0)
    rank = _rank(hankel, tolerance)
    certified = rows >= n and columns >= n
    return DiscreteHankelBound(n, rows, columns, rank, rank, certified, False)


@dataclass(frozen=True)
class StaticRelationPathBounds:
    response_map: Tensor
    bounds: LinearMapBounds


def static_relation_path_bounds(
    relation_operator: Tensor,
    observation: Tensor,
    *,
    tolerance: float = 1e-10,
) -> StaticRelationPathBounds:
    if relation_operator.ndim != 2:
        raise ValueError("relation operator must be a matrix")
    if observation.ndim != 2 or observation.shape[1] != relation_operator.shape[0]:
        raise ValueError("observation has the wrong quantity dimension")
    response = observation @ relation_operator
    return StaticRelationPathBounds(response, linear_map_bounds(response, tolerance=tolerance))
