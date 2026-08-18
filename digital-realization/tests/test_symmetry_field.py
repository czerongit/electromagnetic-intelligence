import math

import torch

from information_field.causal_minimal import exact_second_order_state
from information_field.matrix_free_field import compile_matrix_free_relation_field
from information_field.quotient_response import SparseRelationSource
from information_field.symmetry_field import (
    validate_declared_linear_gauge,
    validate_noether_generator,
    classify_null_sectors,
    compile_involutive_symmetry,
    noether_charge,
)


DTYPE = torch.float64


def paired_diagonal(pair_count: int):
    diagonal = torch.repeat_interleave(
        torch.linspace(0.7, 1.7, pair_count, dtype=DTYPE), 2
    )
    source = SparseRelationSource.from_dense(torch.diag(diagonal))
    permutation = torch.arange(2 * pair_count, dtype=torch.int64)
    permutation[0::2] += 1
    permutation[1::2] -= 1
    return source, permutation


def parity_ports(pair_count: int):
    n = 2 * pair_count
    symmetric = torch.zeros(n, dtype=DTYPE)
    antisymmetric = torch.zeros(n, dtype=DTYPE)
    symmetric[0::2] = 1.0
    symmetric[1::2] = 1.0
    antisymmetric[0::2] = 1.0
    antisymmetric[1::2] = -1.0
    port = torch.stack((symmetric, antisymmetric), dim=1)
    observation = torch.stack((symmetric, antisymmetric), dim=0)
    return port, observation


def test_certified_source_symmetry_preserves_complete_response():
    source, permutation = paired_diagonal(4)
    port, observation = parity_ports(4)
    split = compile_involutive_symmetry(
        source, port, observation, permutation, permutation
    )
    full = compile_matrix_free_relation_field(source, port, observation)
    incident = torch.tensor([0.3, -0.7], dtype=DTYPE)
    actual = split.respond_constant(incident, time=0.83, mass=1.2)
    expected = full.realization.respond_constant(incident, time=0.83, mass=1.2)
    assert split.certificate.used_symmetry
    assert split.accounting.sector_count == 2
    assert torch.allclose(actual, expected, atol=1e-10)


def test_operator_symmetry_without_port_symmetry_falls_back():
    source, permutation = paired_diagonal(2)
    port = torch.eye(4, dtype=DTYPE)[:, :1]
    observation = torch.eye(4, dtype=DTYPE)[:1]
    result = compile_involutive_symmetry(
        source, port, observation, permutation, permutation
    )
    assert not result.certificate.used_symmetry
    assert result.fallback is not None
    assert "incident port" in result.certificate.fallback_reason
    assert "readout" in result.certificate.fallback_reason


def test_declared_coordinate_symmetry_must_intertwine_the_source():
    source = SparseRelationSource.from_dense(
        torch.diag(torch.tensor([1.0, 2.0], dtype=DTYPE))
    )
    permutation = torch.tensor([1, 0], dtype=torch.int64)
    port = torch.eye(2, dtype=DTYPE)
    observation = torch.eye(2, dtype=DTYPE)
    result = compile_involutive_symmetry(
        source, port, observation, permutation, permutation
    )
    assert not result.certificate.used_symmetry
    assert "source automorphism" in result.certificate.fallback_reason


def test_initial_state_ports_are_part_of_the_symmetry_contract():
    source, permutation = paired_diagonal(2)
    port, observation = parity_ports(2)
    position = torch.eye(4, dtype=DTYPE)[:, :1]
    result = compile_involutive_symmetry(
        source,
        port,
        observation,
        permutation,
        permutation,
        initial_position_port=position,
    )
    assert not result.certificate.used_symmetry
    assert "initial-position port" in result.certificate.fallback_reason


def test_silent_and_redundant_null_sectors_are_distinct():
    operator = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    source = SparseRelationSource.from_dense(operator)
    sectors = classify_null_sectors(source)
    assert sectors.operator_rank == 2
    assert sectors.silent_quantity_dimension == 1
    assert sectors.redundant_relation_dimension == 1
    relation_null = torch.tensor([-1.0, -1.0, 1.0], dtype=DTYPE)
    assert torch.allclose(source.whitened_apply(relation_null), torch.zeros(3, dtype=DTYPE))


def test_silent_direction_is_not_gauge_without_a_declared_action():
    source = SparseRelationSource.from_dense(
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=DTYPE)
    )
    port = torch.eye(2, dtype=DTYPE)
    observation = torch.tensor([[1.0, 0.0]], dtype=DTYPE)
    undeclared = validate_declared_linear_gauge(source, None, port, observation)
    declared = validate_declared_linear_gauge(
        source,
        torch.tensor([[0.0], [1.0]], dtype=DTYPE),
        port,
        observation,
    )
    wrong = validate_declared_linear_gauge(
        source,
        torch.tensor([[1.0], [0.0]], dtype=DTYPE),
        port,
        observation,
    )
    assert not undeclared.declared and not undeclared.can_quotient
    assert declared.can_quotient
    assert not wrong.can_quotient
    assert wrong.silent_residual > 0


def test_conserved_zero_mode_can_remain_observable():
    source = SparseRelationSource.from_dense(
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=DTYPE)
    )
    port = torch.tensor([[1.0], [0.0]], dtype=DTYPE)
    observation = torch.tensor([[0.0, 1.0]], dtype=DTYPE)
    velocity_port = torch.tensor([[0.0], [1.0]], dtype=DTYPE)
    compiled = compile_matrix_free_relation_field(
        source,
        port,
        observation,
        initial_velocity_port=velocity_port,
    )
    response = compiled.realization.respond_constant(
        torch.tensor([0.0], dtype=DTYPE),
        time=1.5,
        initial_velocity=torch.tensor([2.0], dtype=DTYPE),
    )
    assert compiled.accounting.minimal_dimension == 1
    assert torch.allclose(response, torch.tensor([3.0], dtype=DTYPE))


def test_valid_noether_generator_has_conserved_charge():
    source = SparseRelationSource.from_dense(torch.eye(2, dtype=DTYPE))
    generator = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=DTYPE)
    validation = validate_noether_generator(source, generator)
    position = torch.tensor([1.0, 0.0], dtype=DTYPE)
    velocity = torch.tensor([0.0, 1.0], dtype=DTYPE)
    initial_charge = noether_charge(position, velocity, generator)
    evolved_position, evolved_velocity = exact_second_order_state(
        torch.eye(2, dtype=DTYPE),
        torch.zeros(2, dtype=DTYPE),
        position,
        velocity,
        time=0.73,
        mass=1.0,
        tolerance=1e-12,
    )
    final_charge = noether_charge(evolved_position, evolved_velocity, generator)
    assert validation.valid_generator
    assert torch.allclose(final_charge, initial_charge, atol=1e-12)


def test_degenerate_eigenvalue_does_not_authorize_quotienting():
    source, permutation = paired_diagonal(1)
    port = torch.eye(2, dtype=DTYPE)
    observation = torch.eye(2, dtype=DTYPE)
    result = compile_involutive_symmetry(
        source, port, observation, permutation, permutation
    )
    assert result.certificate.used_symmetry
    assert result.accounting.total_minimal_dimension == 2
    assert tuple(sorted(result.accounting.sector_minimal_dimensions)) == (1, 1)


def test_compilation_invalidates_after_source_or_symmetry_change():
    source, permutation = paired_diagonal(2)
    port, observation = parity_ports(2)
    result = compile_involutive_symmetry(
        source, port, observation, permutation, permutation
    )
    changed = SparseRelationSource.from_dense(
        source.dense_operator() + torch.diag(torch.tensor([0.1, 0.1, 0.0, 0.0], dtype=DTYPE))
    )
    identity = torch.arange(4, dtype=torch.int64)
    signs = torch.ones(4, dtype=DTYPE)
    assert result.is_valid_for(
        source, port, observation, permutation, permutation, signs, signs
    )
    assert not result.is_valid_for(
        changed, port, observation, permutation, permutation, signs, signs
    )
    assert not result.is_valid_for(
        source, port, observation, identity, identity, signs, signs
    )
