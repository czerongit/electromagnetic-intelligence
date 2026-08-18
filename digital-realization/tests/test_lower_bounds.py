import math

import torch

from information_field.field_lower_bounds import (
    basis_lookup_bound,
    certify_linear_factorization,
    continuous_spectral_degree_bound,
    discrete_hankel_degree_bound,
    linear_map_bounds,
    static_relation_path_bounds,
)


DTYPE = torch.float64


def test_rank_is_the_exact_linear_factor_carrier_bound():
    response = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 2.0, 2.0], [1.0, 2.0, 3.0]],
        dtype=DTYPE,
    )
    bounds = linear_map_bounds(response)
    left, singular, right = torch.linalg.svd(response, full_matrices=False)
    rank = bounds.rank
    output_factor = left[:, :rank] * singular[:rank]
    input_factor = right[:rank]
    certificate = certify_linear_factorization(
        response, output_factor, input_factor
    )
    assert rank == 2
    assert certificate.exact
    assert certificate.factor_dimension == certificate.response_rank


def test_narrower_linear_factorization_cannot_match_higher_rank_map():
    response = torch.eye(3, dtype=DTYPE)
    output_factor = torch.ones((3, 2), dtype=DTYPE)
    input_factor = torch.ones((2, 3), dtype=DTYPE)
    certificate = certify_linear_factorization(
        response, output_factor, input_factor
    )
    assert not certificate.exact
    assert not certificate.meets_rank_lower_bound


def test_active_coordinate_reads_follow_adversarial_input_changes():
    response = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 3.0, 0.0]], dtype=DTYPE
    )
    bounds = linear_map_bounds(response)
    assert bounds.active_input_coordinates == (0, 2)
    assert bounds.minimum_worst_case_coordinate_reads == 2
    for coordinate in bounds.active_input_coordinates:
        difference = response @ torch.eye(4, dtype=DTYPE)[:, coordinate]
        assert torch.count_nonzero(difference)


def test_explicit_output_and_basis_lookup_contracts_are_separate():
    response = torch.tensor(
        [[2.0, 0.0], [0.0, 0.0], [3.0, 4.0]], dtype=DTYPE
    )
    bounds = linear_map_bounds(response)
    lookup = basis_lookup_bound(response, 0)
    assert bounds.minimum_variable_output_writes == 2
    assert bounds.full_materialized_output_entries == 3
    assert lookup.variable_output_entries == 2
    assert lookup.dense_materialized_output_entries == 3
    assert lookup.arithmetic_operation_lower_bound == 0


def test_one_silent_evaluation_does_not_change_map_lower_bounds():
    response = 1000.0 * torch.eye(2, dtype=DTYPE)
    value = response @ torch.zeros(2, dtype=DTYPE)
    bounds = linear_map_bounds(response)
    assert torch.equal(value, torch.zeros(2, dtype=DTYPE))
    assert bounds.rank == 2
    assert bounds.minimum_worst_case_coordinate_reads == 2


def test_static_relation_path_bound_applies_to_compiled_od():
    relation = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 2.0]],
        dtype=DTYPE,
    )
    observation = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE)
    result = static_relation_path_bounds(relation, observation)
    assert torch.equal(result.response_map, observation @ relation)
    assert result.bounds.minimum_linear_factor_dimension == 2
    assert not result.bounds.arithmetic_circuit_optimality_claimed


def test_continuous_residue_rank_gives_first_order_degree():
    eigenvalues = torch.tensor([1.0, 4.0, 9.0], dtype=DTYPE)
    residues = torch.stack(
        (
            torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=DTYPE),
            torch.eye(2, dtype=DTYPE),
            torch.tensor([[0.0, 0.0], [0.0, 2.0]], dtype=DTYPE),
        )
    )
    bound = continuous_spectral_degree_bound(eigenvalues, residues)
    assert bound.residue_ranks == (1, 2, 1)
    assert bound.minimum_second_order_carrier_dimension == 4
    assert bound.minimum_continuous_first_order_state_dimension == 8
    assert not bound.sampled_grid_degree_claimed


def test_repeated_spectral_labels_are_merged_before_rank():
    eigenvalues = torch.tensor([2.0, 2.0], dtype=DTYPE)
    residues = torch.tensor([[[1.0]], [[-1.0]]], dtype=DTYPE)
    bound = continuous_spectral_degree_bound(eigenvalues, residues)
    assert bound.distinct_visible_eigenvalues == ()
    assert bound.minimum_continuous_first_order_state_dimension == 0


def test_sampled_hankel_rank_detects_frequency_aliasing():
    theta = 0.7
    rotation = torch.tensor(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]],
        dtype=DTYPE,
    )
    state = torch.block_diag(rotation, rotation)
    incident = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=DTYPE)
    observation = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=DTYPE)
    sampled = discrete_hankel_degree_bound(state, incident, observation)
    continuous = continuous_spectral_degree_bound(
        torch.tensor([theta**2, (theta + 2.0 * math.pi) ** 2], dtype=DTYPE),
        torch.ones((2, 1, 1), dtype=DTYPE),
    )
    assert continuous.minimum_continuous_first_order_state_dimension == 4
    assert sampled.minimality_certified
    assert sampled.minimum_discrete_lti_state_dimension == 2


def test_hankel_rank_recovers_a_nonaliased_discrete_degree():
    state = torch.diag(torch.tensor([0.2, 0.7], dtype=DTYPE))
    incident = torch.ones((2, 1), dtype=DTYPE)
    observation = torch.ones((1, 2), dtype=DTYPE)
    bound = discrete_hankel_degree_bound(state, incident, observation)
    assert bound.minimality_certified
    assert bound.minimum_discrete_lti_state_dimension == 2
    assert not bound.arithmetic_circuit_optimality_claimed
