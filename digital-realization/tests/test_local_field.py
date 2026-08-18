import torch

from information_field.causal_minimal import full_constant_response
from information_field.local_field import (
    compile_component_restricted,
    jet_causal_nodes,
    quantity_components,
)
from information_field.matrix_free_field import compile_matrix_free_relation_field
from information_field.quotient_response import SparseRelationSource


DTYPE = torch.float64


def incidence(edges, nodes):
    matrix = torch.zeros((nodes, len(edges)), dtype=DTYPE)
    for column, (left, right) in enumerate(edges):
        matrix[left, column] = 1.0
        matrix[right, column] = -1.0
    return SparseRelationSource.from_dense(matrix)


def test_components_follow_shared_relation_incidence():
    field = incidence(((0, 1), (1, 2), (3, 4)), 6)
    labels = quantity_components(field).tolist()
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert labels[5] not in {labels[0], labels[3]}


def test_component_restriction_matches_global_causal_response():
    field = incidence(((0, 1), (1, 2), (3, 4), (4, 5)), 6)
    port = torch.tensor(
        [[1.0], [0.5], [0.0], [0.0]], dtype=DTYPE
    )
    observation = torch.tensor(
        [[1.0, -0.5, 0.25, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
        dtype=DTYPE,
    )
    local = compile_component_restricted(field, port, observation)
    global_compiled = compile_matrix_free_relation_field(field, port, observation)
    incident_value = torch.tensor([0.7], dtype=DTYPE)
    actual = local.respond_constant(incident_value, time=0.6, mass=1.2)
    expected = global_compiled.realization.respond_prepared_zero_past_constant(
        incident_value, time=0.6, mass=1.2
    )
    assert torch.allclose(actual, expected, atol=1e-10)
    assert local.accounting.selected_components == 1
    assert local.accounting.selected_quantities == 3
    assert actual[1] == 0


def test_disjoint_incident_and_observation_compile_exact_zero():
    field = incidence(((0, 1), (2, 3)), 4)
    port = torch.tensor([[1.0], [0.0]], dtype=DTYPE)
    observation = torch.tensor([[0.0, 0.0, 1.0, -1.0]], dtype=DTYPE)
    local = compile_component_restricted(field, port, observation)
    assert local.accounting.zero_response
    assert local.compilation is None
    assert torch.equal(
        local.respond_constant(torch.tensor([0.5], dtype=DTYPE), time=1.0),
        torch.zeros(1, dtype=DTYPE),
    )


def test_initial_state_support_participates_in_component_selection():
    field = incidence(((0, 1), (2, 3)), 4)
    port = torch.zeros((2, 1), dtype=DTYPE)
    observation = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=DTYPE)
    position = torch.tensor([[0.0], [0.0], [1.0], [0.0]], dtype=DTYPE)
    local = compile_component_restricted(
        field, port, observation, initial_position_port=position
    )
    assert not local.accounting.zero_response
    assert local.accounting.selected_quantities == 2
    result = local.compilation.realization.respond_constant(
        torch.tensor([0.0], dtype=DTYPE),
        time=0.4,
        initial_position=torch.tensor([0.7], dtype=DTYPE),
    )
    assert result[0] != 0


def test_finite_jet_cone_is_exact_on_a_chain():
    field = incidence(((0, 1), (1, 2), (2, 3), (3, 4)), 5)
    assert jet_causal_nodes(
        field, (0,), (4,), maximum_order=3
    ).numel() == 0
    assert torch.equal(
        jet_causal_nodes(field, (0,), (4,), maximum_order=4),
        torch.arange(5, dtype=torch.int64),
    )
    d = field.whitened_dense()
    operator = d @ d.T
    incident = torch.eye(5, dtype=DTYPE)[:, 0:1]
    observation = torch.eye(5, dtype=DTYPE)[4:5]
    for order in range(4):
        assert torch.count_nonzero(
            observation @ torch.linalg.matrix_power(operator, order) @ incident
        ) == 0
    assert torch.count_nonzero(
        observation @ torch.linalg.matrix_power(operator, 4) @ incident
    ) == 1


def test_graph_wave_has_no_exact_finite_time_hop_cone():
    field = incidence(((0, 1), (1, 2), (2, 3), (3, 4)), 5)
    d = field.whitened_dense()
    operator = d @ d.T
    incident = torch.eye(5, dtype=DTYPE)[:, 0:1]
    observation = torch.eye(5, dtype=DTYPE)[4:5]
    response = full_constant_response(
        operator,
        incident,
        observation,
        torch.tensor([1.0], dtype=DTYPE),
        time=0.5,
    )
    assert abs(float(response[0])) > 0


def test_component_result_invalidates_after_source_change():
    field = incidence(((0, 1), (2, 3)), 4)
    port = torch.tensor([[1.0], [0.0]], dtype=DTYPE)
    observation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    local = compile_component_restricted(field, port, observation)
    changed = incidence(((0, 1), (1, 2), (2, 3)), 4)
    assert local.source_digest != changed.digest
