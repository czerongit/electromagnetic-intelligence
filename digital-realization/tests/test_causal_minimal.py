import pytest
import torch

from information_field.causal_minimal import (
    compile_minimal_realization,
    compile_relation_field,
    full_constant_response,
)
from information_field.quotient_response import SparseRelationSource


DTYPE = torch.float64


def diagonal(values):
    return torch.diag(torch.tensor(values, dtype=DTYPE))


def test_exact_transfer_jets_and_constant_response_survive_reduction():
    operator = diagonal([1.0, 2.0, 4.0, 8.0, 16.0])
    incident = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, -1.0]],
        dtype=DTYPE,
    )
    observation = torch.tensor(
        [[1.0, -0.5, 0.0, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    compiled = compile_minimal_realization(operator, incident, observation)
    assert compiled.certificate.reachable_dimension == 3
    assert compiled.certificate.minimal_dimension == 2
    for order in range(operator.shape[0]):
        expected = observation @ torch.linalg.matrix_power(operator, order) @ incident
        assert torch.allclose(compiled.transfer_jet(order), expected, atol=1e-10)
    query = torch.tensor([0.3, -0.7], dtype=DTYPE)
    expected = full_constant_response(
        operator, incident, observation, query, time=0.73, mass=1.4
    )
    actual = compiled.respond_constant(query, time=0.73, mass=1.4)
    assert torch.allclose(actual, expected, atol=1e-10)
    assert compiled.certificate.maximum_markov_residual < 1e-10


def test_one_silent_probe_does_not_remove_a_port_visible_mode():
    operator = diagonal([1.0, 3.0])
    incident = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    observation = torch.eye(2, dtype=DTYPE)
    compiled = compile_minimal_realization(operator, incident, observation)
    silent_on_second_mode = torch.tensor([1.0, 0.0], dtype=DTYPE)
    assert compiled.certificate.minimal_dimension == 2
    assert torch.allclose(
        compiled.respond_constant(silent_on_second_mode, time=0.5),
        full_constant_response(
            operator, incident, observation, silent_on_second_mode, time=0.5
        ),
    )


def test_widening_either_declared_port_can_restore_a_discarded_sector():
    operator = diagonal([1.0, 2.0, 3.0])
    base_incident = torch.tensor([[1.0], [0.0], [1.0]], dtype=DTYPE)
    base_observation = torch.tensor([[1.0, 1.0, 0.0]], dtype=DTYPE)
    base = compile_minimal_realization(operator, base_incident, base_observation)
    assert base.certificate.minimal_dimension == 1

    widened_incident = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=DTYPE
    )
    after_incident = compile_minimal_realization(
        operator, widened_incident, base_observation
    )
    assert after_incident.certificate.minimal_dimension == 2

    widened_observation = torch.tensor(
        [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=DTYPE
    )
    after_observation = compile_minimal_realization(
        operator, base_incident, widened_observation
    )
    assert after_observation.certificate.minimal_dimension == 2


def test_initial_data_must_enter_through_a_declared_port():
    operator = diagonal([1.0, 2.0, 5.0])
    incident = torch.tensor([[1.0], [0.0], [0.0]], dtype=DTYPE)
    observation = torch.tensor([[1.0, 0.0, 1.0]], dtype=DTYPE)
    without_initial_port = compile_minimal_realization(operator, incident, observation)
    assert without_initial_port.certificate.minimal_dimension == 1
    with pytest.raises(ValueError, match="wrong port dimension"):
        without_initial_port.respond_constant(
            torch.tensor([0.0], dtype=DTYPE),
            time=0.4,
            initial_position=torch.tensor([1.0], dtype=DTYPE),
        )

    initial_port = torch.tensor([[0.0], [0.0], [1.0]], dtype=DTYPE)
    compiled = compile_minimal_realization(
        operator,
        incident,
        observation,
        initial_position_port=initial_port,
    )
    assert compiled.certificate.minimal_dimension == 2
    query = torch.tensor([0.0], dtype=DTYPE)
    coordinate = torch.tensor([0.7], dtype=DTYPE)
    actual = compiled.respond_constant(
        query, time=0.4, initial_position=coordinate
    )
    expected = full_constant_response(
        operator,
        incident,
        observation,
        query,
        time=0.4,
        initial_position_port=initial_port,
        initial_position=coordinate,
    )
    assert torch.allclose(actual, expected, atol=1e-10)


def test_every_structural_or_port_change_invalidates_the_certificate():
    operator = diagonal([1.0, 2.0])
    incident = torch.tensor([[1.0], [0.0]], dtype=DTYPE)
    observation = torch.tensor([[1.0, 0.0]], dtype=DTYPE)
    compiled = compile_minimal_realization(operator, incident, observation)
    assert compiled.is_valid_for(operator, incident, observation)
    assert not compiled.is_valid_for(operator * 2.0, incident, observation)
    assert not compiled.is_valid_for(operator, incident.flip(0), observation)
    assert not compiled.is_valid_for(operator, incident, observation.flip(1))
    with pytest.raises(ValueError, match="recompile"):
        compiled.assert_valid_for(operator * 2.0, incident, observation)


def test_relation_field_bridge_preserves_sparse_dynamics_and_metrics():
    dense = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 2.0, -0.25], [0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    source = SparseRelationSource.from_dense(
        dense,
        quantity_metric=torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE),
        relation_metric=torch.tensor([1.5, 0.75, 2.0], dtype=DTYPE),
    )
    relation_port = torch.eye(3, dtype=DTYPE)
    observation = torch.tensor([[1.0, -0.5, 0.0]], dtype=DTYPE)
    compiled = compile_relation_field(
        source, relation_port, observation, calibration=0.8
    )
    assert compiled.certificate.ambient_dimension == 3
    assert compiled.certificate.minimal_dimension <= 2
    relation_incident = torch.tensor([0.4, -0.2, 0.8], dtype=DTYPE)
    d = source.whitened_dense()
    operator = 0.8 * d @ d.T
    incident_port = d @ (
        torch.sqrt(source.relation_metric)[:, None] * relation_port
    )
    whitened_observation = observation / torch.sqrt(source.quantity_metric)[None, :]
    expected = full_constant_response(
        operator,
        incident_port,
        whitened_observation,
        relation_incident,
        time=0.6,
    )
    actual = compiled.respond_constant(relation_incident, time=0.6)
    assert torch.allclose(actual, expected, atol=1e-10)
    assert compiled.is_valid_for_relation_field(
        source, relation_port, observation, calibration=0.8
    )
    changed_metric_source = SparseRelationSource.from_dense(
        dense,
        quantity_metric=torch.tensor([2.0, 3.0, 5.0], dtype=DTYPE),
        relation_metric=torch.tensor([1.5, 0.75, 2.0], dtype=DTYPE),
    )
    assert not compiled.is_valid_for_relation_field(
        changed_metric_source, relation_port, observation, calibration=0.8
    )
    assert not compiled.is_valid_for_relation_field(
        source, relation_port, observation, calibration=0.9
    )


def test_nonsymmetric_or_indefinite_law_is_rejected():
    incident = torch.ones((2, 1), dtype=DTYPE)
    observation = torch.ones((1, 2), dtype=DTYPE)
    with pytest.raises(ValueError, match="self-adjoint"):
        compile_minimal_realization(
            torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=DTYPE),
            incident,
            observation,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        compile_minimal_realization(diagonal([1.0, -1.0]), incident, observation)


def test_zero_observation_compiles_the_zero_dimensional_response():
    operator = diagonal([1.0, 2.0, 3.0])
    incident = torch.tensor([[1.0], [1.0], [0.0]], dtype=DTYPE)
    observation = torch.zeros((1, 3), dtype=DTYPE)
    compiled = compile_minimal_realization(operator, incident, observation)
    assert compiled.certificate.reachable_dimension == 2
    assert compiled.certificate.minimal_dimension == 0
    actual = compiled.respond_prepared_zero_past_constant(
        torch.tensor([0.8], dtype=DTYPE), time=0.5
    )
    assert torch.equal(actual, torch.zeros(1, dtype=DTYPE))


def test_compiled_execution_can_lower_dtype_without_changing_dimension():
    operator = diagonal([1.0, 2.0, 3.0])
    incident = torch.eye(3, dtype=DTYPE)
    observation = torch.eye(3, dtype=DTYPE)
    compiled = compile_minimal_realization(operator, incident, observation)
    lowered = compiled.to("cpu", torch.float32)
    assert lowered.operator.dtype == torch.float32
    assert lowered.state_dimension == compiled.state_dimension
    query = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    assert torch.allclose(
        lowered.respond_prepared_zero_past_constant(query, time=0.4),
        compiled.respond_prepared_zero_past_constant(query.to(DTYPE), time=0.4).float(),
        atol=1e-6,
    )
