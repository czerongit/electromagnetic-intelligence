import pytest
import torch

from information_field.causal_minimal import compile_relation_field
from information_field.matrix_free_field import compile_matrix_free_relation_field
from information_field.observable_response import compile_fixed_time_green, compile_observable_spectrum
from information_field.quotient_response import SparseRelationSource


DTYPE = torch.float64


def source(operator, *, quantity_metric=None, relation_metric=None):
    value = torch.tensor(operator, dtype=DTYPE)
    return SparseRelationSource.from_dense(
        value,
        quantity_metric=(
            None
            if quantity_metric is None
            else torch.tensor(quantity_metric, dtype=DTYPE)
        ),
        relation_metric=(
            None
            if relation_metric is None
            else torch.tensor(relation_metric, dtype=DTYPE)
        ),
    )


def test_matrix_free_compiler_matches_dense_metric_oracle():
    field = source(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 2.0, -0.25, 0.0],
            [0.3, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
        quantity_metric=[2.0, 3.0, 1.5, 0.8],
        relation_metric=[1.2, 0.7, 2.0, 1.1],
    )
    relation_port = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.2], [0.0, 0.0]], dtype=DTYPE
    )
    observation = torch.tensor(
        [[1.0, -0.5, 0.25, 0.0], [0.0, 0.4, 1.0, 0.0]], dtype=DTYPE
    )
    sparse = compile_matrix_free_relation_field(
        field, relation_port, observation, calibration=0.8
    )
    dense = compile_relation_field(
        field, relation_port, observation, calibration=0.8
    )
    assert sparse.realization.state_dimension == dense.state_dimension
    query = torch.tensor([0.3, -0.7], dtype=DTYPE)
    assert torch.allclose(
        sparse.realization.respond_prepared_zero_past_constant(
            query, time=0.6, mass=1.3
        ),
        dense.respond_prepared_zero_past_constant(query, time=0.6, mass=1.3),
        atol=1e-9,
    )
    assert sparse.realization.certificate.maximum_markov_residual < 1e-9


def test_compiler_never_calls_dense_source_construction(monkeypatch):
    field = source([[1.0, 0.0], [0.0, 2.0], [0.5, 0.0]])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense source construction was called")

    monkeypatch.setattr(SparseRelationSource, "dense_operator", forbidden)
    monkeypatch.setattr(SparseRelationSource, "whitened_dense", forbidden)
    compiled = compile_matrix_free_relation_field(
        field,
        torch.eye(2, dtype=DTYPE),
        torch.tensor([[1.0, 0.0, -0.5]], dtype=DTYPE),
    )
    assert compiled.accounting.dense_relation_operator_materialized is False
    assert compiled.accounting.dense_intrinsic_operator_materialized is False
    assert compiled.accounting.d_applications > 0
    assert compiled.accounting.adjoint_applications > 0


def test_disconnected_unreachable_and_invisible_coordinates_are_removed():
    diagonal = torch.arange(1, 33, dtype=DTYPE)
    field = SparseRelationSource.from_dense(torch.diag(diagonal))
    relation_port = torch.eye(32, dtype=DTYPE)[:, :4]
    observation = torch.eye(32, dtype=DTYPE)[:2]
    compiled = compile_matrix_free_relation_field(field, relation_port, observation)
    assert compiled.accounting.reachable_dimension == 4
    assert compiled.accounting.minimal_dimension == 2
    assert compiled.realization.state_dimension == 2


def test_declared_initial_ports_expand_reachability_and_preserve_response():
    field = source([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    relation_port = torch.tensor([[1.0], [0.0]], dtype=DTYPE)
    observation = torch.tensor([[1.0, 0.0, 1.0]], dtype=DTYPE)
    position_port = torch.tensor([[0.0], [0.0], [1.0]], dtype=DTYPE)
    compiled = compile_matrix_free_relation_field(
        field,
        relation_port,
        observation,
        initial_position_port=position_port,
    )
    assert compiled.realization.state_dimension == 2
    spectrum = compile_observable_spectrum(compiled.realization)
    fixed = compile_fixed_time_green(
        compiled.realization, spectrum, time=0.4, mass=1.1
    )
    output = fixed.run(
        torch.tensor([0.2], dtype=DTYPE),
        initial_position=torch.tensor([0.7], dtype=DTYPE),
    )
    expected = compiled.realization.respond_constant(
        torch.tensor([0.2], dtype=DTYPE),
        time=0.4,
        mass=1.1,
        initial_position=torch.tensor([0.7], dtype=DTYPE),
    )
    assert torch.allclose(output, expected, atol=1e-10)


def test_every_metric_source_port_and_calibration_change_invalidates():
    field = source([[1.0, 0.0], [0.0, 2.0]])
    relation_port = torch.eye(2, dtype=DTYPE)
    observation = torch.eye(2, dtype=DTYPE)
    compiled = compile_matrix_free_relation_field(field, relation_port, observation)
    assert compiled.is_valid_for(field, relation_port, observation)
    metric_change = source(
        [[1.0, 0.0], [0.0, 2.0]], quantity_metric=[1.0, 2.0]
    )
    assert not compiled.is_valid_for(metric_change, relation_port, observation)
    assert not compiled.is_valid_for(field, relation_port.flip(1), observation)
    assert not compiled.is_valid_for(field, relation_port, observation.flip(1))
    assert not compiled.is_valid_for(
        field, relation_port, observation, calibration=2.0
    )


def test_one_silent_incident_does_not_remove_another_port_column():
    field = source([[1.0, 0.0], [0.0, 2.0]])
    relation_port = torch.eye(2, dtype=DTYPE)
    observation = torch.eye(2, dtype=DTYPE)
    compiled = compile_matrix_free_relation_field(field, relation_port, observation)
    assert compiled.realization.state_dimension == 2
    response = compiled.realization.respond_prepared_zero_past_constant(
        torch.tensor([1.0, 0.0], dtype=DTYPE), time=0.5
    )
    assert response[1] == 0
    other = compiled.realization.respond_prepared_zero_past_constant(
        torch.tensor([0.0, 1.0], dtype=DTYPE), time=0.5
    )
    assert other[1] != 0


def test_weak_but_certified_relation_is_retained():
    field = source([[1.0, 0.0], [0.0, 1e-4]])
    compiled = compile_matrix_free_relation_field(
        field,
        torch.eye(2, dtype=DTYPE),
        torch.eye(2, dtype=DTYPE),
        tolerance=1e-12,
    )
    assert compiled.realization.state_dimension == 2
