import torch

from information_field.incremental_field import try_reduced_update, update_or_recompile
from information_field.matrix_free_field import compile_matrix_free_relation_field
from information_field.quotient_response import SparseRelationSource


DTYPE = torch.float64


def source(matrix, *, quantity_metric=None):
    return SparseRelationSource.from_dense(
        torch.tensor(matrix, dtype=DTYPE),
        quantity_metric=(
            None
            if quantity_metric is None
            else torch.tensor(quantity_metric, dtype=DTYPE)
        ),
    )


def fixture(values=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
    operator = torch.diag(torch.tensor(values, dtype=DTYPE))
    field = SparseRelationSource.from_dense(operator)
    port = torch.eye(len(values), dtype=DTYPE)[:, :4]
    observation = torch.eye(len(values), dtype=DTYPE)[:2]
    compiled = compile_matrix_free_relation_field(field, port, observation)
    return field, port, observation, compiled


def assert_same_response(left, right, width):
    incident = torch.linspace(-0.4, 0.7, width, dtype=DTYPE)
    a = left.respond_prepared_zero_past_constant(incident, time=0.6, mass=1.2)
    b = right.respond_prepared_zero_past_constant(incident, time=0.6, mass=1.2)
    assert torch.allclose(a, b, atol=1e-10)


def test_changed_dynamics_inside_old_reachable_carrier_updates_exactly():
    old, port, observation, compiled = fixture()
    new = SparseRelationSource.from_dense(
        torch.diag(torch.tensor([1.5, 2.5, 3.5, 4.5, 5.0, 6.0], dtype=DTYPE))
    )
    decision = try_reduced_update(compiled, old, new, port, observation)
    assert decision.status == "reduced-exact"
    assert decision.factorized_operator_applications == 1
    oracle = compile_matrix_free_relation_field(new, port, observation)
    assert decision.compilation.realization.state_dimension == oracle.realization.state_dimension
    assert_same_response(decision.compilation.realization, oracle.realization, 4)


def test_port_narrowing_prunes_old_reachable_directions_inside_carrier():
    old, _, observation, compiled = fixture()
    narrow = torch.eye(6, dtype=DTYPE)[:, :2]
    decision = try_reduced_update(compiled, old, old, narrow, observation)
    assert decision.status == "reduced-exact"
    assert decision.compilation.accounting.reachable_dimension == 2
    assert decision.compilation.accounting.minimal_dimension == 2
    oracle = compile_matrix_free_relation_field(old, narrow, observation)
    assert_same_response(decision.compilation.realization, oracle.realization, 2)


def test_disconnected_hidden_source_edit_reuses_reachable_carrier():
    old, port, observation, compiled = fixture()
    new = SparseRelationSource.from_dense(
        torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0, 11.0], dtype=DTYPE))
    )
    decision = try_reduced_update(compiled, old, new, port, observation)
    assert decision.status == "reduced-exact"
    assert decision.invariance_outside_residual == 0.0
    assert_same_response(decision.compilation.realization, compiled.realization, 4)


def test_new_coupling_leaving_old_carrier_requires_global_recompile():
    old, port, observation, compiled = fixture()
    changed = old.dense_operator()
    changed[4, 0] = 0.75
    new = SparseRelationSource.from_dense(changed)
    decision = try_reduced_update(compiled, old, new, port, observation)
    assert decision.status == "global-recompile-required"
    assert decision.invariance_outside_residual > 0
    assert decision.compilation is None

    completed = update_or_recompile(compiled, old, new, port, observation)
    assert completed.status == "global-recompiled"
    oracle = compile_matrix_free_relation_field(new, port, observation)
    assert_same_response(completed.compilation.realization, oracle.realization, 4)


def test_new_incident_outside_old_carrier_requires_global_recompile():
    old, _, observation, compiled = fixture()
    expanded = torch.eye(6, dtype=DTYPE)[:, :5]
    decision = try_reduced_update(compiled, old, old, expanded, observation)
    assert decision.status == "global-recompile-required"
    assert decision.incident_outside_residual > 0


def test_metric_change_always_uses_global_fallback():
    old, port, observation, compiled = fixture()
    changed = SparseRelationSource.from_dense(
        old.dense_operator(),
        quantity_metric=torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 1.0], dtype=DTYPE),
    )
    decision = try_reduced_update(compiled, old, changed, port, observation)
    assert decision.status == "global-recompile-required"
    assert decision.reason == "carrier metric changed"


def test_readout_change_inside_reachable_carrier_updates_observable_quotient():
    old, port, _, compiled = fixture()
    observation = torch.eye(6, dtype=DTYPE)[2:4]
    decision = try_reduced_update(compiled, old, old, port, observation)
    assert decision.status == "reduced-exact"
    assert decision.compilation.accounting.minimal_dimension == 2
    oracle = compile_matrix_free_relation_field(old, port, observation)
    assert_same_response(decision.compilation.realization, oracle.realization, 4)
