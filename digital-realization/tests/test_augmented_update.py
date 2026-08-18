import torch

from information_field.augmented_update import try_augmented_update, update_with_augmentation_or_recompile
from information_field.matrix_free_field import compile_matrix_free_relation_field
from information_field.quotient_response import SparseRelationSource


DTYPE = torch.float64


def fixture(dimension=12):
    diagonal = torch.arange(1, dimension + 1, dtype=DTYPE)
    old = SparseRelationSource.from_dense(torch.diag(diagonal))
    port = torch.eye(dimension, dtype=DTYPE)[:, :4]
    observation = torch.eye(dimension, dtype=DTYPE)[:4]
    compiled = compile_matrix_free_relation_field(old, port, observation)
    return old, port, observation, compiled


def compare(left, right, width):
    incident = torch.linspace(-0.4, 0.7, width, dtype=DTYPE)
    a = left.respond_prepared_zero_past_constant(incident, time=0.6, mass=1.2)
    b = right.respond_prepared_zero_past_constant(incident, time=0.6, mass=1.2)
    assert torch.allclose(a, b, atol=1e-10)


def test_one_new_reachable_direction_is_augmented_exactly():
    old, port, _, compiled = fixture()
    changed = old.dense_operator()
    changed[4, 0] = 0.5
    new = SparseRelationSource.from_dense(changed)
    observation = torch.eye(12, dtype=DTYPE)[:5]
    decision = try_augmented_update(compiled, old, new, port, observation)
    assert decision.status == "augmented-exact"
    assert decision.added_dimensions == 1
    assert decision.augmented_dimension == 5
    assert decision.final_reachable_dimension == 5
    oracle = compile_matrix_free_relation_field(new, port, observation)
    compare(decision.compilation.realization, oracle.realization, 4)


def test_expanded_incident_port_adds_only_its_new_direction():
    old, _, _, compiled = fixture()
    port = torch.eye(12, dtype=DTYPE)[:, :5]
    observation = torch.eye(12, dtype=DTYPE)[:5]
    decision = try_augmented_update(compiled, old, old, port, observation)
    assert decision.status == "augmented-exact"
    assert decision.added_dimensions == 1
    oracle = compile_matrix_free_relation_field(old, port, observation)
    compare(decision.compilation.realization, oracle.realization, 5)


def test_multistep_changed_law_closes_every_new_direction():
    old, port, _, compiled = fixture()
    changed = old.dense_operator()
    changed[4, 0] = 0.5
    changed[5, 4] = 0.25
    new = SparseRelationSource.from_dense(changed)
    observation = torch.eye(12, dtype=DTYPE)[:6]
    decision = try_augmented_update(compiled, old, new, port, observation)
    assert decision.status == "augmented-exact"
    assert decision.added_dimensions == 2
    assert decision.final_reachable_dimension == 6
    oracle = compile_matrix_free_relation_field(new, port, observation)
    compare(decision.compilation.realization, oracle.realization, 4)


def test_update_budget_forces_visible_global_fallback():
    old, port, _, compiled = fixture()
    changed = old.dense_operator()
    changed[4, 0] = 0.5
    changed[5, 4] = 0.25
    new = SparseRelationSource.from_dense(changed)
    observation = torch.eye(12, dtype=DTYPE)[:6]
    decision = try_augmented_update(
        compiled,
        old,
        new,
        port,
        observation,
        maximum_added_dimensions=1,
    )
    assert decision.status == "global-recompile-required"
    assert decision.added_dimensions == 2
    completed = update_with_augmentation_or_recompile(
        compiled,
        old,
        new,
        port,
        observation,
        maximum_added_dimensions=1,
    )
    assert completed.status == "global-recompiled"
    oracle = compile_matrix_free_relation_field(new, port, observation)
    compare(completed.compilation.realization, oracle.realization, 4)


def test_obsolete_old_directions_are_pruned_after_augmentation():
    old, _, _, compiled = fixture()
    changed = old.dense_operator()
    changed[4, 0] = 0.5
    new = SparseRelationSource.from_dense(changed)
    narrow_port = torch.eye(12, dtype=DTYPE)[:, :1]
    observation = torch.eye(12, dtype=DTYPE)[:5]
    decision = try_augmented_update(
        compiled, old, new, narrow_port, observation
    )
    assert decision.status == "augmented-exact"
    assert decision.augmented_dimension == 5
    assert decision.final_reachable_dimension == 2
    oracle = compile_matrix_free_relation_field(new, narrow_port, observation)
    compare(decision.compilation.realization, oracle.realization, 1)


def test_metric_change_never_reuses_old_carrier():
    old, port, observation, compiled = fixture()
    new = SparseRelationSource.from_dense(
        old.dense_operator(),
        quantity_metric=torch.linspace(1.0, 2.0, 12, dtype=DTYPE),
    )
    decision = try_augmented_update(compiled, old, new, port, observation)
    assert decision.status == "global-recompile-required"
    assert decision.reason == "carrier metric changed"


def test_initial_state_port_can_supply_new_reachable_direction():
    old, port, _, compiled = fixture()
    observation = torch.eye(12, dtype=DTYPE)[:5]
    position = torch.eye(12, dtype=DTYPE)[:, 4:5]
    decision = try_augmented_update(
        compiled,
        old,
        old,
        port,
        observation,
        initial_position_port=position,
    )
    assert decision.status == "augmented-exact"
    assert decision.added_dimensions == 1
    oracle = compile_matrix_free_relation_field(
        old, port, observation, initial_position_port=position
    )
    incident = torch.linspace(-0.4, 0.7, 4, dtype=DTYPE)
    initial = torch.tensor([0.6], dtype=DTYPE)
    actual = decision.compilation.realization.respond_constant(
        incident, time=0.5, initial_position=initial
    )
    expected = oracle.realization.respond_constant(
        incident, time=0.5, initial_position=initial
    )
    assert torch.allclose(actual, expected, atol=1e-10)
