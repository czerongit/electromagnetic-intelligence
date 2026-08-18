import torch

from information_field.causal_minimal import compile_minimal_realization
from information_field.observable_response import (
    compile_fixed_time_green,
    compile_grid_recurrence,
    compile_observable_spectrum,
    compile_sampled_green,
)
from information_field.quotient_response import (
    SparseIncidentBatch,
    SparseRelationSource,
    compile_static_response,
)
from information_field.response_ir import (
    InvalidationKey,
    ResponseContract,
    SemanticOperation,
    lower_fixed_time,
    lower_grid_recurrence,
    lower_sampled_times,
    lower_static_response,
)


DTYPE = torch.float64


def temporal_realization():
    operator = torch.diag(torch.tensor([1.0, 4.0, 9.0], dtype=DTYPE))
    incident = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]], dtype=DTYPE
    )
    observation = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=DTYPE
    )
    position = torch.tensor([[1.0], [0.0], [0.5]], dtype=DTYPE)
    velocity = torch.tensor([[0.0], [1.0], [-0.25]], dtype=DTYPE)
    realization = compile_minimal_realization(
        operator,
        incident,
        observation,
        initial_position_port=position,
        initial_velocity_port=velocity,
    )
    return realization, compile_observable_spectrum(realization)


def test_static_column_ir_matches_relation_native_compilation():
    source = SparseRelationSource.from_dense(
        torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 2.0, -1.0], [1.0, 1.0, 0.0]],
            dtype=DTYPE,
        )
    )
    observation = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=DTYPE)
    selected = torch.tensor([0, 2], dtype=torch.int64)
    compiled = compile_static_response(source, observation, selected)
    ir = lower_static_response(compiled)
    incidents = SparseIncidentBatch(
        torch.tensor([[0, 2], [2, 0]], dtype=torch.int64),
        torch.tensor([[0.4, -0.3], [0.7, 0.0]], dtype=DTYPE),
        torch.tensor([[True, True], [True, False]]),
    )
    prepared = compiled.prepare(incidents)
    actual = ir.execute(
        prepared.amplitudes,
        local_indices=prepared.local_indices,
        valid=prepared.valid,
    ).output
    assert ir.contract is ResponseContract.STATIC_COLUMNS
    assert ir.operations == (SemanticOperation.WEIGHTED_COLUMN_REDUCE,)
    assert torch.allclose(actual, compiled.run_prepared(prepared))


def test_fixed_time_ir_preserves_incident_and_initial_state_ports():
    realization, spectrum = temporal_realization()
    fixed = compile_fixed_time_green(
        realization, spectrum, time=0.73, mass=1.4
    )
    ir = lower_fixed_time(fixed)
    incident = torch.tensor([0.2, -0.6], dtype=DTYPE)
    position = torch.tensor([0.4], dtype=DTYPE)
    velocity = torch.tensor([-0.3], dtype=DTYPE)
    actual = ir.execute(
        incident,
        initial_position=position,
        initial_velocity=velocity,
    ).output
    expected = fixed.run(
        incident, initial_position=position, initial_velocity=velocity
    )
    assert torch.allclose(actual, expected)


def test_sampled_time_ir_preserves_the_complete_map_family():
    realization, spectrum = temporal_realization()
    sampled = compile_sampled_green(
        realization, spectrum, times=(0.2, 0.5, 1.1), mass=1.3
    )
    ir = lower_sampled_times(sampled)
    incident = torch.tensor([0.3, 0.8], dtype=DTYPE)
    position = torch.tensor([-0.2], dtype=DTYPE)
    velocity = torch.tensor([0.1], dtype=DTYPE)
    expected = (
        torch.einsum("tzi,i->tz", sampled.incident_maps, incident)
        + torch.einsum("tzp,p->tz", sampled.initial_position_maps, position)
        + torch.einsum("tzv,v->tz", sampled.initial_velocity_maps, velocity)
    )
    actual = ir.execute(
        incident, initial_position=position, initial_velocity=velocity
    ).output
    assert ir.contract is ResponseContract.SAMPLED_TIMES
    assert torch.allclose(actual, expected)


def test_grid_ir_matches_exact_stateful_recurrence():
    realization, _ = temporal_realization()
    grid = compile_grid_recurrence(realization, step_size=0.17, mass=1.2)
    ir = lower_grid_recurrence(grid)
    incidents = torch.tensor(
        [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.8], [0.0, -0.5]], dtype=DTYPE
    )
    expected, expected_state = grid.rollout(incidents)
    actual = ir.execute(incidents)
    assert ir.state_dimension == grid.state_dimension
    assert torch.allclose(actual.output, expected)
    assert torch.allclose(actual.final_position, expected_state.position)
    assert torch.allclose(actual.final_velocity, expected_state.velocity)


def test_grid_ir_accepts_an_explicit_prior_state():
    realization, _ = temporal_realization()
    grid = compile_grid_recurrence(realization, step_size=0.2)
    ir = lower_grid_recurrence(grid)
    prior = grid.initial_state(
        position=torch.tensor([0.7], dtype=DTYPE),
        velocity=torch.tensor([-0.2], dtype=DTYPE),
    )
    incidents = torch.tensor([[0.1, 0.2], [-0.4, 0.3]], dtype=DTYPE)
    expected, expected_state = grid.rollout(incidents, initial=prior)
    actual = ir.execute(
        incidents,
        state_position=prior.position,
        state_velocity=prior.velocity,
    )
    assert torch.allclose(actual.output, expected)
    assert torch.allclose(actual.final_position, expected_state.position)
    assert torch.allclose(actual.final_velocity, expected_state.velocity)


def test_workload_change_invalidates_a_fixed_time_view():
    realization, spectrum = temporal_realization()
    first = lower_fixed_time(
        compile_fixed_time_green(realization, spectrum, time=0.5)
    )
    second = lower_fixed_time(
        compile_fixed_time_green(realization, spectrum, time=0.6)
    )
    assert first.is_valid_for(first.invalidation_key)
    assert not first.is_valid_for(second.invalidation_key)


def test_unrelated_key_cannot_validate_an_execution_view():
    realization, spectrum = temporal_realization()
    ir = lower_fixed_time(
        compile_fixed_time_green(realization, spectrum, time=0.5)
    )
    wrong = InvalidationKey.create(source="changed", workload="fixed")
    assert not ir.is_valid_for(wrong)


def test_binding_mutation_is_detected_before_execution():
    realization, spectrum = temporal_realization()
    ir = lower_fixed_time(
        compile_fixed_time_green(realization, spectrum, time=0.5)
    )
    ir.tensor("incident_map")[0, 0] += 1.0
    try:
        ir.execute(torch.tensor([0.0, 0.0], dtype=DTYPE))
    except ValueError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("mutated IR tensor was accepted")


def test_precision_is_unqualified_until_a_backend_is_checked():
    realization, spectrum = temporal_realization()
    ir = lower_fixed_time(
        compile_fixed_time_green(realization, spectrum, time=0.5)
    )
    assert ir.precision.reference_dtype == "torch.float64"
    assert ir.precision.exact_reference
    assert not ir.precision.target_dtype_qualified
    assert ir.precision.maximum_absolute_error is None


def test_ir_contains_field_operations_without_attention_nodes():
    values = {operation.value for operation in SemanticOperation}
    assert "query-key-score" not in values
    assert "softmax" not in values
    assert "weighted-attention-sum" not in values
    assert {contract.value for contract in ResponseContract} == {
        "static-columns",
        "fixed-time",
        "sampled-times",
        "regular-grid",
    }
