import math

import pytest
import torch

from information_field.quotient_response import (
    CausalState,
    QuotientResponseAdapter,
    SparseIncident,
    SparseIncidentBatch,
    SparseRelationSource,
    ambient_projector_oracle,
    certify_compatible_incidents,
    choose_causal_plan,
    compile_exact_modal,
    compile_static_response,
    dense_causal_oracle,
    dense_second_order_rk4,
    dense_static_oracle,
    from_declarative_field,
    source_change_invalidation,
    sparse_first_order_evolve,
    update_static_compilation,
)


DTYPE = torch.float64


def source(operator, *, quantity_metric=None, relation_metric=None):
    value = torch.tensor(operator, dtype=DTYPE)
    return SparseRelationSource.from_dense(
        value,
        quantity_metric=(
            torch.tensor(quantity_metric, dtype=DTYPE)
            if quantity_metric is not None
            else None
        ),
        relation_metric=(
            torch.tensor(relation_metric, dtype=DTYPE)
            if relation_metric is not None
            else None
        ),
    )


def batch(indices, amplitudes, valid=None):
    index = torch.tensor(indices, dtype=torch.int64)
    amplitude = torch.tensor(amplitudes, dtype=DTYPE)
    return SparseIncidentBatch(
        index,
        amplitude,
        torch.ones_like(index, dtype=torch.bool)
        if valid is None
        else torch.tensor(valid, dtype=torch.bool),
    )


def incident(indices, amplitudes):
    return SparseIncident(
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(amplitudes, dtype=DTYPE),
    )


def test_sparse_source_and_metric_adjoint_identity():
    field = source(
        [[1.0, 0.0, 2.0], [0.0, -1.0, 0.5]],
        quantity_metric=[2.0, 3.0],
        relation_metric=[5.0, 7.0, 11.0],
    )
    g = torch.tensor([0.3, -0.8, 1.2], dtype=DTYPE)
    h = torch.tensor([-0.4, 0.9], dtype=DTYPE)
    left = torch.sum(field.apply(g) * field.quantity_metric * h)
    right = torch.sum(g * field.relation_metric * field.adjoint_apply(h))
    assert torch.allclose(left, right)


def test_kernel_equivalent_readings_determine_one_covector():
    field = source([[0.5, 0.5], [1.0, 1.0]])
    result = certify_compatible_incidents(
        field,
        (incident([0], [1.0]), incident([1], [1.0])),
    )
    assert result.accepted
    assert result.compatible_count == 2
    assert torch.allclose(result.induced_source, torch.tensor([0.5, 1.0], dtype=DTYPE))


def test_materially_different_readings_reject_single_response():
    field = source([[1.0, 0.0], [0.0, 1.0]])
    result = certify_compatible_incidents(
        field,
        (incident([0], [1.0]), incident([1], [1.0])),
    )
    assert result.status == "ambiguous"
    assert result.induced_source is None
    assert result.maximum_image_difference == 1.0


def test_silent_incident_is_distinct_from_unsupported_question():
    field = source([[1.0, 1.0]])
    silent = certify_compatible_incidents(field, (incident([0, 1], [1.0, -1.0]),))
    unsupported = certify_compatible_incidents(field, ())
    invalid = certify_compatible_incidents(field, (incident([2], [1.0]),))
    assert silent.status == "determined"
    assert torch.count_nonzero(silent.induced_source) == 0
    assert unsupported.status == "unsupported"
    assert invalid.status == "unsupported"


def test_fused_static_response_matches_dense_and_ambient_without_intermediates():
    field = source(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0], [1.0, 0.0, -1.0, 0.0]],
        quantity_metric=[2.0, 1.0, 3.0],
    )
    observation = torch.tensor([[1.0, 0.5, -0.25], [0.0, 1.0, 0.5]], dtype=DTYPE)
    inputs = batch([[0, 2], [1, 3]], [[0.5, -1.0], [0.25, 2.0]])
    compiled = compile_static_response(field, observation, inputs.admitted_features())
    actual = compiled.run(inputs)
    assert torch.allclose(actual, dense_static_oracle(field, observation, inputs), atol=1e-10)
    assert torch.allclose(actual, ambient_projector_oracle(field, observation, inputs), atol=1e-10)
    assert compiled.accounting.projector_applications == 0
    assert compiled.accounting.ambient_source_materializations == 0
    assert not compiled.accounting.dense_observation_operator_materialized


def test_compiled_static_path_rejects_uncompiled_feature():
    field = source([[1.0, 0.0], [0.0, 1.0]])
    compiled = compile_static_response(
        field, torch.eye(2, dtype=DTYPE), torch.tensor([0], dtype=torch.int64)
    )
    with pytest.raises(ValueError, match="outside the compiled"):
        compiled.run(batch([[1]], [[1.0]]))


def test_prepared_static_trace_contains_no_matrix_or_projector_operation():
    field = source([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    inputs = batch([[0, 2]], [[0.5, -1.0]])
    compiled = compile_static_response(
        field,
        torch.tensor([[1.0, -0.5]], dtype=DTYPE),
        inputs.admitted_features(),
    )
    prepared = compiled.prepare(inputs)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        compiled.run_prepared(prepared)
    operations = {event.key for event in profile.key_averages()}
    assert "aten::index" in operations
    assert "aten::mul" in operations
    assert "aten::sum" in operations
    assert not any("mm" in operation or "linalg" in operation for operation in operations)


def test_static_adapter_preserves_batch_behavior_and_is_inference_only():
    field = source([[1.0, 0.0], [0.0, 1.0]])
    adapter = QuotientResponseAdapter(
        field,
        torch.eye(2, dtype=DTYPE),
        torch.tensor([0, 1], dtype=torch.int64),
    )
    inputs = batch([[0], [1]], [[2.0], [3.0]])
    assert torch.equal(adapter(inputs), torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=DTYPE))
    with pytest.raises(RuntimeError, match="inference-only"):
        adapter(
            SparseIncidentBatch(
                inputs.indices,
                inputs.amplitudes.clone().requires_grad_(),
                inputs.valid,
            )
        )


def test_sparse_first_order_matches_dense_second_order_without_forming_l():
    field = source(
        [[1.0, 0.0, 0.5], [0.0, 2.0, -0.25], [0.3, 0.0, 1.0]],
        quantity_metric=[2.0, 3.0, 1.5],
        relation_metric=[1.2, 0.7, 2.0],
    )
    q = torch.tensor([0.4, -0.2, 0.8], dtype=DTYPE)
    initial = CausalState(
        torch.tensor([0.2, -0.1, 0.5], dtype=DTYPE),
        torch.tensor([-0.3, 0.4, 0.1], dtype=DTYPE),
    )
    sparse = sparse_first_order_evolve(
        field, q, initial, time=0.4, steps=80, mass=1.7, calibration=0.8
    )
    dense = dense_second_order_rk4(
        field, q, initial, time=0.4, steps=80, mass=1.7, calibration=0.8
    )
    assert torch.allclose(sparse.position, dense.position, atol=1e-10)
    assert torch.allclose(sparse.velocity, dense.velocity, atol=1e-10)


def test_exact_modal_response_matches_dense_causal_oracle_and_keeps_null_motion():
    field = source(
        [[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]],
        quantity_metric=[2.0, 3.0, 4.0],
        relation_metric=[5.0, 7.0],
    )
    q = torch.tensor([0.25, -0.5], dtype=DTYPE)
    initial = CausalState(
        torch.tensor([0.1, 0.2, 3.0], dtype=DTYPE),
        torch.tensor([-0.2, 0.3, 0.7], dtype=DTYPE),
    )
    modal = compile_exact_modal(field)
    actual = modal.evolve_constant(q, initial, time=0.6, mass=1.3, calibration=0.9)
    expected = dense_causal_oracle(
        field, q, initial, time=0.6, mass=1.3, calibration=0.9
    )
    assert modal.rank == 2
    assert torch.allclose(actual.position, expected.position, atol=1e-10)
    assert torch.allclose(actual.velocity, expected.velocity, atol=1e-10)
    assert math.isclose(float(actual.position[2]), 3.0 + 0.6 * 0.7)


def test_plan_selector_uses_modal_only_when_work_and_storage_both_win():
    modal = choose_causal_plan(
        quantity_dim=256,
        relation_dim=512,
        nonzeros=20_000,
        exact_rank=8,
        time_steps=100,
        expected_runs=100,
    )
    sparse = choose_causal_plan(
        quantity_dim=256,
        relation_dim=512,
        nonzeros=512,
        exact_rank=128,
        time_steps=2,
        expected_runs=1,
    )
    assert modal.plan == "exact-modal"
    assert sparse.plan == "sparse-first-order"


def test_source_edit_invalidates_only_dependent_static_features_but_all_modal_data():
    before = source([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    observation = torch.eye(2, dtype=DTYPE)
    compiled = compile_static_response(
        before, observation, torch.tensor([0, 1], dtype=torch.int64)
    )
    unrelated = source([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])
    related = source([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    unrelated_decision = source_change_invalidation(before, unrelated, compiled)
    related_decision = source_change_invalidation(before, related, compiled)
    assert unrelated_decision.static_compilation_valid
    assert not unrelated_decision.modal_compilation_valid
    assert not related_decision.static_compilation_valid
    assert not related_decision.modal_compilation_valid


def test_static_source_edit_updates_only_changed_compiled_columns():
    before = source([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]])
    after = source([[4.0, 0.0, 3.0], [0.0, 2.0, 0.0]])
    observation = torch.tensor([[1.0, -0.5]], dtype=DTYPE)
    inputs = batch([[0, 1]], [[0.25, 0.75]])
    compiled = compile_static_response(
        before, observation, inputs.admitted_features()
    )
    updated = update_static_compilation(
        compiled, before, after, observation
    )
    fresh = compile_static_response(after, observation, inputs.admitted_features())
    assert torch.equal(
        updated.observed_columns[1], compiled.observed_columns[1]
    )
    assert torch.equal(updated.observed_columns, fresh.observed_columns)
    assert torch.equal(updated.run(inputs), fresh.run(inputs))


def test_query_change_does_not_invalidate_source_certificate_but_source_change_does():
    field = source([[1.0, 0.0], [0.0, 1.0]])
    certificate = certify_compatible_incidents(field, (incident([0], [1.0]),))
    adapter = QuotientResponseAdapter(
        field,
        torch.eye(2, dtype=DTYPE),
        torch.tensor([0, 1], dtype=torch.int64),
    )
    adapter.verify_certificate(certificate)
    changed = source([[2.0, 0.0], [0.0, 1.0]])
    changed_adapter = QuotientResponseAdapter(
        changed,
        torch.eye(2, dtype=DTYPE),
        torch.tensor([0, 1], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="another source"):
        changed_adapter.verify_certificate(certificate)


def test_declarative_source_lowers_to_the_same_sparse_dq():
    from information_field.geometric_observation import determine_declarative_field

    declarations = (
        "Red fox crosses the hill.",
        "Blue wolf crosses the hill.",
        "Red fox enters the cave.",
        "Blue wolf enters the cave.",
    )
    field = determine_declarative_field(
        declarations,
        radius=2,
        minimum_occurrences=1,
        normalization="joint",
    )
    bridge = from_declarative_field(field)
    context = ((-2, "red"), (-1, "fox"), (1, "the"), (2, "cave"))
    q = bridge.incident(context)
    induced = bridge.source.apply(q.dense(bridge.source.relation_dim))
    expected_sparse = field.source_covector(context)
    expected = torch.tensor(
        [expected_sparse.get(term, 0.0) for term in bridge.terms], dtype=DTYPE
    )
    assert torch.allclose(induced, expected)


ACCELERATORS = tuple(
    device
    for device, available in (
        ("mps", torch.backends.mps.is_available()),
        ("cuda", torch.cuda.is_available()),
    )
    if available
)


@pytest.mark.parametrize("device", ACCELERATORS or ("unavailable",))
def test_available_accelerator_static_lowering_matches_cpu(device):
    if device == "unavailable":
        pytest.skip("no accelerator available")
    cpu_source = source([[1.0, 0.0, 2.0], [0.0, -1.0, 0.5]])
    observation = torch.tensor([[0.5, 1.0]], dtype=DTYPE)
    inputs = batch([[0, 2], [1, 2]], [[1.0, -0.5], [0.25, 2.0]])
    expected = compile_static_response(
        cpu_source, observation, inputs.admitted_features()
    ).run(inputs)
    accelerator_source = cpu_source.to(device, torch.float32)
    accelerator_inputs = inputs.to(device, torch.float32)
    actual = compile_static_response(
        accelerator_source,
        observation.to(device, torch.float32),
        accelerator_inputs.admitted_features(),
    ).run(accelerator_inputs)
    assert torch.allclose(actual.cpu().to(DTYPE), expected, atol=1e-5)


@pytest.mark.parametrize("device", ACCELERATORS or ("unavailable",))
def test_available_accelerator_sparse_causal_lowering_matches_cpu(device):
    if device == "unavailable":
        pytest.skip("no accelerator available")
    cpu_source = source([[1.0, 0.0], [0.25, 2.0]])
    q = torch.tensor([0.4, -0.3], dtype=torch.float32)
    initial = CausalState(
        torch.tensor([0.2, -0.1], dtype=torch.float32),
        torch.tensor([-0.3, 0.5], dtype=torch.float32),
    )
    expected = sparse_first_order_evolve(
        cpu_source.to("cpu", torch.float32),
        q,
        initial,
        time=0.2,
        steps=20,
    )
    accelerator_source = cpu_source.to(device, torch.float32)
    actual = sparse_first_order_evolve(
        accelerator_source,
        q.to(device),
        CausalState(initial.position.to(device), initial.velocity.to(device)),
        time=0.2,
        steps=20,
    )
    assert torch.allclose(actual.position.cpu(), expected.position, atol=1e-5)
    assert torch.allclose(actual.velocity.cpu(), expected.velocity, atol=1e-5)
