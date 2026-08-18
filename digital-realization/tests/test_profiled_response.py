import pytest
import torch

from information_field.causal_minimal import compile_minimal_realization
from information_field.observable_response import (
    compile_fixed_time_green,
    compile_grid_recurrence,
    compile_observable_spectrum,
    compile_sampled_green,
)
from information_field.profiled_response import (
    certify_dispatch_fusion,
    execute_prepared,
    prepare_backend_request,
    trace_prepared_dispatch,
)
from information_field.quotient_response import SparseIncidentBatch, SparseRelationSource, compile_static_response
from information_field.response_backends import BackendKind, BackendRequest, backend_capability, lower_backend_plan
from information_field.response_ir import (
    lower_fixed_time,
    lower_grid_recurrence,
    lower_sampled_times,
    lower_static_response,
)


DTYPE = torch.float64
MPS_AVAILABLE = backend_capability(BackendKind.MPS).available


def temporal_ir():
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
    spectrum = compile_observable_spectrum(realization)
    return (
        lower_fixed_time(
            compile_fixed_time_green(realization, spectrum, time=0.73, mass=1.4)
        ),
        lower_sampled_times(
            compile_sampled_green(realization, spectrum, times=(0.2, 0.5, 1.1))
        ),
        lower_grid_recurrence(
            compile_grid_recurrence(realization, step_size=0.17, mass=1.2)
        ),
    )


def fixed_request():
    return BackendRequest(
        torch.tensor([0.2, -0.6], dtype=DTYPE),
        initial_position=torch.tensor([0.4], dtype=DTYPE),
        initial_velocity=torch.tensor([-0.3], dtype=DTYPE),
    )


def test_prepared_fixed_response_matches_reference_ir():
    fixed, _, _ = temporal_ir()
    plan = lower_backend_plan(fixed, BackendKind.CPU, torch.float64)
    request = fixed_request()
    prepared = prepare_backend_request(plan, request)
    actual = execute_prepared(plan, prepared).result.output
    expected = fixed.execute(
        request.incident,
        initial_position=request.initial_position,
        initial_velocity=request.initial_velocity,
    ).output
    assert torch.allclose(actual, expected)


def test_prepared_sampled_response_uses_one_flattened_map():
    _, sampled, _ = temporal_ir()
    plan = lower_backend_plan(sampled, BackendKind.CPU, torch.float64)
    request = fixed_request()
    prepared = prepare_backend_request(plan, request)
    execution = execute_prepared(plan, prepared)
    expected = sampled.execute(
        request.incident,
        initial_position=request.initial_position,
        initial_velocity=request.initial_velocity,
    ).output
    assert execution.host_operator_calls == 1
    assert execution.result.output.shape == (3, 2)
    assert torch.allclose(execution.result.output, expected)


def test_prepared_static_embedding_bag_preserves_empty_batch_rows():
    source = SparseRelationSource.from_dense(
        torch.tensor([[1.0, 0.0], [0.5, -1.0], [0.0, 2.0]], dtype=DTYPE)
    )
    observation = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=DTYPE)
    compiled = compile_static_response(
        source, observation, torch.tensor([0, 1], dtype=torch.int64)
    )
    incidents = SparseIncidentBatch(
        torch.tensor([[0, 1], [0, 0], [1, 0]], dtype=torch.int64),
        torch.tensor([[0.3, -0.4], [0.0, 0.0], [0.7, 0.0]], dtype=DTYPE),
        torch.tensor([[True, True], [False, False], [True, False]]),
    )
    source_prepared = compiled.prepare(incidents)
    request = BackendRequest(
        source_prepared.amplitudes,
        local_indices=source_prepared.local_indices,
        valid=source_prepared.valid,
    )
    ir = lower_static_response(compiled)
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    prepared = prepare_backend_request(plan, request)
    actual = execute_prepared(plan, prepared).result.output
    assert torch.allclose(actual, compiled.run_prepared(source_prepared))
    assert torch.equal(actual[1], torch.zeros(2, dtype=DTYPE))


def test_prepared_grid_preserves_output_and_final_state():
    _, _, grid = temporal_ir()
    plan = lower_backend_plan(grid, BackendKind.CPU, torch.float64)
    request = BackendRequest(
        torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.8]], dtype=DTYPE)
    )
    prepared = prepare_backend_request(plan, request)
    actual = execute_prepared(plan, prepared).result
    expected = grid.execute(request.incident)
    assert torch.equal(actual.output, expected.output)
    assert torch.equal(actual.final_position, expected.final_position)
    assert torch.equal(actual.final_velocity, expected.final_velocity)


@pytest.mark.parametrize("index", [0, 1])
def test_cpu_packed_maps_have_one_profiled_core_mv(index):
    fixed, sampled, _ = temporal_ir()
    ir = (fixed, sampled)[index]
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    prepared = prepare_backend_request(plan, fixed_request())
    trace = trace_prepared_dispatch(plan, prepared)
    certificate = certify_dispatch_fusion(plan, trace)
    assert dict(trace.core_operation_counts)["aten::addmv_"] == 1
    assert certificate.fusion_certified
    assert certificate.backend_kernel_trace


def test_cpu_static_wrapper_decomposes_and_is_not_called_fused():
    source = SparseRelationSource.from_dense(torch.eye(2, dtype=DTYPE))
    compiled = compile_static_response(
        source, torch.eye(2, dtype=DTYPE), torch.tensor([0, 1], dtype=torch.int64)
    )
    incidents = SparseIncidentBatch(
        torch.tensor([[0, 1]], dtype=torch.int64),
        torch.tensor([[0.3, -0.4]], dtype=DTYPE),
        torch.tensor([[True, True]]),
    )
    item = compiled.prepare(incidents)
    ir = lower_static_response(compiled)
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    prepared = prepare_backend_request(
        plan,
        BackendRequest(item.amplitudes, local_indices=item.local_indices, valid=item.valid),
    )
    trace = trace_prepared_dispatch(plan, prepared)
    certificate = certify_dispatch_fusion(plan, trace)
    assert trace.core_calls >= 2
    assert not certificate.fusion_certified


def test_regular_grid_trace_is_not_mislabeled_as_fused():
    _, _, grid = temporal_ir()
    plan = lower_backend_plan(grid, BackendKind.CPU, torch.float64)
    prepared = prepare_backend_request(
        plan,
        BackendRequest(torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=DTYPE)),
    )
    trace = trace_prepared_dispatch(plan, prepared)
    certificate = certify_dispatch_fusion(plan, trace)
    assert not certificate.fusion_certified
    assert trace.core_calls > 1


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is unavailable")
def test_mps_host_trace_does_not_claim_metal_kernel_fusion():
    fixed, _, _ = temporal_ir()
    plan = lower_backend_plan(fixed, BackendKind.MPS, torch.float32)
    prepared = prepare_backend_request(plan, fixed_request())
    trace = trace_prepared_dispatch(plan, prepared)
    certificate = certify_dispatch_fusion(plan, trace)
    assert dict(trace.core_operation_counts)["aten::mm"] == 1
    assert not certificate.backend_kernel_trace
    assert not certificate.fusion_certified


def test_prepared_request_cannot_cross_plan_boundaries():
    fixed, sampled, _ = temporal_ir()
    fixed_plan = lower_backend_plan(fixed, BackendKind.CPU, torch.float64)
    sampled_plan = lower_backend_plan(sampled, BackendKind.CPU, torch.float64)
    prepared = prepare_backend_request(fixed_plan, fixed_request())
    with pytest.raises(ValueError):
        execute_prepared(sampled_plan, prepared)
