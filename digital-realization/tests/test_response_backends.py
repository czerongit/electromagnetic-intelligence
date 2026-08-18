import pytest
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
from information_field.response_backends import (
    BackendKind,
    BackendRequest,
    backend_capability,
    choose_backend,
    execute_qualified,
    lower_backend_plan,
    qualify_backend_plan,
)
from information_field.response_ir import (
    lower_fixed_time,
    lower_grid_recurrence,
    lower_sampled_times,
    lower_static_response,
)


DTYPE = torch.float64
MPS_AVAILABLE = backend_capability(BackendKind.MPS).available


def temporal_objects(time=0.73):
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
    fixed = compile_fixed_time_green(realization, spectrum, time=time, mass=1.4)
    sampled = compile_sampled_green(
        realization, spectrum, times=(0.2, 0.5, 1.1), mass=1.3
    )
    grid = compile_grid_recurrence(realization, step_size=0.17, mass=1.2)
    return fixed, sampled, grid


def fixed_request():
    return BackendRequest(
        torch.tensor([0.2, -0.6], dtype=DTYPE),
        initial_position=torch.tensor([0.4], dtype=DTYPE),
        initial_velocity=torch.tensor([-0.3], dtype=DTYPE),
    )


def grid_request():
    return BackendRequest(
        torch.tensor(
            [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.8], [0.1, -0.5]],
            dtype=DTYPE,
        )
    )


def test_cpu_fixed_plan_is_precision_qualified_before_execution():
    fixed, _, _ = temporal_objects()
    ir = lower_fixed_time(fixed)
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    request = fixed_request()
    certificate = qualify_backend_plan(
        ir,
        plan,
        (request,),
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    execution = execute_qualified(ir, plan, certificate, request)
    assert certificate.passed
    assert execution.estimated_tensor_dispatches == 1
    assert torch.allclose(execution.result.output, fixed.run(
        request.incident,
        initial_position=request.initial_position,
        initial_velocity=request.initial_velocity,
    ))
    assert not plan.fusion_claimed


def test_cpu_sampled_plan_packs_all_three_port_families():
    _, sampled, _ = temporal_objects()
    ir = lower_sampled_times(sampled)
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    request = fixed_request()
    certificate = qualify_backend_plan(
        ir, plan, (request,), absolute_tolerance=1e-12, relative_tolerance=1e-12
    )
    assert certificate.passed
    assert execute_qualified(ir, plan, certificate, request).result.output.shape == (3, 2)


def test_cpu_static_plan_matches_weighted_columns():
    source = SparseRelationSource.from_dense(
        torch.tensor([[1.0, 0.0], [0.5, -1.0], [0.0, 2.0]], dtype=DTYPE)
    )
    observation = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=DTYPE)
    compiled = compile_static_response(
        source, observation, torch.tensor([0, 1], dtype=torch.int64)
    )
    incidents = SparseIncidentBatch(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        torch.tensor([[0.3, -0.4], [0.7, 0.0]], dtype=DTYPE),
        torch.tensor([[True, True], [True, False]]),
    )
    prepared = compiled.prepare(incidents)
    request = BackendRequest(
        prepared.amplitudes,
        local_indices=prepared.local_indices,
        valid=prepared.valid,
    )
    ir = lower_static_response(compiled)
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    certificate = qualify_backend_plan(
        ir, plan, (request,), absolute_tolerance=0.0, relative_tolerance=0.0
    )
    assert certificate.passed
    assert torch.equal(
        execute_qualified(ir, plan, certificate, request).result.output,
        compiled.run_prepared(prepared),
    )


def test_cpu_grid_plan_qualifies_output_and_final_state():
    _, _, grid = temporal_objects()
    ir = lower_grid_recurrence(grid)
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    request = grid_request()
    certificate = qualify_backend_plan(
        ir, plan, (request,), absolute_tolerance=0.0, relative_tolerance=0.0
    )
    result = execute_qualified(ir, plan, certificate, request).result
    expected, state = grid.rollout(request.incident)
    assert certificate.passed
    assert torch.equal(result.output, expected)
    assert torch.equal(result.final_position, state.position)
    assert torch.equal(result.final_velocity, state.velocity)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is unavailable")
def test_mps_fp32_fixed_plan_receives_a_measured_certificate():
    fixed, _, _ = temporal_objects()
    ir = lower_fixed_time(fixed)
    plan = lower_backend_plan(ir, BackendKind.MPS, torch.float32)
    certificate = qualify_backend_plan(
        ir,
        plan,
        (fixed_request(),),
        absolute_tolerance=2e-6,
        relative_tolerance=2e-5,
    )
    assert certificate.passed
    assert certificate.maximum_absolute_error > 0
    assert certificate.values_compared == 2


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is unavailable")
def test_mps_fp32_grid_certificate_includes_accumulated_state_error():
    _, _, grid = temporal_objects()
    ir = lower_grid_recurrence(grid)
    plan = lower_backend_plan(ir, BackendKind.MPS, torch.float32)
    certificate = qualify_backend_plan(
        ir,
        plan,
        (grid_request(),),
        absolute_tolerance=2e-5,
        relative_tolerance=2e-4,
    )
    assert certificate.passed
    assert certificate.values_compared == 14


def test_unavailable_cuda_selects_deterministic_cpu_fallback():
    decision = choose_backend(BackendKind.CUDA, torch.float32)
    if backend_capability(BackendKind.CUDA).available:
        assert decision.selected_backend is BackendKind.CUDA
        assert not decision.fallback_used
    else:
        assert decision.selected_backend is BackendKind.CPU
        assert decision.fallback_used


def test_mps_float64_falls_back_to_cpu_even_when_mps_exists():
    decision = choose_backend(BackendKind.MPS, torch.float64)
    assert decision.selected_backend is BackendKind.CPU
    assert decision.selected_dtype is torch.float64
    assert decision.fallback_used


def test_disabling_fallback_rejects_an_unavailable_backend_or_dtype():
    with pytest.raises(RuntimeError):
        choose_backend(BackendKind.MPS, torch.float64, allow_fallback=False)


def test_plan_selection_is_deterministic():
    first = choose_backend(BackendKind.MPS, torch.float64)
    second = choose_backend(BackendKind.MPS, torch.float64)
    assert first == second


def test_certificate_cannot_authorize_a_different_artifact():
    first_fixed, _, _ = temporal_objects(time=0.5)
    second_fixed, _, _ = temporal_objects(time=0.6)
    first_ir = lower_fixed_time(first_fixed)
    second_ir = lower_fixed_time(second_fixed)
    plan = lower_backend_plan(first_ir, BackendKind.CPU, torch.float64)
    certificate = qualify_backend_plan(
        first_ir,
        plan,
        (fixed_request(),),
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    with pytest.raises(ValueError):
        execute_qualified(second_ir, plan, certificate, fixed_request())


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is unavailable")
def test_failed_precision_certificate_does_not_authorize_execution():
    fixed, _, _ = temporal_objects()
    ir = lower_fixed_time(fixed)
    plan = lower_backend_plan(ir, BackendKind.MPS, torch.float32)
    certificate = qualify_backend_plan(
        ir,
        plan,
        (fixed_request(),),
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    assert not certificate.passed
    with pytest.raises(ValueError):
        execute_qualified(ir, plan, certificate, fixed_request())
