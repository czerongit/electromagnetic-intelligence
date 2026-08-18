from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from information_field.causal_minimal import compile_minimal_realization
from information_field.native_response_kernels import (
    CPUResponseKernels,
    calibrate_executor,
    execute_selected,
    MPSResponseKernels,
    execute_qualified_native,
    qualify_native_plan,
)
from information_field.observable_response import (
    compile_fixed_time_green,
    compile_grid_recurrence,
    compile_observable_spectrum,
    compile_sampled_green,
)
from information_field.profiled_response import prepare_backend_request
from information_field.quotient_response import (
    SparseIncidentBatch,
    SparseRelationSource,
    compile_static_response,
)
from information_field.response_backends import (
    BackendKind,
    BackendRequest,
    backend_capability,
    lower_backend_plan,
)
from information_field.response_ir import (
    ResponseContract,
    lower_fixed_time,
    lower_grid_recurrence,
    lower_sampled_times,
    lower_static_response,
)


DTYPE = torch.float64
MPS_AVAILABLE = backend_capability(BackendKind.MPS).available


@pytest.fixture(scope="module")
def kernels() -> MPSResponseKernels:
    if not MPS_AVAILABLE:
        pytest.skip("MPS is unavailable")
    return MPSResponseKernels()


@pytest.fixture(scope="module")
def cpu_kernels() -> CPUResponseKernels:
    return CPUResponseKernels()


def temporal_ir(*, time: float = 0.73):
    operator = torch.diag(torch.tensor([1.0, 4.0, 9.0], dtype=DTYPE))
    incident = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]], dtype=DTYPE
    )
    observation = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=DTYPE
    )
    initial_position_port = torch.tensor([[1.0], [0.0], [0.5]], dtype=DTYPE)
    initial_velocity_port = torch.tensor([[0.0], [1.0], [-0.25]], dtype=DTYPE)
    realization = compile_minimal_realization(
        operator,
        incident,
        observation,
        initial_position_port=initial_position_port,
        initial_velocity_port=initial_velocity_port,
    )
    spectrum = compile_observable_spectrum(realization)
    return (
        lower_fixed_time(
            compile_fixed_time_green(realization, spectrum, time=time, mass=1.4)
        ),
        lower_sampled_times(
            compile_sampled_green(
                realization, spectrum, times=(0.2, 0.5, 1.1), mass=1.3
            )
        ),
        lower_grid_recurrence(
            compile_grid_recurrence(realization, step_size=0.17, mass=1.2)
        ),
    )


def fixed_request() -> BackendRequest:
    return BackendRequest(
        torch.tensor([0.2, -0.6], dtype=DTYPE),
        initial_position=torch.tensor([0.4], dtype=DTYPE),
        initial_velocity=torch.tensor([-0.3], dtype=DTYPE),
    )


def static_ir_and_request():
    source = SparseRelationSource.from_dense(
        torch.tensor([[1.0, 0.0], [0.5, -1.0], [0.0, 2.0]], dtype=DTYPE)
    )
    observation = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=DTYPE
    )
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
    return lower_static_response(compiled), request


def run_native(kernels, ir, request):
    plan = lower_backend_plan(ir, BackendKind.MPS, torch.float32)
    prepared = prepare_backend_request(plan, request)
    execution = kernels.execute(plan, prepared)
    torch.mps.synchronize()
    return plan, prepared, execution


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.detach().cpu().to(torch.float64),
        expected.detach().cpu().to(torch.float64),
        atol=2e-5,
        rtol=2e-4,
    )


def test_native_static_preserves_weighted_columns_and_empty_rows(kernels):
    ir, request = static_ir_and_request()
    _, _, execution = run_native(kernels, ir, request)
    expected = ir.execute(
        request.incident,
        local_indices=request.local_indices,
        valid=request.valid,
    ).output
    assert_close(execution.result.output, expected)
    assert torch.equal(
        execution.result.output[1].cpu(), torch.zeros(2, dtype=torch.float32)
    )
    assert execution.record.contract is ResponseContract.STATIC_COLUMNS
    assert execution.record.kernel_name == "weighted_columns"


@pytest.mark.parametrize("contract_index", [0, 1])
def test_native_packed_maps_preserve_fixed_and_sampled_contracts(
    kernels, contract_index
):
    ir = temporal_ir()[contract_index]
    request = fixed_request()
    _, _, execution = run_native(kernels, ir, request)
    expected = ir.execute(
        request.incident,
        initial_position=request.initial_position,
        initial_velocity=request.initial_velocity,
    ).output
    assert_close(execution.result.output, expected)
    assert execution.record.kernel_name == "packed_mv"
    assert execution.record.dispatches == 1
    assert execution.record.fusion_certified


def test_native_recurrence_preserves_outputs_and_final_state(kernels):
    ir = temporal_ir()[2]
    request = BackendRequest(
        torch.tensor(
            [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.8], [0.1, -0.5]],
            dtype=DTYPE,
        )
    )
    _, _, execution = run_native(kernels, ir, request)
    expected = ir.execute(request.incident)
    assert_close(execution.result.output, expected.output)
    assert_close(execution.result.final_position, expected.final_position)
    assert_close(execution.result.final_velocity, expected.final_velocity)
    assert execution.record.kernel_name == "modal_history_threadgroup"
    assert not execution.record.serial_device_execution
    assert execution.record.dispatches == 1


def test_native_recurrence_preserves_nonzero_prior_state(kernels):
    ir = temporal_ir()[2]
    request = BackendRequest(
        torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=DTYPE),
        state_position=torch.tensor([0.2, -0.5, 0.7], dtype=DTYPE),
        state_velocity=torch.tensor([-0.3, 0.8, 0.1], dtype=DTYPE),
    )
    _, _, execution = run_native(kernels, ir, request)
    expected = ir.execute(
        request.incident,
        state_position=request.state_position,
        state_velocity=request.state_velocity,
    )
    assert_close(execution.result.output, expected.output)
    assert_close(execution.result.final_position, expected.final_position)
    assert_close(execution.result.final_velocity, expected.final_velocity)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS is unavailable")
def test_serial_recurrence_remains_an_exact_negative_control():
    kernels = MPSResponseKernels(recurrence_mode="serial")
    ir = temporal_ir()[2]
    request = BackendRequest(
        torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=DTYPE)
    )
    _, _, execution = run_native(kernels, ir, request)
    expected = ir.execute(request.incident)
    assert_close(execution.result.output, expected.output)
    assert execution.record.kernel_name == "modal_history_serial"
    assert execution.record.serial_device_execution


def test_zero_step_recurrence_preserves_supplied_state_without_dispatch(kernels):
    ir = temporal_ir()[2]
    request = BackendRequest(
        torch.empty((0, 2), dtype=DTYPE),
        state_position=torch.tensor([0.2, -0.5, 0.7], dtype=DTYPE),
        state_velocity=torch.tensor([-0.3, 0.8, 0.1], dtype=DTYPE),
    )
    _, _, execution = run_native(kernels, ir, request)
    assert execution.result.output.shape == (0, 2)
    assert_close(execution.result.final_position, request.state_position)
    assert_close(execution.result.final_velocity, request.state_velocity)
    assert execution.record.dispatches == 0
    assert not execution.record.fusion_certified


def test_prepared_request_cannot_cross_native_plan_boundaries(kernels):
    first = temporal_ir(time=0.5)[0]
    second = temporal_ir(time=0.6)[0]
    first_plan = lower_backend_plan(first, BackendKind.MPS, torch.float32)
    second_plan = lower_backend_plan(second, BackendKind.MPS, torch.float32)
    prepared = prepare_backend_request(first_plan, fixed_request())
    with pytest.raises(ValueError, match="another backend plan"):
        kernels.execute(second_plan, prepared)


def test_native_record_binds_exact_shader_source(kernels):
    ir = temporal_ir()[0]
    _, _, execution = run_native(kernels, ir, fixed_request())
    source = (
        Path(__file__).parents[1]
        / "src"
        / "information_field"
        / "native"
        / "response.metal"
    ).read_bytes()
    assert execution.record.source_digest == hashlib.sha256(source).hexdigest()
    assert execution.record.plan_digest


def test_native_kernel_rejects_non_mps_plan(kernels):
    ir = temporal_ir()[0]
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float32)
    prepared = prepare_backend_request(plan, fixed_request())
    with pytest.raises(ValueError, match="MPS FP32"):
        kernels.execute(plan, prepared)


@pytest.mark.parametrize("contract_index", [0, 1, 2])
def test_native_precision_certificate_binds_plan_shader_and_contract(
    kernels, contract_index
):
    ir = temporal_ir()[contract_index]
    request = (
        fixed_request()
        if contract_index < 2
        else BackendRequest(
            torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=DTYPE),
            state_position=torch.tensor([0.2, -0.5, 0.7], dtype=DTYPE),
            state_velocity=torch.tensor([-0.3, 0.8, 0.1], dtype=DTYPE),
        )
    )
    plan = lower_backend_plan(ir, BackendKind.MPS, torch.float32)
    certificate = qualify_native_plan(
        ir,
        plan,
        kernels,
        (request,),
        absolute_tolerance=2e-5,
        relative_tolerance=2e-4,
    )
    execution = execute_qualified_native(
        ir, plan, kernels, certificate, request
    )
    assert certificate.passed
    assert certificate.source_digest == kernels.source_digest
    assert certificate.contract is ir.contract
    assert certificate.values_compared > 0
    assert execution.record.plan_digest == certificate.plan_digest


def test_static_native_precision_certificate_covers_empty_rows(kernels):
    ir, request = static_ir_and_request()
    plan = lower_backend_plan(ir, BackendKind.MPS, torch.float32)
    certificate = qualify_native_plan(
        ir,
        plan,
        kernels,
        (request,),
        absolute_tolerance=2e-5,
        relative_tolerance=2e-4,
    )
    assert certificate.passed
    assert certificate.values_compared == 6


def test_native_certificate_cannot_authorize_changed_response_plan(kernels):
    first = temporal_ir(time=0.5)[0]
    second = temporal_ir(time=0.6)[0]
    first_plan = lower_backend_plan(first, BackendKind.MPS, torch.float32)
    second_plan = lower_backend_plan(second, BackendKind.MPS, torch.float32)
    request = fixed_request()
    certificate = qualify_native_plan(
        first,
        first_plan,
        kernels,
        (request,),
        absolute_tolerance=2e-5,
        relative_tolerance=2e-4,
    )
    with pytest.raises(ValueError, match="invalid for this execution"):
        execute_qualified_native(
            second, second_plan, kernels, certificate, request
        )


@pytest.mark.parametrize("contract_index", [0, 1, 2])
def test_native_cpu_temporal_contracts_are_precision_qualified(
    cpu_kernels, contract_index
):
    ir = temporal_ir()[contract_index]
    request = (
        fixed_request()
        if contract_index < 2
        else BackendRequest(
            torch.tensor(
                [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.8]], dtype=DTYPE
            ),
            state_position=torch.tensor([0.2, -0.5, 0.7], dtype=DTYPE),
            state_velocity=torch.tensor([-0.3, 0.8, 0.1], dtype=DTYPE),
        )
    )
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    certificate = qualify_native_plan(
        ir,
        plan,
        cpu_kernels,
        (request,),
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    execution = execute_qualified_native(
        ir, plan, cpu_kernels, certificate, request
    )
    expected = ir.execute(
        request.incident,
        initial_position=request.initial_position,
        initial_velocity=request.initial_velocity,
        state_position=request.state_position,
        state_velocity=request.state_velocity,
    )
    assert certificate.passed
    assert execution.record.backend is BackendKind.CPU
    assert execution.record.fusion_certified
    torch.testing.assert_close(execution.result.output, expected.output)
    if expected.final_position is not None:
        torch.testing.assert_close(
            execution.result.final_position, expected.final_position
        )
        torch.testing.assert_close(
            execution.result.final_velocity, expected.final_velocity
        )


def test_native_cpu_static_contract_preserves_empty_rows(cpu_kernels):
    ir, request = static_ir_and_request()
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    certificate = qualify_native_plan(
        ir,
        plan,
        cpu_kernels,
        (request,),
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    execution = execute_qualified_native(
        ir, plan, cpu_kernels, certificate, request
    )
    expected = ir.execute(
        request.incident,
        local_indices=request.local_indices,
        valid=request.valid,
    )
    assert certificate.passed
    assert execution.record.kernel_name == "weighted_columns_cpu"
    assert torch.equal(execution.result.output[1], torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(execution.result.output, expected.output)


def test_executor_selection_is_bound_to_the_exact_workload_shape(cpu_kernels):
    ir = temporal_ir()[2]
    plan = lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    request = BackendRequest(
        torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=DTYPE)
    )
    certificate = calibrate_executor(
        ir,
        plan,
        cpu_kernels,
        request,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
        warmup=1,
        iterations=3,
    )
    execution = execute_selected(
        ir, plan, cpu_kernels, certificate, request
    )
    expected = ir.execute(request.incident)
    torch.testing.assert_close(execution.result.output, expected.output)
    changed_shape = BackendRequest(
        torch.tensor(
            [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.8]], dtype=DTYPE
        )
    )
    with pytest.raises(ValueError, match="invalid for this response workload"):
        execute_selected(
            ir, plan, cpu_kernels, certificate, changed_shape
        )
