from __future__ import annotations

import pytest
import torch

from information_field.native_response_kernels import qualify_native_plan
from information_field.profiled_response import prepare_backend_request
from information_field.response_backends import (
    BackendKind,
    BackendRequest,
    backend_capability,
    lower_backend_plan,
)
from information_field.wide_response_kernels import (
    WideCUDAResponseKernels,
    WideMPSResponseKernels,
    make_wide_grid_ir,
)


MPS_AVAILABLE = backend_capability(BackendKind.MPS).available
CUDA_AVAILABLE = backend_capability(BackendKind.CUDA).available


@pytest.fixture(scope="module")
def kernels():
    if not MPS_AVAILABLE:
        pytest.skip("MPS is unavailable")
    return WideMPSResponseKernels()


def test_wide_metal_recurrence_preserves_response_and_state(kernels):
    ir = make_wide_grid_ir()
    generator = torch.Generator().manual_seed(9302)
    request = BackendRequest(
        torch.randn((3, 3), generator=generator, dtype=torch.float64),
        state_position=torch.randn(1025, generator=generator, dtype=torch.float64),
        state_velocity=torch.randn(1025, generator=generator, dtype=torch.float64),
    )
    plan = lower_backend_plan(ir, BackendKind.MPS, torch.float32)
    prepared = prepare_backend_request(plan, request)
    actual = kernels.execute(plan, prepared)
    torch.mps.synchronize()
    expected = ir.execute(
        request.incident,
        state_position=request.state_position,
        state_velocity=request.state_velocity,
    )
    torch.testing.assert_close(
        actual.result.output.cpu().to(torch.float64), expected.output, atol=2e-5, rtol=2e-4
    )
    torch.testing.assert_close(
        actual.result.final_position.cpu().to(torch.float64),
        expected.final_position,
        atol=2e-5,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        actual.result.final_velocity.cpu().to(torch.float64),
        expected.final_velocity,
        atol=2e-5,
        rtol=2e-4,
    )
    assert actual.record.dispatches == 6
    assert not actual.record.fusion_certified
    assert actual.record.kernel_name == "modal_history_multidispatch"


def test_wide_metal_plan_receives_source_bound_precision_certificate(kernels):
    ir = make_wide_grid_ir()
    request = BackendRequest(torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64))
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
    assert certificate.source_digest == kernels.source_digest
    assert certificate.values_compared == 5 + 2 * 1025


def test_wide_executor_rejects_a_narrow_recurrence(kernels):
    ir = make_wide_grid_ir(modes=32)
    request = BackendRequest(torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64))
    plan = lower_backend_plan(ir, BackendKind.MPS, torch.float32)
    prepared = prepare_backend_request(plan, request)
    with pytest.raises(ValueError, match="one-threadgroup executor"):
        kernels.execute(plan, prepared)


@pytest.fixture(scope="module")
def cuda_kernels():
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA is unavailable")
    return WideCUDAResponseKernels()


def test_wide_cuda_recurrence_preserves_nonzero_state(cuda_kernels):
    ir = make_wide_grid_ir()
    generator = torch.Generator().manual_seed(9303)
    request = BackendRequest(
        torch.randn((3, 3), generator=generator, dtype=torch.float64),
        state_position=torch.randn(1025, generator=generator, dtype=torch.float64),
        state_velocity=torch.randn(1025, generator=generator, dtype=torch.float64),
    )
    plan = lower_backend_plan(ir, BackendKind.CUDA, torch.float32)
    prepared = prepare_backend_request(plan, request)
    actual = cuda_kernels.execute(plan, prepared)
    torch.cuda.synchronize()
    expected = ir.execute(
        request.incident,
        state_position=request.state_position,
        state_velocity=request.state_velocity,
    )
    torch.testing.assert_close(
        actual.result.output.cpu().to(torch.float64),
        expected.output,
        atol=2e-5,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        actual.result.final_position.cpu().to(torch.float64),
        expected.final_position,
        atol=2e-5,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        actual.result.final_velocity.cpu().to(torch.float64),
        expected.final_velocity,
        atol=2e-5,
        rtol=2e-4,
    )
    assert actual.record.dispatches == 6
    assert not actual.record.fusion_certified


def test_wide_cuda_zero_step_preserves_prior_state(cuda_kernels):
    ir = make_wide_grid_ir()
    position = torch.linspace(-0.2, 0.3, 1025, dtype=torch.float64)
    velocity = torch.linspace(0.4, -0.1, 1025, dtype=torch.float64)
    request = BackendRequest(
        torch.empty((0, 3), dtype=torch.float64),
        state_position=position,
        state_velocity=velocity,
    )
    plan = lower_backend_plan(ir, BackendKind.CUDA, torch.float32)
    prepared = prepare_backend_request(plan, request)
    actual = cuda_kernels.execute(plan, prepared)
    assert actual.record.dispatches == 0
    assert actual.result.output.shape == (0, 5)
    torch.testing.assert_close(
        actual.result.final_position.cpu().to(torch.float64),
        position,
        atol=2e-5,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        actual.result.final_velocity.cpu().to(torch.float64),
        velocity,
        atol=2e-5,
        rtol=2e-4,
    )
