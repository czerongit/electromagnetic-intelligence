import math

import pytest
import torch

from information_field.causal_minimal import (
    CausalMinimalRealization,
    compile_minimal_realization,
    exact_second_order_state,
)
from information_field.observable_response import (
    TemporalWorkload,
    WorkloadKind,
    compile_fixed_time_green,
    compile_grid_recurrence,
    compile_observable_spectrum,
    compile_sampled_green,
)


DTYPE = torch.float64


def realization(
    eigenvalues,
    incident=None,
    observation=None,
    *,
    position_port=None,
    velocity_port=None,
) -> CausalMinimalRealization:
    values = torch.tensor(eigenvalues, dtype=DTYPE)
    n = values.numel()
    return compile_minimal_realization(
        torch.diag(values),
        torch.eye(n, dtype=DTYPE) if incident is None else torch.tensor(incident, dtype=DTYPE),
        torch.eye(n, dtype=DTYPE) if observation is None else torch.tensor(observation, dtype=DTYPE),
        initial_position_port=(
            None if position_port is None else torch.tensor(position_port, dtype=DTYPE)
        ),
        initial_velocity_port=(
            None if velocity_port is None else torch.tensor(velocity_port, dtype=DTYPE)
        ),
    )


def test_degenerate_modes_form_one_residue_per_frequency_and_preserve_moments():
    field = realization([1.0, 1.0, 4.0, 4.0])
    spectrum = compile_observable_spectrum(field)
    assert spectrum.certificate.distinct_frequencies == 2
    assert spectrum.certificate.residue_rank_sum == 4
    assert [item.multiplicity for item in spectrum.residues] == [2, 2]
    assert [item.rank for item in spectrum.residues] == [2, 2]
    for order in range(4):
        assert torch.allclose(spectrum.moment(order), field.transfer_jet(order), atol=1e-12)
    assert spectrum.certificate.maximum_moment_residual == 0.0


def test_nearly_equal_frequencies_are_not_merged_without_certificate():
    field = realization([1.0, 1.0 + 1e-6])
    spectrum = compile_observable_spectrum(field, eigenvalue_tolerance=1e-10)
    assert spectrum.certificate.distinct_frequencies == 2
    assert spectrum.certificate.maximum_eigenvalue_spread == 0.0


def test_fixed_constant_green_map_matches_modal_field_response():
    field = realization(
        [0.0, 1.0, 4.0],
        incident=[[1.0, 0.0], [0.5, 1.0], [0.0, -0.75]],
        observation=[[1.0, -0.5, 0.25], [0.0, 0.4, 1.0]],
    )
    spectrum = compile_observable_spectrum(field)
    compiled = compile_fixed_time_green(
        field, spectrum, time=0.7, mass=1.3, kind="constant"
    )
    incident = torch.tensor([0.3, -0.8], dtype=DTYPE)
    assert torch.allclose(
        compiled.run(incident),
        field.respond_prepared_zero_past_constant(incident, time=0.7, mass=1.3),
        atol=1e-10,
    )
    assert torch.allclose(compiled.run_basis(1, -0.8), -0.8 * compiled.incident_map[:, 1])
    batch = torch.stack((incident, -incident))
    assert torch.allclose(compiled.run_batch(batch), torch.stack((compiled.run(incident), compiled.run(-incident))))


def test_fixed_impulse_green_map_matches_direct_spectral_evaluation():
    field = realization(
        [0.0, 2.0, 5.0],
        incident=[[1.0], [0.5], [-0.25]],
        observation=[[0.5, 1.0, -0.75]],
    )
    spectrum = compile_observable_spectrum(field)
    compiled = compile_fixed_time_green(
        field, spectrum, time=0.4, mass=1.7, kind="impulse"
    )
    incident = torch.tensor([0.8], dtype=DTYPE)
    omega = torch.sqrt(field.eigenvalues / 1.7)
    coefficient = torch.full_like(omega, 0.4 / 1.7)
    active = field.eigenvalues > field.certificate.tolerance
    coefficient[active] = torch.sin(omega[active] * 0.4) / (1.7 * omega[active])
    expected = field.modal_observation @ (
        coefficient * (field.modal_incident_port @ incident)
    )
    assert torch.allclose(compiled.run(incident), expected, atol=1e-10)


def test_fixed_map_includes_declared_initial_position_and_velocity_ports():
    field = realization(
        [1.0, 4.0],
        incident=[[1.0], [0.5]],
        observation=[[1.0, -0.25]],
        position_port=[[1.0], [0.0]],
        velocity_port=[[0.0], [1.0]],
    )
    spectrum = compile_observable_spectrum(field)
    compiled = compile_fixed_time_green(field, spectrum, time=0.6, mass=1.2)
    u = torch.tensor([0.4], dtype=DTYPE)
    x0 = torch.tensor([0.7], dtype=DTYPE)
    v0 = torch.tensor([-0.3], dtype=DTYPE)
    expected = field.respond_constant(
        u,
        time=0.6,
        mass=1.2,
        initial_position=x0,
        initial_velocity=v0,
    )
    assert torch.allclose(
        compiled.run(u, initial_position=x0, initial_velocity=v0),
        expected,
        atol=1e-10,
    )


def test_sampled_family_equals_independently_compiled_fixed_maps():
    field = realization([1.0, 2.0, 3.0])
    spectrum = compile_observable_spectrum(field)
    times = (0.0, 0.2, 0.7, 1.1)
    family = compile_sampled_green(field, spectrum, times=times, mass=0.9)
    incident = torch.tensor([0.1, -0.2, 0.5], dtype=DTYPE)
    expected = torch.stack(
        [
            compile_fixed_time_green(
                field, spectrum, time=time, mass=0.9
            ).run(incident)
            for time in times
        ]
    )
    assert torch.allclose(family.run(incident), expected, atol=1e-12)


def test_regular_grid_recurrence_matches_exact_piecewise_constant_oracle():
    field = realization(
        [0.0, 1.0, 3.0],
        incident=[[1.0, 0.0], [0.5, 1.0], [0.25, -0.4]],
        observation=[[1.0, -0.2, 0.5], [0.0, 0.3, 1.0]],
    )
    recurrence = compile_grid_recurrence(field, step_size=0.08, mass=1.4)
    inputs = torch.tensor(
        [[0.2, -0.1], [0.0, 0.7], [-0.3, 0.4], [0.5, 0.0]],
        dtype=DTYPE,
    )
    actual, _ = recurrence.rollout(inputs)

    position = torch.zeros(field.state_dimension, dtype=DTYPE)
    velocity = torch.zeros_like(position)
    expected = []
    for incident in inputs:
        position, velocity = exact_second_order_state(
            field.operator,
            field.incident_port @ incident,
            position,
            velocity,
            time=0.08,
            mass=1.4,
            tolerance=field.certificate.tolerance,
        )
        expected.append(field.observation @ position)
    assert torch.allclose(actual, torch.stack(expected), atol=1e-10)
    assert recurrence.certificate.continuous_time_minimal
    assert recurrence.certificate.first_order_degree == 2 * field.state_dimension
    assert not recurrence.certificate.sampled_minimality_claimed


def test_regular_grid_initial_ports_are_explicit():
    field = realization(
        [1.0, 2.0],
        position_port=[[1.0], [0.0]],
        velocity_port=[[0.0], [1.0]],
    )
    recurrence = compile_grid_recurrence(field, step_size=0.1)
    initial = recurrence.initial_state(
        position=torch.tensor([0.5], dtype=DTYPE),
        velocity=torch.tensor([-0.2], dtype=DTYPE),
    )
    assert torch.count_nonzero(initial.position) == 1
    assert torch.count_nonzero(initial.velocity) == 1
    with pytest.raises(ValueError, match="wrong port dimension"):
        recurrence.initial_state(position=torch.ones(2, dtype=DTYPE))


def test_compiled_spectrum_and_green_map_invalidate_after_realization_change():
    first = realization([1.0, 2.0])
    second = realization([1.0, 3.0])
    spectrum = compile_observable_spectrum(first)
    fixed = compile_fixed_time_green(first, spectrum, time=0.5)
    assert spectrum.is_valid_for(first)
    assert fixed.is_valid_for(first)
    assert not spectrum.is_valid_for(second)
    assert not fixed.is_valid_for(second)
    with pytest.raises(ValueError, match="recompile"):
        spectrum.assert_valid_for(second)


def test_workload_contracts_do_not_cross_temporal_boundaries():
    fixed = TemporalWorkload(
        WorkloadKind.FIXED_TIME_CONSTANT, times=(0.5,), zero_past=True
    )
    fixed.validate()
    assert fixed.source_fixed_compilation_supported
    assert not fixed.requires_state

    grid = TemporalWorkload(
        WorkloadKind.REGULAR_GRID_PIECEWISE_CONSTANT, step=0.1
    )
    grid.validate()
    assert grid.requires_state

    changing = TemporalWorkload(WorkloadKind.TIME_DEPENDENT_GEOMETRY)
    changing.validate()
    assert not changing.source_fixed_compilation_supported
    assert changing.requires_state

    with pytest.raises(ValueError, match="exactly one"):
        TemporalWorkload(WorkloadKind.FIXED_TIME_IMPULSE).validate()
    with pytest.raises(ValueError, match="positive step"):
        TemporalWorkload(WorkloadKind.REGULAR_GRID_PIECEWISE_CONSTANT).validate()
