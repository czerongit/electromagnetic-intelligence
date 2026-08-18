from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from information_field.causal_minimal import CausalMinimalRealization

from .spectral import ObservableSpectrum


Tensor = torch.Tensor


def _coefficient_vectors(
    eigenvalues: Tensor,
    *,
    time: float,
    mass: float,
    tolerance: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    omega = torch.sqrt(torch.clamp(eigenvalues, min=0) / mass)
    active = eigenvalues > tolerance
    cosine = torch.cos(omega * time)
    sine_over_omega = torch.full_like(eigenvalues, time)
    step = torch.full_like(eigenvalues, 0.5 * time * time / mass)
    impulse = torch.full_like(eigenvalues, time / mass)
    if bool(active.any()):
        sine = torch.sin(omega[active] * time)
        sine_over_omega[active] = sine / omega[active]
        step[active] = (1.0 - cosine[active]) / eigenvalues[active]
        impulse[active] = sine / (mass * omega[active])
    return cosine, sine_over_omega, step, impulse


@dataclass(frozen=True)
class FixedTimeGreenMap:
    time: float
    mass: float
    kind: str
    incident_map: Tensor
    initial_position_map: Tensor
    initial_velocity_map: Tensor
    realization_digest: str

    @property
    def retained_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.incident_map,
                self.initial_position_map,
                self.initial_velocity_map,
            )
        )

    def run(
        self,
        incident: Tensor,
        *,
        initial_position: Tensor | None = None,
        initial_velocity: Tensor | None = None,
    ) -> Tensor:
        if incident.ndim != 1 or incident.shape[0] != self.incident_map.shape[1]:
            raise ValueError("incident has the wrong port dimension")
        result = self.incident_map @ incident
        if initial_position is not None:
            if initial_position.ndim != 1 or initial_position.shape[0] != self.initial_position_map.shape[1]:
                raise ValueError("initial position has the wrong port dimension")
            result = result + self.initial_position_map @ initial_position
        if initial_velocity is not None:
            if initial_velocity.ndim != 1 or initial_velocity.shape[0] != self.initial_velocity_map.shape[1]:
                raise ValueError("initial velocity has the wrong port dimension")
            result = result + self.initial_velocity_map @ initial_velocity
        return result

    def run_prepared(self, incident: Tensor) -> Tensor:
        return self.incident_map @ incident

    def run_basis(self, index: int, amplitude: float = 1.0) -> Tensor:
        if index < 0 or index >= self.incident_map.shape[1]:
            raise ValueError("basis index is outside the incident port")
        return amplitude * self.incident_map[:, index]

    def run_batch(self, incidents: Tensor) -> Tensor:
        if incidents.ndim != 2 or incidents.shape[1] != self.incident_map.shape[1]:
            raise ValueError("incident batch has the wrong port dimension")
        return incidents @ self.incident_map.T

    def is_valid_for(self, realization: CausalMinimalRealization) -> bool:
        return self.realization_digest == realization.certificate.execution_digest

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "FixedTimeGreenMap":
        target_dtype = self.incident_map.dtype if dtype is None else dtype
        return replace(
            self,
            incident_map=self.incident_map.to(device=device, dtype=target_dtype),
            initial_position_map=self.initial_position_map.to(device=device, dtype=target_dtype),
            initial_velocity_map=self.initial_velocity_map.to(device=device, dtype=target_dtype),
        )


@dataclass(frozen=True)
class SampledGreenFamily:
    times: Tensor
    incident_maps: Tensor
    initial_position_maps: Tensor
    initial_velocity_maps: Tensor
    mass: float
    kind: str
    realization_digest: str

    def run(self, incident: Tensor) -> Tensor:
        if incident.ndim != 1 or incident.shape[0] != self.incident_maps.shape[2]:
            raise ValueError("incident has the wrong port dimension")
        return torch.einsum("tzi,i->tz", self.incident_maps, incident)

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "SampledGreenFamily":
        target_dtype = self.incident_maps.dtype if dtype is None else dtype
        return replace(
            self,
            times=self.times.to(device=device, dtype=target_dtype),
            incident_maps=self.incident_maps.to(device=device, dtype=target_dtype),
            initial_position_maps=self.initial_position_maps.to(device=device, dtype=target_dtype),
            initial_velocity_maps=self.initial_velocity_maps.to(device=device, dtype=target_dtype),
        )


def compile_fixed_time_green(
    realization: CausalMinimalRealization,
    spectrum: ObservableSpectrum,
    *,
    time: float,
    mass: float = 1.0,
    kind: str = "constant",
) -> FixedTimeGreenMap:
    spectrum.assert_valid_for(realization)
    if kind not in {"constant", "impulse"}:
        raise ValueError("fixed-time kind must be constant or impulse")
    if time < 0 or mass <= 0:
        raise ValueError("time must be nonnegative and mass must be positive")
    incident_map = (
        spectrum.step_map(time, mass=mass)
        if kind == "constant"
        else spectrum.impulse_map(time, mass=mass)
    )
    cosine, sine_over_omega, _, _ = _coefficient_vectors(
        realization.eigenvalues,
        time=time,
        mass=mass,
        tolerance=realization.certificate.tolerance,
    )
    position_map = (
        realization.modal_observation
        @ (cosine[:, None] * realization.modal_initial_position_port)
    )
    velocity_map = (
        realization.modal_observation
        @ (sine_over_omega[:, None] * realization.modal_initial_velocity_port)
    )
    return FixedTimeGreenMap(
        time,
        mass,
        kind,
        incident_map,
        position_map,
        velocity_map,
        realization.certificate.execution_digest,
    )


def compile_sampled_green(
    realization: CausalMinimalRealization,
    spectrum: ObservableSpectrum,
    *,
    times: tuple[float, ...],
    mass: float = 1.0,
    kind: str = "constant",
) -> SampledGreenFamily:
    if not times or any(time < 0 for time in times):
        raise ValueError("sampled times must be a nonempty nonnegative sequence")
    maps = [
        compile_fixed_time_green(
            realization, spectrum, time=time, mass=mass, kind=kind
        )
        for time in times
    ]
    reference = realization.operator
    return SampledGreenFamily(
        torch.tensor(times, dtype=reference.dtype, device=reference.device),
        torch.stack([item.incident_map for item in maps]),
        torch.stack([item.initial_position_map for item in maps]),
        torch.stack([item.initial_velocity_map for item in maps]),
        mass,
        kind,
        realization.certificate.execution_digest,
    )
