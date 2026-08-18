from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from information_field.causal_minimal import CausalMinimalRealization, block_krylov_basis


Tensor = torch.Tensor


@dataclass(frozen=True)
class OnlineState:
    position: Tensor
    velocity: Tensor


@dataclass(frozen=True)
class OnlineMinimalityCertificate:
    second_order_dimension: int
    first_order_degree: int
    controllable_rank: int
    observable_rank: int
    continuous_time_minimal: bool
    sampled_minimality_claimed: bool
    realization_digest: str


@dataclass(frozen=True)
class ExactGridRecurrence:
    step_size: float
    mass: float
    cosine: Tensor
    sine_over_omega: Tensor
    negative_omega_sine: Tensor
    force_position: Tensor
    force_velocity: Tensor
    modal_incident: Tensor
    modal_observation: Tensor
    modal_initial_position: Tensor
    modal_initial_velocity: Tensor
    certificate: OnlineMinimalityCertificate

    @property
    def state_dimension(self) -> int:
        return 2 * self.cosine.numel()

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "ExactGridRecurrence":
        target_dtype = self.cosine.dtype if dtype is None else dtype
        changes = {
            name: value.to(device=device, dtype=target_dtype)
            for name, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        return replace(self, **changes)

    def zero_state(self) -> OnlineState:
        return OnlineState(torch.zeros_like(self.cosine), torch.zeros_like(self.cosine))

    def initial_state(
        self,
        *,
        position: Tensor | None = None,
        velocity: Tensor | None = None,
    ) -> OnlineState:
        x = torch.zeros_like(self.cosine)
        v = torch.zeros_like(self.cosine)
        if position is not None:
            if position.ndim != 1 or position.shape[0] != self.modal_initial_position.shape[1]:
                raise ValueError("initial position has the wrong port dimension")
            x = self.modal_initial_position @ position
        if velocity is not None:
            if velocity.ndim != 1 or velocity.shape[0] != self.modal_initial_velocity.shape[1]:
                raise ValueError("initial velocity has the wrong port dimension")
            v = self.modal_initial_velocity @ velocity
        return OnlineState(x, v)

    def observe(self, state: OnlineState) -> Tensor:
        return self.modal_observation @ state.position

    def step(self, state: OnlineState, incident: Tensor) -> OnlineState:
        if incident.ndim != 1 or incident.shape[0] != self.modal_incident.shape[1]:
            raise ValueError("incident has the wrong port dimension")
        force = self.modal_incident @ incident
        position = (
            self.cosine * state.position
            + self.sine_over_omega * state.velocity
            + self.force_position * force
        )
        velocity = (
            self.negative_omega_sine * state.position
            + self.cosine * state.velocity
            + self.force_velocity * force
        )
        return OnlineState(position, velocity)

    def rollout(
        self,
        incidents: Tensor,
        *,
        initial: OnlineState | None = None,
    ) -> tuple[Tensor, OnlineState]:
        if incidents.ndim != 2 or incidents.shape[1] != self.modal_incident.shape[1]:
            raise ValueError("incident history has the wrong port dimension")
        state = self.zero_state() if initial is None else initial
        outputs = []
        for incident in incidents:
            state = self.step(state, incident)
            outputs.append(self.observe(state))
        if outputs:
            return torch.stack(outputs), state
        return torch.empty(
            (0, self.modal_observation.shape[0]),
            dtype=self.modal_observation.dtype,
            device=self.modal_observation.device,
        ), state


def _first_order_data(
    realization: CausalMinimalRealization,
    mass: float,
) -> tuple[Tensor, Tensor, Tensor]:
    rank = realization.state_dimension
    zero = torch.zeros(
        (rank, rank), dtype=realization.operator.dtype, device=realization.operator.device
    )
    identity = torch.eye(
        rank, dtype=realization.operator.dtype, device=realization.operator.device
    )
    diagonal = torch.diag(realization.eigenvalues)
    generator = torch.cat(
        (
            torch.cat((zero, identity), dim=1),
            torch.cat((-diagonal / mass, zero), dim=1),
        ),
        dim=0,
    )
    incident = torch.cat(
        (
            torch.zeros_like(realization.modal_incident_port),
            realization.modal_incident_port / mass,
        ),
        dim=0,
    )
    observation = torch.cat(
        (
            realization.modal_observation,
            torch.zeros_like(realization.modal_observation),
        ),
        dim=1,
    )
    return generator, incident, observation


def compile_grid_recurrence(
    realization: CausalMinimalRealization,
    *,
    step_size: float,
    mass: float = 1.0,
) -> ExactGridRecurrence:
    if step_size <= 0 or mass <= 0:
        raise ValueError("step size and mass must be positive")
    eigenvalues = realization.eigenvalues
    omega = torch.sqrt(torch.clamp(eigenvalues, min=0) / mass)
    active = eigenvalues > realization.certificate.tolerance
    cosine = torch.cos(omega * step_size)
    sine_over_omega = torch.full_like(eigenvalues, step_size)
    negative_omega_sine = torch.zeros_like(eigenvalues)
    force_position = torch.full_like(eigenvalues, 0.5 * step_size * step_size / mass)
    force_velocity = torch.full_like(eigenvalues, step_size / mass)
    if bool(active.any()):
        sine = torch.sin(omega[active] * step_size)
        sine_over_omega[active] = sine / omega[active]
        negative_omega_sine[active] = -omega[active] * sine
        force_position[active] = (1.0 - cosine[active]) / eigenvalues[active]
        force_velocity[active] = sine / (mass * omega[active])

    generator, incident, observation = _first_order_data(realization, mass)
    tolerance = realization.certificate.tolerance
    reachable = block_krylov_basis(generator, incident, tolerance=tolerance)
    observable = block_krylov_basis(generator.T, observation.T, tolerance=tolerance)
    degree = generator.shape[0]
    certificate = OnlineMinimalityCertificate(
        realization.state_dimension,
        degree,
        reachable.shape[1],
        observable.shape[1],
        reachable.shape[1] == degree and observable.shape[1] == degree,
        False,
        realization.certificate.execution_digest,
    )
    if not certificate.continuous_time_minimal:
        raise ValueError(
            "reachable-and-observable second-order carrier did not produce a minimal continuous-time online realization"
        )
    return ExactGridRecurrence(
        step_size,
        mass,
        cosine,
        sine_over_omega,
        negative_omega_sine,
        force_position,
        force_velocity,
        realization.modal_incident_port,
        realization.modal_observation,
        realization.modal_initial_position_port,
        realization.modal_initial_velocity_port,
        certificate,
    )
