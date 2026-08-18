from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .source import SparseRelationSource, Tensor


@dataclass(frozen=True)
class CausalState:
    position: Tensor
    velocity: Tensor


@dataclass(frozen=True)
class SparseExchangeState:
    whitened_position: Tensor
    scaled_velocity: Tensor
    relation_strength: Tensor


def _spectral_oscillator(
    stiffness: Tensor,
    force: Tensor,
    position: Tensor,
    velocity: Tensor,
    *,
    mass: float,
    time: float,
    zero_tolerance: float,
) -> tuple[Tensor, Tensor]:
    values, vectors = torch.linalg.eigh(0.5 * (stiffness + stiffness.T))
    values = torch.clamp(values, min=0)
    x0 = vectors.T @ position
    v0 = vectors.T @ velocity
    f = vectors.T @ force
    omega = torch.sqrt(values / mass)
    active = values > zero_tolerance
    x = x0 + time * v0 + 0.5 * time * time * f / mass
    v = v0 + time * f / mass
    if bool(active.any()):
        w = omega[active]
        cosine = torch.cos(w * time)
        sine = torch.sin(w * time)
        equilibrium = f[active] / values[active]
        x[active] = (
            cosine * x0[active]
            + sine * v0[active] / w
            + (1 - cosine) * equilibrium
        )
        v[active] = (
            -w * sine * x0[active]
            + cosine * v0[active]
            + sine * f[active] / (mass * w)
        )
    return vectors @ x, vectors @ v


def dense_causal_oracle(
    source: SparseRelationSource,
    relation_incident: Tensor,
    initial: CausalState,
    *,
    time: float,
    mass: float = 1.0,
    calibration: float = 1.0,
    zero_tolerance: float = 1e-12,
) -> CausalState:
    if mass <= 0 or calibration <= 0:
        raise ValueError("mass and calibration must be positive")
    d = source.whitened_dense()
    stiffness = calibration * d @ d.T
    whitened_incident = torch.sqrt(source.relation_metric) * relation_incident
    force = d @ whitened_incident
    x0 = torch.sqrt(source.quantity_metric) * initial.position
    v0 = torch.sqrt(source.quantity_metric) * initial.velocity
    position, velocity = _spectral_oscillator(
        stiffness,
        force,
        x0,
        v0,
        mass=mass,
        time=time,
        zero_tolerance=zero_tolerance,
    )
    scale = torch.sqrt(source.quantity_metric)
    return CausalState(position / scale, velocity / scale)


@dataclass(frozen=True)
class ExactModalResponse:
    source_digest: str
    left_modes: Tensor
    singular_values: Tensor
    right_modes: Tensor
    quantity_scale: Tensor
    relation_scale: Tensor
    discarded_norm: float

    @property
    def rank(self) -> int:
        return self.singular_values.numel()

    @property
    def retained_scalars(self) -> int:
        return (
            self.left_modes.numel()
            + self.singular_values.numel()
            + self.right_modes.numel()
            + self.quantity_scale.numel()
            + self.relation_scale.numel()
        )

    def evolve_constant(
        self,
        relation_incident: Tensor,
        initial: CausalState,
        *,
        time: float,
        mass: float = 1.0,
        calibration: float = 1.0,
    ) -> CausalState:
        if mass <= 0 or calibration <= 0:
            raise ValueError("mass and calibration must be positive")
        x0 = self.quantity_scale * initial.position
        velocity0 = self.quantity_scale * initial.velocity
        z0 = self.left_modes.T @ x0
        zdot0 = self.left_modes.T @ velocity0
        null_position = x0 - self.left_modes @ z0
        null_velocity = velocity0 - self.left_modes @ zdot0
        relation = self.relation_scale * relation_incident
        coordinates = self.right_modes.T @ relation
        force = self.singular_values * coordinates
        omega = math.sqrt(calibration / mass) * self.singular_values
        cosine = torch.cos(omega * time)
        sine = torch.sin(omega * time)
        stiffness = calibration * self.singular_values.square()
        equilibrium = force / stiffness
        z = cosine * z0 + sine * zdot0 / omega + (1 - cosine) * equilibrium
        zdot = (
            -omega * sine * z0
            + cosine * zdot0
            + sine * force / (mass * omega)
        )
        x = self.left_modes @ z + null_position + time * null_velocity
        velocity = self.left_modes @ zdot + null_velocity
        return CausalState(x / self.quantity_scale, velocity / self.quantity_scale)


def compile_exact_modal(
    source: SparseRelationSource,
    *,
    tolerance: float | None = None,
) -> ExactModalResponse:
    dense = source.whitened_dense()
    left, singular, right_transpose = torch.linalg.svd(dense, full_matrices=False)
    if tolerance is None:
        tolerance = (
            max(dense.shape)
            * torch.finfo(dense.dtype).eps
            * float(singular.max().item() if singular.numel() else 0.0)
        )
    keep = singular > tolerance
    discarded = float(singular[~keep].max().item()) if bool((~keep).any()) else 0.0
    return ExactModalResponse(
        source.digest,
        left[:, keep],
        singular[keep],
        right_transpose[keep].T,
        torch.sqrt(source.quantity_metric),
        torch.sqrt(source.relation_metric),
        discarded,
    )


def _rk4_step(state: tuple[Tensor, ...], derivative, step: float) -> tuple[Tensor, ...]:
    k1 = derivative(state)
    k2 = derivative(tuple(value + 0.5 * step * delta for value, delta in zip(state, k1)))
    k3 = derivative(tuple(value + 0.5 * step * delta for value, delta in zip(state, k2)))
    k4 = derivative(tuple(value + step * delta for value, delta in zip(state, k3)))
    return tuple(
        value + step * (a + 2 * b + 2 * c + d) / 6
        for value, a, b, c, d in zip(state, k1, k2, k3, k4)
    )


def sparse_first_order_evolve(
    source: SparseRelationSource,
    relation_incident: Tensor,
    initial: CausalState,
    *,
    time: float,
    steps: int,
    mass: float = 1.0,
    calibration: float = 1.0,
) -> CausalState:
    if steps < 1 or mass <= 0 or calibration <= 0:
        raise ValueError("steps, mass, and calibration must be positive")
    q = torch.sqrt(source.relation_metric) * relation_incident
    force = source.whitened_apply(q)
    x = torch.sqrt(source.quantity_metric) * initial.position
    v = math.sqrt(mass) * torch.sqrt(source.quantity_metric) * initial.velocity
    u = torch.sqrt(torch.tensor(calibration, dtype=x.dtype, device=x.device)) * source.whitened_adjoint(x)
    root_ratio = (calibration / mass) ** 0.5
    root_mass = mass ** 0.5

    def derivative(state: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        position, scaled_velocity, relation_strength = state
        return (
            scaled_velocity / root_mass,
            -root_ratio * source.whitened_apply(relation_strength) + force / root_mass,
            root_ratio * source.whitened_adjoint(scaled_velocity),
        )

    state: tuple[Tensor, Tensor, Tensor] = (x, v, u)
    step = time / steps
    for _ in range(steps):
        state = _rk4_step(state, derivative, step)
    x, v, _ = state
    scale = torch.sqrt(source.quantity_metric)
    return CausalState(x / scale, v / (root_mass * scale))


def dense_second_order_rk4(
    source: SparseRelationSource,
    relation_incident: Tensor,
    initial: CausalState,
    *,
    time: float,
    steps: int,
    mass: float = 1.0,
    calibration: float = 1.0,
) -> CausalState:
    d = source.whitened_dense()
    stiffness = calibration * d @ d.T
    force = d @ (torch.sqrt(source.relation_metric) * relation_incident)
    x = torch.sqrt(source.quantity_metric) * initial.position
    velocity = torch.sqrt(source.quantity_metric) * initial.velocity

    def derivative(state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        position, speed = state
        return speed, (force - stiffness @ position) / mass

    state: tuple[Tensor, Tensor] = (x, velocity)
    step = time / steps
    for _ in range(steps):
        state = _rk4_step(state, derivative, step)
    x, velocity = state
    scale = torch.sqrt(source.quantity_metric)
    return CausalState(x / scale, velocity / scale)
