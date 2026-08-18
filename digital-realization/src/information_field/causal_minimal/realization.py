from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Iterable

import torch


Tensor = torch.Tensor


def _matrix(value: Tensor, *, name: str, rows: int | None = None) -> Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if rows is not None and value.shape[0] != rows:
        raise ValueError(f"{name} has the wrong state dimension")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")
    return value


def _empty_port(reference: Tensor, rows: int) -> Tensor:
    return torch.empty((rows, 0), dtype=reference.dtype, device=reference.device)


def _digest_tensors(*values: Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        cpu = value.detach().contiguous().cpu()
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _default_tolerance(*values: Tensor) -> float:
    if not values:
        return 1e-12
    eps = max(torch.finfo(value.dtype).eps for value in values)
    dimension = max(max(value.shape, default=1) for value in values)
    scale = max(1.0, *(float(torch.linalg.matrix_norm(value).item()) for value in values))
    return 64.0 * dimension * eps * scale


def _orthogonal_block(candidate: Tensor, basis: Tensor, tolerance: float) -> Tensor:
    if candidate.shape[1] == 0:
        return candidate
    residual = candidate
    if basis.shape[1]:
        # Two passes suppress loss of orthogonality in block Arnoldi.
        residual = residual - basis @ (basis.T @ residual)
        residual = residual - basis @ (basis.T @ residual)
    left, singular, _ = torch.linalg.svd(residual, full_matrices=False)
    if singular.numel() == 0:
        return residual[:, :0]
    keep = singular > tolerance
    return left[:, keep]


def block_krylov_basis(
    operator: Tensor,
    seed: Tensor,
    *,
    tolerance: float,
) -> Tensor:
    """Return an orthonormal basis of span{operator**k seed}.

    No response values are inspected. Termination follows invariant-subspace
    closure or the ambient dimension bound.
    """

    operator = _matrix(operator, name="operator")
    seed = _matrix(seed, name="seed", rows=operator.shape[0])
    if operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be square")
    basis = operator[:, :0]
    frontier = _orthogonal_block(seed, basis, tolerance)
    for _ in range(operator.shape[0]):
        if frontier.shape[1] == 0:
            break
        basis = torch.cat((basis, frontier), dim=1)
        if basis.shape[1] == operator.shape[0]:
            break
        frontier = _orthogonal_block(operator @ frontier, basis, tolerance)
    return basis


def _relative_residual(actual: Tensor, expected: Tensor) -> float:
    difference = float(torch.linalg.matrix_norm(actual - expected).item())
    scale = max(1.0, float(torch.linalg.matrix_norm(expected).item()))
    return difference / scale


def _maximum_markov_residual(
    operator: Tensor,
    ports: Iterable[Tensor],
    observation: Tensor,
    reduced_operator: Tensor,
    reduced_ports: Iterable[Tensor],
    reduced_observation: Tensor,
) -> float:
    full_ports = tuple(ports)
    small_ports = tuple(reduced_ports)
    full_powers = list(full_ports)
    small_powers = list(small_ports)
    maximum = 0.0
    scale = max(1.0, float(torch.linalg.matrix_norm(operator).item()))
    full_step = operator / scale
    small_step = reduced_operator / scale
    for _ in range(operator.shape[0]):
        for full, small in zip(full_powers, small_powers):
            maximum = max(
                maximum,
                _relative_residual(
                    reduced_observation @ small,
                    observation @ full,
                ),
            )
        full_powers = [full_step @ value for value in full_powers]
        small_powers = [small_step @ value for value in small_powers]
    return maximum


@dataclass(frozen=True)
class ReductionCertificate:
    ambient_dimension: int
    reachable_dimension: int
    minimal_dimension: int
    incident_width: int
    observation_width: int
    initial_position_width: int
    initial_velocity_width: int
    tolerance: float
    symmetry_residual: float
    positivity_floor: float
    reachable_invariance_residual: float
    observable_invariance_residual: float
    maximum_markov_residual: float
    execution_digest: str
    structural_digest: str

    @property
    def removed_dimensions(self) -> int:
        return self.ambient_dimension - self.minimal_dimension

    @property
    def retained_fraction(self) -> float:
        if self.ambient_dimension == 0:
            return 0.0
        return self.minimal_dimension / self.ambient_dimension


@dataclass(frozen=True)
class CausalMinimalRealization:
    operator: Tensor
    incident_port: Tensor
    observation: Tensor
    lift: Tensor
    initial_position_port: Tensor
    initial_velocity_port: Tensor
    eigenvalues: Tensor
    eigenmodes: Tensor
    modal_incident_port: Tensor
    modal_observation: Tensor
    modal_initial_position_port: Tensor
    modal_initial_velocity_port: Tensor
    certificate: ReductionCertificate

    @property
    def state_dimension(self) -> int:
        return self.operator.shape[0]

    @property
    def retained_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.operator,
                self.incident_port,
                self.observation,
                self.initial_position_port,
                self.initial_velocity_port,
                self.eigenvalues,
                self.eigenmodes,
            )
        )

    @property
    def compiled_bytes_with_lift(self) -> int:
        return self.retained_bytes + self.lift.numel() * self.lift.element_size()

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "CausalMinimalRealization":
        target_dtype = self.operator.dtype if dtype is None else dtype
        changes = {
            name: value.to(device=device, dtype=target_dtype)
            for name, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        return replace(self, **changes)

    @property
    def execution_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.eigenvalues,
                self.modal_incident_port,
                self.modal_observation,
                self.modal_initial_position_port,
                self.modal_initial_velocity_port,
            )
        )

    def is_valid_for(
        self,
        operator: Tensor,
        incident_port: Tensor,
        observation: Tensor,
        *,
        initial_position_port: Tensor | None = None,
        initial_velocity_port: Tensor | None = None,
    ) -> bool:
        n = operator.shape[0]
        position = (
            _empty_port(operator, n)
            if initial_position_port is None
            else initial_position_port
        )
        velocity = (
            _empty_port(operator, n)
            if initial_velocity_port is None
            else initial_velocity_port
        )
        return self.certificate.execution_digest == _digest_tensors(
            operator, incident_port, observation, position, velocity
        )

    def is_valid_for_relation_field(
        self,
        source,
        relation_port: Tensor,
        observation: Tensor,
        *,
        calibration: float = 1.0,
    ) -> bool:
        return self.certificate.structural_digest == _relation_compilation_digest(
            source, relation_port, observation, calibration
        )

    def assert_valid_for(
        self,
        operator: Tensor,
        incident_port: Tensor,
        observation: Tensor,
        *,
        initial_position_port: Tensor | None = None,
        initial_velocity_port: Tensor | None = None,
    ) -> None:
        if not self.is_valid_for(
            operator,
            incident_port,
            observation,
            initial_position_port=initial_position_port,
            initial_velocity_port=initial_velocity_port,
        ):
            raise ValueError(
                "field operator, metric realization, incident port, observation, "
                "or initial-state port changed; recompile the causal realization"
            )

    def transfer_jet(self, order: int) -> Tensor:
        if order < 0:
            raise ValueError("jet order must be nonnegative")
        value = self.incident_port
        for _ in range(order):
            value = self.operator @ value
        return self.observation @ value

    def respond_constant(
        self,
        incident: Tensor,
        *,
        time: float,
        mass: float = 1.0,
        initial_position: Tensor | None = None,
        initial_velocity: Tensor | None = None,
    ) -> Tensor:
        if time < 0 or mass <= 0:
            raise ValueError("time must be nonnegative and mass must be positive")
        incident = _coordinate_vector(
            incident, self.incident_port.shape[1], "incident"
        )
        position_coordinates = _optional_coordinates(
            initial_position,
            self.initial_position_port.shape[1],
            self.operator,
            "initial position",
        )
        velocity_coordinates = _optional_coordinates(
            initial_velocity,
            self.initial_velocity_port.shape[1],
            self.operator,
            "initial velocity",
        )
        modal_force = self.modal_incident_port @ incident
        modal_position = self.modal_initial_position_port @ position_coordinates
        modal_velocity = self.modal_initial_velocity_port @ velocity_coordinates
        position, _ = _diagonal_second_order_state(
            self.eigenvalues,
            modal_force,
            modal_position,
            modal_velocity,
            time=time,
            mass=mass,
            tolerance=self.certificate.tolerance,
        )
        return self.modal_observation @ position

    def respond_prepared_zero_past_constant(
        self,
        incident: Tensor,
        *,
        time: float,
        mass: float = 1.0,
    ) -> Tensor:
        """Fast path after port admission and zero-past validation."""

        force = self.modal_incident_port @ incident
        displacement = 0.5 * time * time * force / mass
        active = self.eigenvalues > self.certificate.tolerance
        if bool(active.any()):
            stiffness = self.eigenvalues[active]
            omega = torch.sqrt(stiffness / mass)
            displacement[active] = (
                (1.0 - torch.cos(omega * time)) * force[active] / stiffness
            )
        return self.modal_observation @ displacement


def _coordinate_vector(value: Tensor, width: int, name: str) -> Tensor:
    if value.ndim != 1 or value.shape[0] != width:
        raise ValueError(f"{name} has the wrong port dimension")
    return value


def _optional_coordinates(
    value: Tensor | None,
    width: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    if value is None:
        return torch.zeros(width, dtype=reference.dtype, device=reference.device)
    return _coordinate_vector(value, width, name)


def exact_second_order_state(
    operator: Tensor,
    force: Tensor,
    position: Tensor,
    velocity: Tensor,
    *,
    time: float,
    mass: float,
    tolerance: float,
) -> tuple[Tensor, Tensor]:
    if operator.shape[0] == 0:
        return position.clone(), velocity.clone()
    eigenvalues, modes = torch.linalg.eigh(0.5 * (operator + operator.T))
    eigenvalues = torch.clamp(eigenvalues, min=0)
    return _modal_second_order_state(
        eigenvalues,
        modes,
        force,
        position,
        velocity,
        time=time,
        mass=mass,
        tolerance=tolerance,
    )


def _modal_second_order_state(
    eigenvalues: Tensor,
    modes: Tensor,
    force: Tensor,
    position: Tensor,
    velocity: Tensor,
    *,
    time: float,
    mass: float,
    tolerance: float,
) -> tuple[Tensor, Tensor]:
    if eigenvalues.numel() == 0:
        return position.clone(), velocity.clone()
    x0 = modes.T @ position
    v0 = modes.T @ velocity
    forcing = modes.T @ force
    x, v = _diagonal_second_order_state(
        eigenvalues,
        forcing,
        x0,
        v0,
        time=time,
        mass=mass,
        tolerance=tolerance,
    )
    return modes @ x, modes @ v


def _diagonal_second_order_state(
    eigenvalues: Tensor,
    forcing: Tensor,
    position: Tensor,
    velocity: Tensor,
    *,
    time: float,
    mass: float,
    tolerance: float,
) -> tuple[Tensor, Tensor]:
    if eigenvalues.numel() == 0:
        return position.clone(), velocity.clone()
    x0 = position
    v0 = velocity
    active = eigenvalues > tolerance
    x = x0 + time * v0 + 0.5 * time * time * forcing / mass
    v = v0 + time * forcing / mass
    if bool(active.any()):
        stiffness = eigenvalues[active]
        omega = torch.sqrt(stiffness / mass)
        cosine = torch.cos(omega * time)
        sine = torch.sin(omega * time)
        equilibrium = forcing[active] / stiffness
        x[active] = (
            cosine * x0[active]
            + sine * v0[active] / omega
            + (1.0 - cosine) * equilibrium
        )
        v[active] = (
            -omega * sine * x0[active]
            + cosine * v0[active]
            + sine * forcing[active] / (mass * omega)
        )
    return x, v


def full_constant_response(
    operator: Tensor,
    incident_port: Tensor,
    observation: Tensor,
    incident: Tensor,
    *,
    time: float,
    mass: float = 1.0,
    initial_position_port: Tensor | None = None,
    initial_velocity_port: Tensor | None = None,
    initial_position: Tensor | None = None,
    initial_velocity: Tensor | None = None,
    tolerance: float | None = None,
) -> Tensor:
    n = operator.shape[0]
    position_port = (
        _empty_port(operator, n)
        if initial_position_port is None
        else initial_position_port
    )
    velocity_port = (
        _empty_port(operator, n)
        if initial_velocity_port is None
        else initial_velocity_port
    )
    x_coordinates = _optional_coordinates(
        initial_position, position_port.shape[1], operator, "initial position"
    )
    v_coordinates = _optional_coordinates(
        initial_velocity, velocity_port.shape[1], operator, "initial velocity"
    )
    force = incident_port @ _coordinate_vector(
        incident, incident_port.shape[1], "incident"
    )
    position, _ = exact_second_order_state(
        operator,
        force,
        position_port @ x_coordinates,
        velocity_port @ v_coordinates,
        time=time,
        mass=mass,
        tolerance=(
            _default_tolerance(operator, incident_port, observation)
            if tolerance is None
            else tolerance
        ),
    )
    return observation @ position


def compile_minimal_realization(
    operator: Tensor,
    incident_port: Tensor,
    observation: Tensor,
    *,
    initial_position_port: Tensor | None = None,
    initial_velocity_port: Tensor | None = None,
    tolerance: float | None = None,
) -> CausalMinimalRealization:
    operator = _matrix(operator, name="operator")
    if operator.shape[0] != operator.shape[1]:
        raise ValueError("operator must be square")
    n = operator.shape[0]
    incident_port = _matrix(incident_port, name="incident port", rows=n)
    observation = _matrix(observation, name="observation")
    if observation.shape[1] != n:
        raise ValueError("observation has the wrong state dimension")
    if operator.device != incident_port.device or operator.device != observation.device:
        raise ValueError("operator and ports must be on one device")
    if operator.dtype != incident_port.dtype or operator.dtype != observation.dtype:
        raise ValueError("operator and ports must use one dtype")
    position_port = (
        _empty_port(operator, n)
        if initial_position_port is None
        else _matrix(initial_position_port, name="initial position port", rows=n)
    )
    velocity_port = (
        _empty_port(operator, n)
        if initial_velocity_port is None
        else _matrix(initial_velocity_port, name="initial velocity port", rows=n)
    )
    if position_port.device != operator.device or velocity_port.device != operator.device:
        raise ValueError("initial-state ports must be on the operator device")
    if position_port.dtype != operator.dtype or velocity_port.dtype != operator.dtype:
        raise ValueError("initial-state ports must use the operator dtype")
    tolerance = (
        _default_tolerance(
            operator, incident_port, observation, position_port, velocity_port
        )
        if tolerance is None
        else float(tolerance)
    )
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    symmetry = _relative_residual(operator, operator.T)
    symmetric = 0.5 * (operator + operator.T)
    eigen_floor = (
        float(torch.linalg.eigvalsh(symmetric).min().item()) if n else 0.0
    )
    if symmetry > tolerance:
        raise ValueError("intrinsic operator must be self-adjoint")
    if eigen_floor < -tolerance:
        raise ValueError("intrinsic operator must be nonnegative")

    seed = torch.cat((incident_port, position_port, velocity_port), dim=1)
    reachable = block_krylov_basis(symmetric, seed, tolerance=tolerance)
    reachable_operator = reachable.T @ symmetric @ reachable
    reachable_incident = reachable.T @ incident_port
    reachable_position = reachable.T @ position_port
    reachable_velocity = reachable.T @ velocity_port
    reachable_observation = observation @ reachable

    observable = block_krylov_basis(
        reachable_operator,
        reachable_observation.T,
        tolerance=tolerance,
    )
    lift = reachable @ observable
    reduced_operator = observable.T @ reachable_operator @ observable
    reduced_incident = observable.T @ reachable_incident
    reduced_position = observable.T @ reachable_position
    reduced_velocity = observable.T @ reachable_velocity
    reduced_observation = reachable_observation @ observable

    reachable_residual = 0.0
    if reachable.shape[1]:
        reachable_residual = _relative_residual(
            reachable @ (reachable.T @ symmetric @ reachable),
            symmetric @ reachable,
        )
    observable_residual = 0.0
    if observable.shape[1]:
        observable_residual = _relative_residual(
            observable @ (observable.T @ reachable_operator @ observable),
            reachable_operator @ observable,
        )
    markov_residual = _maximum_markov_residual(
        symmetric,
        (incident_port, position_port, velocity_port),
        observation,
        reduced_operator,
        (reduced_incident, reduced_position, reduced_velocity),
        reduced_observation,
    )
    certificate = ReductionCertificate(
        ambient_dimension=n,
        reachable_dimension=reachable.shape[1],
        minimal_dimension=observable.shape[1],
        incident_width=incident_port.shape[1],
        observation_width=observation.shape[0],
        initial_position_width=position_port.shape[1],
        initial_velocity_width=velocity_port.shape[1],
        tolerance=tolerance,
        symmetry_residual=symmetry,
        positivity_floor=eigen_floor,
        reachable_invariance_residual=reachable_residual,
        observable_invariance_residual=observable_residual,
        maximum_markov_residual=markov_residual,
        execution_digest=_digest_tensors(
            operator, incident_port, observation, position_port, velocity_port
        ),
        structural_digest=_digest_tensors(
            operator, incident_port, observation, position_port, velocity_port
        ),
    )
    eigenvalues, eigenmodes = torch.linalg.eigh(
        0.5 * (reduced_operator + reduced_operator.T)
    )
    eigenvalues = torch.clamp(eigenvalues, min=0)
    modal_incident = eigenmodes.T @ reduced_incident
    modal_observation = reduced_observation @ eigenmodes
    modal_position = eigenmodes.T @ reduced_position
    modal_velocity = eigenmodes.T @ reduced_velocity
    return CausalMinimalRealization(
        reduced_operator,
        reduced_incident,
        reduced_observation,
        lift,
        reduced_position,
        reduced_velocity,
        eigenvalues,
        eigenmodes,
        modal_incident,
        modal_observation,
        modal_position,
        modal_velocity,
        certificate,
    )


def compile_relation_field(
    source,
    relation_port: Tensor,
    observation: Tensor,
    *,
    calibration: float = 1.0,
    tolerance: float | None = None,
) -> CausalMinimalRealization:
    """Compile m a'' + c D D* a = D J u in whitened coordinates."""

    if calibration <= 0:
        raise ValueError("calibration must be positive")
    relation_port = _matrix(
        relation_port, name="relation port", rows=source.relation_dim
    ).to(device=source.device, dtype=source.dtype)
    observation = _matrix(observation, name="observation")
    if observation.shape[1] != source.quantity_dim:
        raise ValueError("observation has the wrong quantity dimension")
    d = source.whitened_dense()
    operator = calibration * d @ d.T
    incident = d @ (torch.sqrt(source.relation_metric)[:, None] * relation_port)
    whitened_observation = (
        observation.to(device=source.device, dtype=source.dtype)
        / torch.sqrt(source.quantity_metric)[None, :]
    )
    compiled = compile_minimal_realization(
        operator,
        incident,
        whitened_observation,
        tolerance=tolerance,
    )
    return replace(
        compiled,
        certificate=replace(
            compiled.certificate,
            structural_digest=_relation_compilation_digest(
                source, relation_port, observation, calibration
            ),
        ),
    )


def _relation_compilation_digest(
    source,
    relation_port: Tensor,
    observation: Tensor,
    calibration: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(source.digest.encode())
    digest.update(repr(float(calibration)).encode())
    for value in (relation_port, observation):
        cpu = value.detach().contiguous().cpu()
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()
