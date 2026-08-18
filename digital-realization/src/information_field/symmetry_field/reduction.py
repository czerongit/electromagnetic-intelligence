from __future__ import annotations

from dataclasses import dataclass
import hashlib
import torch

from information_field.causal_minimal import CausalMinimalRealization, compile_minimal_realization
from information_field.matrix_free_field import (
    FactorizedIntrinsicOperator,
    MatrixFreeCompilation,
    compile_matrix_free_relation_field,
    matrix_free_block_krylov,
)
from information_field.matrix_free_field.compiler import OperatorCounter
from information_field.quotient_response import SparseRelationSource


Tensor = torch.Tensor


def _relative_norm(value: Tensor, reference: Tensor | None = None) -> float:
    magnitude = float(torch.linalg.matrix_norm(value).item())
    scale = 1.0 if reference is None else max(
        1.0, float(torch.linalg.matrix_norm(reference).item())
    )
    return magnitude / scale


def _digest_tensor(digest, value: Tensor) -> None:
    cpu = value.detach().contiguous().cpu()
    digest.update(str(tuple(cpu.shape)).encode())
    digest.update(str(cpu.dtype).encode())
    digest.update(cpu.numpy().tobytes())


def _compilation_digest(
    source: SparseRelationSource,
    relation_port: Tensor,
    observation: Tensor,
    quantity_permutation: Tensor,
    relation_permutation: Tensor,
    quantity_sign: Tensor,
    relation_sign: Tensor,
    calibration: float,
    position: Tensor,
    velocity: Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"involutive-source-symmetry-v1")
    digest.update(source.digest.encode())
    digest.update(repr(float(calibration)).encode())
    for value in (
        relation_port,
        observation,
        quantity_permutation,
        relation_permutation,
        quantity_sign,
        relation_sign,
        position,
        velocity,
    ):
        _digest_tensor(digest, value)
    return digest.hexdigest()


def _validate_signed_involution(
    permutation: Tensor,
    sign: Tensor,
    dimension: int,
    name: str,
) -> None:
    if permutation.shape != (dimension,) or permutation.dtype != torch.int64:
        raise ValueError(f"{name} permutation must be an int64 coordinate vector")
    if sign.shape != (dimension,) or not sign.is_floating_point():
        raise ValueError(f"{name} signs must be a floating coordinate vector")
    expected = torch.arange(dimension, dtype=torch.int64, device=permutation.device)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError(f"{name} permutation must be bijective")
    if not torch.equal(permutation[permutation], expected):
        raise ValueError(f"{name} permutation must be involutive")
    if not torch.allclose(torch.abs(sign), torch.ones_like(sign), atol=0, rtol=0):
        raise ValueError(f"{name} signs must equal plus or minus one")
    if not torch.allclose(sign * sign[permutation], torch.ones_like(sign), atol=0, rtol=0):
        raise ValueError(f"{name} signed permutation must square to identity")


def apply_signed_permutation(values: Tensor, permutation: Tensor, sign: Tensor) -> Tensor:
    if values.ndim != 2 or values.shape[0] != permutation.numel():
        raise ValueError("signed-permutation input has the wrong carrier dimension")
    result = torch.empty_like(values)
    result[permutation] = sign[:, None] * values
    return result


def _sparse_intertwining_residual(
    source: SparseRelationSource,
    quantity_permutation: Tensor,
    relation_permutation: Tensor,
    quantity_sign: Tensor,
    relation_sign: Tensor,
) -> float:
    rows = source.rows.detach().cpu().tolist()
    columns = source.columns.detach().cpu().tolist()
    whitened_values = (
        source.values
        * torch.sqrt(source.quantity_metric[source.rows])
        / torch.sqrt(source.relation_metric[source.columns])
    ).detach().cpu().tolist()
    hp = quantity_permutation.detach().cpu().tolist()
    gp = relation_permutation.detach().cpu().tolist()
    hs = quantity_sign.detach().cpu().tolist()
    gs = relation_sign.detach().cpu().tolist()
    left: dict[tuple[int, int], float] = {}
    right: dict[tuple[int, int], float] = {}
    for row, column, value in zip(rows, columns, whitened_values):
        left_key = (hp[row], column)
        right_key = (row, gp[column])
        left[left_key] = left.get(left_key, 0.0) + hs[row] * value
        right[right_key] = right.get(right_key, 0.0) + gs[gp[column]] * value
    keys = set(left) | set(right)
    difference = sum((left.get(key, 0.0) - right.get(key, 0.0)) ** 2 for key in keys)
    scale = max(1.0, sum(right.get(key, 0.0) ** 2 for key in keys))
    return (difference / scale) ** 0.5


def _subspace_invariance_residual(
    port: Tensor,
    permutation: Tensor,
    sign: Tensor,
    tolerance: float,
) -> float:
    if port.shape[1] == 0:
        return 0.0
    left, singular, _ = torch.linalg.svd(port, full_matrices=False)
    basis = left[:, singular > tolerance]
    transformed = apply_signed_permutation(port, permutation, sign)
    if basis.shape[1]:
        transformed = transformed - basis @ (basis.T @ transformed)
    return _relative_norm(transformed, port)


def _project_parity(
    values: Tensor,
    permutation: Tensor,
    sign: Tensor,
    parity: int,
) -> Tensor:
    return 0.5 * (
        values + parity * apply_signed_permutation(values, permutation, sign)
    )


@dataclass(frozen=True)
class SymmetryCertificate:
    source_intertwining_residual: float
    incident_invariance_residual: float
    observation_invariance_residual: float
    position_invariance_residual: float
    velocity_invariance_residual: float
    sector_invariance_residual: float
    tolerance: float
    used_symmetry: bool
    fallback_reason: str | None
    structural_digest: str


@dataclass(frozen=True)
class SymmetryAccounting:
    ambient_dimension: int
    sector_count: int
    sector_reachable_dimensions: tuple[int, ...]
    sector_minimal_dimensions: tuple[int, ...]
    total_minimal_dimension: int
    factorized_operator_applications: int
    maximum_block_width: int


@dataclass(frozen=True)
class SymmetrySector:
    parity: int
    reachable_basis: Tensor
    realization: CausalMinimalRealization


@dataclass(frozen=True)
class SymmetryCompilation:
    sectors: tuple[SymmetrySector, ...]
    fallback: MatrixFreeCompilation | None
    certificate: SymmetryCertificate
    accounting: SymmetryAccounting
    output_dimension: int

    def respond_constant(
        self,
        incident: Tensor,
        *,
        time: float,
        mass: float = 1.0,
        initial_position: Tensor | None = None,
        initial_velocity: Tensor | None = None,
    ) -> Tensor:
        if self.fallback is not None:
            return self.fallback.realization.respond_constant(
                incident,
                time=time,
                mass=mass,
                initial_position=initial_position,
                initial_velocity=initial_velocity,
            )
        result = torch.zeros(
            self.output_dimension, dtype=incident.dtype, device=incident.device
        )
        for sector in self.sectors:
            result = result + sector.realization.respond_constant(
                incident,
                time=time,
                mass=mass,
                initial_position=initial_position,
                initial_velocity=initial_velocity,
            )
        return result

    def is_valid_for(
        self,
        source: SparseRelationSource,
        relation_port: Tensor,
        observation: Tensor,
        quantity_permutation: Tensor,
        relation_permutation: Tensor,
        quantity_sign: Tensor,
        relation_sign: Tensor,
        *,
        calibration: float = 1.0,
        initial_position_port: Tensor | None = None,
        initial_velocity_port: Tensor | None = None,
    ) -> bool:
        n = source.quantity_dim
        position = (
            torch.empty((n, 0), dtype=source.dtype, device=source.device)
            if initial_position_port is None
            else initial_position_port
        )
        velocity = (
            torch.empty((n, 0), dtype=source.dtype, device=source.device)
            if initial_velocity_port is None
            else initial_velocity_port
        )
        current = _compilation_digest(
            source,
            relation_port,
            observation,
            quantity_permutation,
            relation_permutation,
            quantity_sign,
            relation_sign,
            calibration,
            position,
            velocity,
        )
        return current == self.certificate.structural_digest


def _fallback_compilation(
    source: SparseRelationSource,
    relation_port: Tensor,
    observation: Tensor,
    *,
    calibration: float,
    position: Tensor,
    velocity: Tensor,
    certificate: SymmetryCertificate,
) -> SymmetryCompilation:
    fallback = compile_matrix_free_relation_field(
        source,
        relation_port,
        observation,
        calibration=calibration,
        initial_position_port=position if position.shape[1] else None,
        initial_velocity_port=velocity if velocity.shape[1] else None,
        tolerance=certificate.tolerance,
    )
    accounting = SymmetryAccounting(
        source.quantity_dim,
        0,
        (),
        (),
        fallback.accounting.minimal_dimension,
        fallback.accounting.factorized_operator_applications,
        fallback.accounting.maximum_block_width,
    )
    return SymmetryCompilation((), fallback, certificate, accounting, observation.shape[0])


def compile_involutive_symmetry(
    source: SparseRelationSource,
    relation_port: Tensor,
    observation: Tensor,
    quantity_permutation: Tensor,
    relation_permutation: Tensor,
    *,
    quantity_sign: Tensor | None = None,
    relation_sign: Tensor | None = None,
    calibration: float = 1.0,
    initial_position_port: Tensor | None = None,
    initial_velocity_port: Tensor | None = None,
    tolerance: float = 1e-10,
) -> SymmetryCompilation:
    if calibration <= 0 or tolerance <= 0:
        raise ValueError("calibration and tolerance must be positive")
    if relation_port.ndim != 2 or relation_port.shape[0] != source.relation_dim:
        raise ValueError("relation port has the wrong relation dimension")
    if observation.ndim != 2 or observation.shape[1] != source.quantity_dim:
        raise ValueError("observation has the wrong quantity dimension")
    device, dtype = source.device, source.dtype
    hp = quantity_permutation.to(device=device)
    gp = relation_permutation.to(device=device)
    hs = (
        torch.ones(source.quantity_dim, dtype=dtype, device=device)
        if quantity_sign is None
        else quantity_sign.to(device=device, dtype=dtype)
    )
    gs = (
        torch.ones(source.relation_dim, dtype=dtype, device=device)
        if relation_sign is None
        else relation_sign.to(device=device, dtype=dtype)
    )
    _validate_signed_involution(hp, hs, source.quantity_dim, "quantity")
    _validate_signed_involution(gp, gs, source.relation_dim, "relation")
    relation_port = relation_port.to(device=device, dtype=dtype)
    observation = observation.to(device=device, dtype=dtype)
    n = source.quantity_dim
    position = (
        torch.empty((n, 0), dtype=dtype, device=device)
        if initial_position_port is None
        else initial_position_port.to(device=device, dtype=dtype)
    )
    velocity = (
        torch.empty((n, 0), dtype=dtype, device=device)
        if initial_velocity_port is None
        else initial_velocity_port.to(device=device, dtype=dtype)
    )
    if position.ndim != 2 or position.shape[0] != n:
        raise ValueError("initial position port has the wrong quantity dimension")
    if velocity.ndim != 2 or velocity.shape[0] != n:
        raise ValueError("initial velocity port has the wrong quantity dimension")

    whitened_relation_port = torch.sqrt(source.relation_metric)[:, None] * relation_port
    incident = source.whitened_apply(whitened_relation_port.T).T
    whitened_observation = observation / torch.sqrt(source.quantity_metric)[None, :]
    whitened_position = torch.sqrt(source.quantity_metric)[:, None] * position
    whitened_velocity = torch.sqrt(source.quantity_metric)[:, None] * velocity
    source_residual = _sparse_intertwining_residual(source, hp, gp, hs, gs)
    incident_residual = _subspace_invariance_residual(incident, hp, hs, tolerance)
    observation_residual = _subspace_invariance_residual(
        whitened_observation.T, hp, hs, tolerance
    )
    position_residual = _subspace_invariance_residual(
        whitened_position, hp, hs, tolerance
    )
    velocity_residual = _subspace_invariance_residual(
        whitened_velocity, hp, hs, tolerance
    )
    digest = _compilation_digest(
        source,
        relation_port,
        observation,
        hp,
        gp,
        hs,
        gs,
        calibration,
        position,
        velocity,
    )
    failures = []
    for label, residual in (
        ("source automorphism", source_residual),
        ("incident port", incident_residual),
        ("readout", observation_residual),
        ("initial-position port", position_residual),
        ("initial-velocity port", velocity_residual),
    ):
        if residual > tolerance:
            failures.append(label)
    if failures:
        certificate = SymmetryCertificate(
            source_residual,
            incident_residual,
            observation_residual,
            position_residual,
            velocity_residual,
            0.0,
            tolerance,
            False,
            "symmetry does not preserve " + ", ".join(failures),
            digest,
        )
        return _fallback_compilation(
            source,
            relation_port,
            observation,
            calibration=calibration,
            position=position,
            velocity=velocity,
            certificate=certificate,
        )

    counter = OperatorCounter()
    apply_operator = FactorizedIntrinsicOperator(source, calibration, counter)
    seed_ports = (incident, whitened_position, whitened_velocity)
    sectors = []
    maximum_sector_residual = 0.0
    for parity in (1, -1):
        fixed = hp == torch.arange(n, dtype=torch.int64, device=device)
        trace = int(torch.sum(hs[fixed]).item())
        sector_dimension = (n + parity * trace) // 2
        projected_ports = tuple(
            _project_parity(value, hp, hs, parity) for value in seed_ports
        )
        seed = torch.cat(projected_ports, dim=1)
        def apply_sector(values: Tensor) -> Tensor:
            return _project_parity(
                apply_operator(values), hp, hs, parity
            )
        reachable = matrix_free_block_krylov(
            apply_sector,
            seed,
            ambient_dimension=sector_dimension,
            tolerance=tolerance,
        )
        if reachable.shape[1] == 0:
            continue
        transformed = apply_signed_permutation(reachable, hp, hs)
        maximum_sector_residual = max(
            maximum_sector_residual,
            _relative_norm(transformed - parity * reachable, reachable),
        )
        applied = apply_sector(reachable)
        reduced_operator = reachable.T @ applied
        reduced_operator = 0.5 * (reduced_operator + reduced_operator.T)
        realization = compile_minimal_realization(
            reduced_operator,
            reachable.T @ projected_ports[0],
            whitened_observation @ reachable,
            initial_position_port=reachable.T @ projected_ports[1],
            initial_velocity_port=reachable.T @ projected_ports[2],
            tolerance=tolerance,
        )
        sectors.append(SymmetrySector(parity, reachable, realization))

    certificate = SymmetryCertificate(
        source_residual,
        incident_residual,
        observation_residual,
        position_residual,
        velocity_residual,
        maximum_sector_residual,
        tolerance,
        True,
        None,
        digest,
    )
    accounting = SymmetryAccounting(
        n,
        len(sectors),
        tuple(sector.reachable_basis.shape[1] for sector in sectors),
        tuple(sector.realization.state_dimension for sector in sectors),
        sum(sector.realization.state_dimension for sector in sectors),
        counter.applications,
        counter.maximum_block_width,
    )
    return SymmetryCompilation(
        tuple(sectors), None, certificate, accounting, observation.shape[0]
    )


@dataclass(frozen=True)
class NullSectorDimensions:
    operator_rank: int
    silent_quantity_dimension: int
    redundant_relation_dimension: int


def classify_null_sectors(
    source: SparseRelationSource, *, tolerance: float = 1e-10
) -> NullSectorDimensions:
    singular = torch.linalg.svdvals(source.whitened_dense())
    rank = int(torch.count_nonzero(singular > tolerance).item())
    return NullSectorDimensions(
        rank,
        source.quantity_dim - rank,
        source.relation_dim - rank,
    )


@dataclass(frozen=True)
class GaugeValidation:
    declared: bool
    gauge_dimension: int
    orthonormality_residual: float
    silent_residual: float
    readout_residual: float
    incident_residual: float
    can_quotient: bool


def validate_declared_linear_gauge(
    source: SparseRelationSource,
    declared_gauge_basis: Tensor | None,
    relation_port: Tensor,
    observation: Tensor,
    *,
    tolerance: float = 1e-10,
) -> GaugeValidation:
    if declared_gauge_basis is None:
        return GaugeValidation(False, 0, 0.0, 0.0, 0.0, 0.0, False)
    basis = declared_gauge_basis.to(device=source.device, dtype=source.dtype)
    if basis.ndim != 2 or basis.shape[0] != source.quantity_dim:
        raise ValueError("declared gauge basis has the wrong quantity dimension")
    gram = basis.T @ basis
    identity = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    orthonormality = _relative_norm(gram - identity, identity)
    adjoint = source.whitened_adjoint(basis.T).T
    silent = _relative_norm(adjoint, basis)
    whitened_observation = (
        observation.to(source.device, source.dtype)
        / torch.sqrt(source.quantity_metric)[None, :]
    )
    readout = _relative_norm(whitened_observation @ basis, whitened_observation)
    whitened_port = (
        torch.sqrt(source.relation_metric)[:, None]
        * relation_port.to(source.device, source.dtype)
    )
    incident = source.whitened_apply(whitened_port.T).T
    incident_residual = _relative_norm(basis.T @ incident, incident)
    can_quotient = max(
        orthonormality, silent, readout, incident_residual
    ) <= tolerance
    return GaugeValidation(
        True,
        basis.shape[1],
        orthonormality,
        silent,
        readout,
        incident_residual,
        can_quotient,
    )


@dataclass(frozen=True)
class NoetherValidation:
    skew_residual: float
    commutator_residual: float
    valid_generator: bool


def validate_noether_generator(
    source: SparseRelationSource,
    generator: Tensor,
    *,
    calibration: float = 1.0,
    tolerance: float = 1e-10,
) -> NoetherValidation:
    generator = generator.to(device=source.device, dtype=source.dtype)
    if generator.shape != (source.quantity_dim, source.quantity_dim):
        raise ValueError("generator has the wrong quantity-carrier shape")
    skew = _relative_norm(generator + generator.T, generator)
    operator = calibration * source.whitened_dense() @ source.whitened_dense().T
    commutator = operator @ generator - generator @ operator
    commute = _relative_norm(commutator, operator)
    return NoetherValidation(skew, commute, max(skew, commute) <= tolerance)


def noether_charge(
    position: Tensor, velocity: Tensor, generator: Tensor, *, mass: float = 1.0
) -> Tensor:
    if mass <= 0:
        raise ValueError("mass must be positive")
    return mass * torch.dot(velocity, generator @ position)
