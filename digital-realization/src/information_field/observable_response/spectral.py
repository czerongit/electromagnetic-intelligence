from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from information_field.causal_minimal import CausalMinimalRealization


Tensor = torch.Tensor


def _tensor_digest(*values: Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        cpu = value.detach().contiguous().cpu()
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SpectralResidue:
    eigenvalue: float
    multiplicity: int
    residue: Tensor
    left_factor: Tensor
    right_factor: Tensor
    rank: int
    eigenvalue_spread: float
    discarded_singular_norm: float

    @property
    def factor_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (self.left_factor, self.right_factor)
        )

    @property
    def dense_bytes(self) -> int:
        return self.residue.numel() * self.residue.element_size()

    @property
    def factor_multiply_estimate(self) -> int:
        return self.rank * (
            self.residue.shape[0] + self.residue.shape[1]
        )

    @property
    def dense_multiply_estimate(self) -> int:
        return self.residue.numel()


@dataclass(frozen=True)
class SpectralCertificate:
    realization_digest: str
    state_dimension: int
    distinct_frequencies: int
    residue_rank_sum: int
    eigenvalue_tolerance: float
    residue_tolerance: float
    maximum_eigenvalue_spread: float
    maximum_discarded_singular_norm: float
    maximum_moment_residual: float


@dataclass(frozen=True)
class ObservableSpectrum:
    residues: tuple[SpectralResidue, ...]
    certificate: SpectralCertificate
    output_dimension: int
    input_dimension: int

    @property
    def eigenvalues(self) -> tuple[float, ...]:
        return tuple(residue.eigenvalue for residue in self.residues)

    @property
    def dense_residue_bytes(self) -> int:
        return sum(residue.dense_bytes for residue in self.residues)

    @property
    def factor_bytes(self) -> int:
        return sum(residue.factor_bytes for residue in self.residues)

    def is_valid_for(self, realization: CausalMinimalRealization) -> bool:
        return self.certificate.realization_digest == realization.certificate.execution_digest

    def assert_valid_for(self, realization: CausalMinimalRealization) -> None:
        if not self.is_valid_for(realization):
            raise ValueError("causal realization changed; recompile the observable spectrum")

    def moment(self, order: int) -> Tensor:
        if order < 0:
            raise ValueError("moment order must be nonnegative")
        result = torch.zeros(
            (self.output_dimension, self.input_dimension),
            dtype=(self.residues[0].residue.dtype if self.residues else torch.float64),
            device=(self.residues[0].residue.device if self.residues else "cpu"),
        )
        for item in self.residues:
            result = result + (item.eigenvalue**order) * item.residue
        return result

    def step_map(self, time: float, *, mass: float = 1.0) -> Tensor:
        return self._response_map(time, mass=mass, impulse=False)

    def impulse_map(self, time: float, *, mass: float = 1.0) -> Tensor:
        return self._response_map(time, mass=mass, impulse=True)

    def _response_map(self, time: float, *, mass: float, impulse: bool) -> Tensor:
        if time < 0 or mass <= 0:
            raise ValueError("time must be nonnegative and mass must be positive")
        if self.residues:
            result = torch.zeros_like(self.residues[0].residue)
        else:
            result = torch.zeros((self.output_dimension, self.input_dimension))
        zero_tolerance = self.certificate.eigenvalue_tolerance
        for item in self.residues:
            eigenvalue = item.eigenvalue
            if eigenvalue <= zero_tolerance:
                coefficient = time / mass if impulse else 0.5 * time * time / mass
            else:
                omega = (eigenvalue / mass) ** 0.5
                coefficient = (
                    torch.sin(torch.as_tensor(omega * time, dtype=result.dtype, device=result.device)).item()
                    / (mass * omega)
                    if impulse
                    else (
                        1.0
                        - torch.cos(
                            torch.as_tensor(omega * time, dtype=result.dtype, device=result.device)
                        ).item()
                    )
                    / eigenvalue
                )
            result = result + coefficient * item.residue
        return result


def _groups(eigenvalues: Tensor, tolerance: float) -> tuple[Tensor, ...]:
    if eigenvalues.numel() == 0:
        return ()
    groups: list[list[int]] = [[0]]
    anchor = float(eigenvalues[0].item())
    for index in range(1, eigenvalues.numel()):
        value = float(eigenvalues[index].item())
        if abs(value - anchor) <= tolerance:
            groups[-1].append(index)
        else:
            groups.append([index])
            anchor = value
    return tuple(
        torch.tensor(group, dtype=torch.int64, device=eigenvalues.device)
        for group in groups
    )


def compile_observable_spectrum(
    realization: CausalMinimalRealization,
    *,
    eigenvalue_tolerance: float | None = None,
    residue_tolerance: float | None = None,
) -> ObservableSpectrum:
    eigenvalues = realization.eigenvalues
    if eigenvalue_tolerance is None:
        eigenvalue_tolerance = realization.certificate.tolerance
    if residue_tolerance is None:
        residue_tolerance = realization.certificate.tolerance
    if eigenvalue_tolerance <= 0 or residue_tolerance <= 0:
        raise ValueError("spectral tolerances must be positive")

    residues: list[SpectralResidue] = []
    maximum_spread = 0.0
    maximum_discarded = 0.0
    for indices in _groups(eigenvalues, eigenvalue_tolerance):
        values = eigenvalues[indices]
        eigenvalue = float(values.mean().item())
        spread = float(torch.max(torch.abs(values - values.mean())).item())
        maximum_spread = max(maximum_spread, spread)
        dense = (
            realization.modal_observation[:, indices]
            @ realization.modal_incident_port[indices, :]
        )
        left, singular, right = torch.linalg.svd(dense, full_matrices=False)
        keep = singular > residue_tolerance
        rank = int(keep.sum().item())
        discarded = (
            float(singular[~keep].max().item()) if bool((~keep).any()) else 0.0
        )
        maximum_discarded = max(maximum_discarded, discarded)
        if rank == 0:
            continue
        root = torch.sqrt(singular[keep])
        left_factor = left[:, keep] * root[None, :]
        right_factor = root[:, None] * right[keep, :]
        residues.append(
            SpectralResidue(
                eigenvalue,
                int(indices.numel()),
                dense,
                left_factor,
                right_factor,
                rank,
                spread,
                discarded,
            )
        )

    output_dimension = realization.observation.shape[0]
    input_dimension = realization.incident_port.shape[1]
    maximum_moment = 0.0
    full = realization.modal_observation @ realization.modal_incident_port
    spectral = sum((item.residue for item in residues), torch.zeros_like(full))
    scale = max(1.0, float(torch.linalg.matrix_norm(full).item()))
    maximum_moment = max(
        maximum_moment,
        float(torch.linalg.matrix_norm(full - spectral).item()) / scale,
    )
    spectral_scale = max(
        1.0,
        float(torch.max(torch.abs(realization.eigenvalues)).item())
        if realization.eigenvalues.numel()
        else 1.0,
    )
    normalized_eigenvalues = realization.eigenvalues / spectral_scale
    full_power = realization.modal_incident_port
    for order in range(1, realization.state_dimension):
        full_power = normalized_eigenvalues[:, None] * full_power
        full = realization.modal_observation @ full_power
        spectral = sum(
            (
                ((item.eigenvalue / spectral_scale) ** order) * item.residue
                for item in residues
            ),
            torch.zeros_like(full),
        )
        scale = max(1.0, float(torch.linalg.matrix_norm(full).item()))
        maximum_moment = max(
            maximum_moment,
            float(torch.linalg.matrix_norm(full - spectral).item()) / scale,
        )
    certificate = SpectralCertificate(
        realization.certificate.execution_digest,
        realization.state_dimension,
        len(residues),
        sum(item.rank for item in residues),
        float(eigenvalue_tolerance),
        float(residue_tolerance),
        maximum_spread,
        maximum_discarded,
        maximum_moment,
    )
    return ObservableSpectrum(
        tuple(residues), certificate, output_dimension, input_dimension
    )
