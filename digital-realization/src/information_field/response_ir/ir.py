from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib

import torch

from information_field.observable_response import ExactGridRecurrence, FixedTimeGreenMap, SampledGreenFamily
from information_field.quotient_response import CompiledStaticResponse


Tensor = torch.Tensor


class ResponseContract(str, Enum):
    STATIC_COLUMNS = "static-columns"
    FIXED_TIME = "fixed-time"
    SAMPLED_TIMES = "sampled-times"
    REGULAR_GRID = "regular-grid"


class SemanticOperation(str, Enum):
    WEIGHTED_COLUMN_REDUCE = "weighted-column-reduce"
    INCIDENT_LINEAR_MAP = "incident-linear-map"
    INITIAL_POSITION_LINEAR_MAP = "initial-position-linear-map"
    INITIAL_VELOCITY_LINEAR_MAP = "initial-velocity-linear-map"
    SAMPLED_LINEAR_FAMILY = "sampled-linear-family"
    MODAL_FORCE = "modal-force"
    MODAL_POSITION_STEP = "modal-position-step"
    MODAL_VELOCITY_STEP = "modal-velocity-step"
    MODAL_READOUT = "modal-readout"


def _tensor_digest(value: Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode())
    digest.update(str(cpu.dtype).encode())
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _text_digest(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class TensorBinding:
    name: str
    value: Tensor
    digest: str

    @classmethod
    def freeze(cls, name: str, value: Tensor) -> "TensorBinding":
        frozen = value.detach().contiguous().clone()
        return cls(name, frozen, _tensor_digest(frozen))

    def intact(self) -> bool:
        return self.digest == _tensor_digest(self.value)


@dataclass(frozen=True)
class InvalidationKey:
    entries: tuple[tuple[str, str], ...]
    digest: str

    @classmethod
    def create(cls, **entries: str) -> "InvalidationKey":
        ordered = tuple(sorted((str(key), str(value)) for key, value in entries.items()))
        digest = _text_digest(*(f"{key}={value}" for key, value in ordered))
        return cls(ordered, digest)


@dataclass(frozen=True)
class PrecisionRequirement:
    reference_dtype: str
    exact_reference: bool
    target_dtype_qualified: bool
    maximum_absolute_error: float | None
    maximum_relative_error: float | None


@dataclass(frozen=True)
class IRExecution:
    output: Tensor
    final_position: Tensor | None = None
    final_velocity: Tensor | None = None


@dataclass(frozen=True)
class CompiledResponseIR:
    version: str
    contract: ResponseContract
    operations: tuple[SemanticOperation, ...]
    bindings: tuple[TensorBinding, ...]
    invalidation_key: InvalidationKey
    precision: PrecisionRequirement
    input_dimension: int
    output_dimension: int
    state_dimension: int
    artifact_digest: str

    def tensor(self, name: str) -> Tensor:
        for binding in self.bindings:
            if binding.name == name:
                return binding.value
        raise KeyError(name)

    def assert_integrity(self) -> None:
        if not all(binding.intact() for binding in self.bindings):
            raise ValueError("compiled response tensor changed; recompile the execution view")
        current = _artifact_digest(
            self.contract, self.operations, self.bindings, self.invalidation_key
        )
        if current != self.artifact_digest:
            raise ValueError("compiled response metadata changed; recompile the execution view")

    def is_valid_for(self, key: InvalidationKey) -> bool:
        return self.invalidation_key.digest == key.digest

    def assert_valid_for(self, key: InvalidationKey) -> None:
        if not self.is_valid_for(key):
            raise ValueError("source, ports, initial-state contract, or workload changed")

    def execute(
        self,
        incident: Tensor,
        *,
        local_indices: Tensor | None = None,
        valid: Tensor | None = None,
        initial_position: Tensor | None = None,
        initial_velocity: Tensor | None = None,
        state_position: Tensor | None = None,
        state_velocity: Tensor | None = None,
    ) -> IRExecution:
        self.assert_integrity()
        if self.contract is ResponseContract.STATIC_COLUMNS:
            return self._execute_static(incident, local_indices, valid)
        if self.contract is ResponseContract.FIXED_TIME:
            return self._execute_fixed(incident, initial_position, initial_velocity)
        if self.contract is ResponseContract.SAMPLED_TIMES:
            return self._execute_sampled(incident, initial_position, initial_velocity)
        if self.contract is ResponseContract.REGULAR_GRID:
            return self._execute_grid(incident, state_position, state_velocity)
        raise AssertionError("unhandled response contract")

    def _execute_static(
        self,
        amplitudes: Tensor,
        local_indices: Tensor | None,
        valid: Tensor | None,
    ) -> IRExecution:
        if local_indices is None or valid is None:
            raise ValueError("static column execution requires indices and validity")
        if amplitudes.shape != local_indices.shape or valid.shape != local_indices.shape:
            raise ValueError("static incident tensors must have one shape")
        if local_indices.dtype != torch.int64 or valid.dtype != torch.bool:
            raise ValueError("static indices and validity have wrong dtypes")
        columns = self.tensor("observed_columns")
        safe = torch.where(valid, local_indices, torch.zeros_like(local_indices))
        if bool(valid.any()) and (
            int(safe[valid].min()) < 0 or int(safe[valid].max()) >= columns.shape[0]
        ):
            raise ValueError("static local index is outside the compiled columns")
        weights = torch.where(valid, amplitudes, torch.zeros_like(amplitudes))
        output = torch.sum(columns[safe] * weights[..., None], dim=1)
        return IRExecution(output)

    def _execute_fixed(
        self,
        incident: Tensor,
        initial_position: Tensor | None,
        initial_velocity: Tensor | None,
    ) -> IRExecution:
        output = self.tensor("incident_map") @ incident
        if initial_position is not None:
            output = output + self.tensor("initial_position_map") @ initial_position
        if initial_velocity is not None:
            output = output + self.tensor("initial_velocity_map") @ initial_velocity
        return IRExecution(output)

    def _execute_sampled(
        self,
        incident: Tensor,
        initial_position: Tensor | None,
        initial_velocity: Tensor | None,
    ) -> IRExecution:
        output = torch.einsum("tzi,i->tz", self.tensor("incident_maps"), incident)
        if initial_position is not None:
            output = output + torch.einsum(
                "tzp,p->tz", self.tensor("initial_position_maps"), initial_position
            )
        if initial_velocity is not None:
            output = output + torch.einsum(
                "tzv,v->tz", self.tensor("initial_velocity_maps"), initial_velocity
            )
        return IRExecution(output)

    def _execute_grid(
        self,
        incidents: Tensor,
        state_position: Tensor | None,
        state_velocity: Tensor | None,
    ) -> IRExecution:
        if incidents.ndim != 2 or incidents.shape[1] != self.input_dimension:
            raise ValueError("incident history has the wrong port dimension")
        cosine = self.tensor("cosine")
        position = torch.zeros_like(cosine) if state_position is None else state_position
        velocity = torch.zeros_like(cosine) if state_velocity is None else state_velocity
        outputs = []
        for incident in incidents:
            force = self.tensor("modal_incident") @ incident
            next_position = (
                cosine * position
                + self.tensor("sine_over_omega") * velocity
                + self.tensor("force_position") * force
            )
            next_velocity = (
                self.tensor("negative_omega_sine") * position
                + cosine * velocity
                + self.tensor("force_velocity") * force
            )
            position, velocity = next_position, next_velocity
            outputs.append(self.tensor("modal_observation") @ position)
        output = (
            torch.stack(outputs)
            if outputs
            else torch.empty(
                (0, self.output_dimension), dtype=cosine.dtype, device=cosine.device
            )
        )
        return IRExecution(output, position, velocity)


def _artifact_digest(
    contract: ResponseContract,
    operations: tuple[SemanticOperation, ...],
    bindings: tuple[TensorBinding, ...],
    key: InvalidationKey,
) -> str:
    return _text_digest(
        "response-ir-v1",
        contract.value,
        *(operation.value for operation in operations),
        *(f"{binding.name}:{binding.digest}" for binding in bindings),
        key.digest,
    )

def _make_ir(
    contract: ResponseContract,
    operations: tuple[SemanticOperation, ...],
    tensors: tuple[tuple[str, Tensor], ...],
    key: InvalidationKey,
    *,
    input_dimension: int,
    output_dimension: int,
    state_dimension: int = 0,
) -> CompiledResponseIR:
    bindings = tuple(TensorBinding.freeze(name, value) for name, value in tensors)
    dtypes = {str(binding.value.dtype) for binding in bindings if binding.value.is_floating_point()}
    if len(dtypes) != 1:
        raise ValueError("compiled floating tensors must use one reference dtype")
    precision = PrecisionRequirement(next(iter(dtypes)), True, False, None, None)
    artifact = _artifact_digest(contract, operations, bindings, key)
    return CompiledResponseIR(
        "89.1",
        contract,
        operations,
        bindings,
        key,
        precision,
        input_dimension,
        output_dimension,
        state_dimension,
        artifact,
    )


def lower_static_response(compiled: CompiledStaticResponse) -> CompiledResponseIR:
    key = InvalidationKey.create(
        source=compiled.source_digest,
        observation=compiled.observation_digest,
        admission=_tensor_digest(compiled.selected_features),
        workload=ResponseContract.STATIC_COLUMNS.value,
    )
    return _make_ir(
        ResponseContract.STATIC_COLUMNS,
        (SemanticOperation.WEIGHTED_COLUMN_REDUCE,),
        (("observed_columns", compiled.observed_columns),),
        key,
        input_dimension=compiled.relation_dim,
        output_dimension=compiled.observed_columns.shape[1],
    )


def lower_fixed_time(fixed: FixedTimeGreenMap) -> CompiledResponseIR:
    key = InvalidationKey.create(
        realization=fixed.realization_digest,
        workload=f"{fixed.kind}:{fixed.time:.17g}:{fixed.mass:.17g}",
    )
    return _make_ir(
        ResponseContract.FIXED_TIME,
        (
            SemanticOperation.INCIDENT_LINEAR_MAP,
            SemanticOperation.INITIAL_POSITION_LINEAR_MAP,
            SemanticOperation.INITIAL_VELOCITY_LINEAR_MAP,
        ),
        (
            ("incident_map", fixed.incident_map),
            ("initial_position_map", fixed.initial_position_map),
            ("initial_velocity_map", fixed.initial_velocity_map),
        ),
        key,
        input_dimension=fixed.incident_map.shape[1],
        output_dimension=fixed.incident_map.shape[0],
    )


def lower_sampled_times(sampled: SampledGreenFamily) -> CompiledResponseIR:
    key = InvalidationKey.create(
        realization=sampled.realization_digest,
        workload=_text_digest(
            sampled.kind,
            f"{sampled.mass:.17g}",
            _tensor_digest(sampled.times),
        ),
    )
    return _make_ir(
        ResponseContract.SAMPLED_TIMES,
        (
            SemanticOperation.SAMPLED_LINEAR_FAMILY,
            SemanticOperation.INITIAL_POSITION_LINEAR_MAP,
            SemanticOperation.INITIAL_VELOCITY_LINEAR_MAP,
        ),
        (
            ("incident_maps", sampled.incident_maps),
            ("initial_position_maps", sampled.initial_position_maps),
            ("initial_velocity_maps", sampled.initial_velocity_maps),
            ("times", sampled.times),
        ),
        key,
        input_dimension=sampled.incident_maps.shape[2],
        output_dimension=sampled.incident_maps.shape[1],
    )


def lower_grid_recurrence(grid: ExactGridRecurrence) -> CompiledResponseIR:
    key = InvalidationKey.create(
        realization=grid.certificate.realization_digest,
        workload=f"regular-grid:{grid.step_size:.17g}:{grid.mass:.17g}",
    )
    return _make_ir(
        ResponseContract.REGULAR_GRID,
        (
            SemanticOperation.MODAL_FORCE,
            SemanticOperation.MODAL_POSITION_STEP,
            SemanticOperation.MODAL_VELOCITY_STEP,
            SemanticOperation.MODAL_READOUT,
        ),
        tuple(
            (name, getattr(grid, name))
            for name in (
                "cosine",
                "sine_over_omega",
                "negative_omega_sine",
                "force_position",
                "force_velocity",
                "modal_incident",
                "modal_observation",
                "modal_initial_position",
                "modal_initial_velocity",
            )
        ),
        key,
        input_dimension=grid.modal_incident.shape[1],
        output_dimension=grid.modal_observation.shape[0],
        state_dimension=grid.state_dimension,
    )
