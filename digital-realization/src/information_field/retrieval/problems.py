"""Role-based attention problems and implementation-independent metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .contracts import AttentionAdapter, AttentionInput, AttentionResult


@dataclass(frozen=True)
class BenchmarkThresholds:
    retrieval_accuracy: float = 0.95
    routing_margin: float = 1e-4
    distractor_accuracy: float = 0.90
    long_range_accuracy: float = 0.90
    causal_tolerance: float = 1e-10
    control_excess_over_chance: float = 0.10


FROZEN_THRESHOLDS = BenchmarkThresholds()


@dataclass(frozen=True)
class BindingProblem:
    attention_input: AttentionInput
    key_bases: Tensor
    payload_bases: Tensor
    target_payload_indices: Tensor
    binding_permutations: Tensor
    basis_fingerprints: Tensor
    split: str
    binding_count: int

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / float(self.binding_count)


@dataclass(frozen=True)
class BindingScore:
    accuracy: float
    predictions: Tensor
    target_scores: Tensor
    passed: bool


@dataclass(frozen=True)
class RoutingScore:
    influence_matrix: Tensor
    reversal: bool
    passed: bool


@dataclass(frozen=True)
class CausalityScore:
    maximum_earlier_change: float
    batched_incremental_error: float
    future_leakage_detected: bool
    passed: bool


@dataclass(frozen=True)
class ScalingScore:
    levels: tuple[float, ...]
    accuracies: tuple[float, ...]
    minimum_accuracy: float
    passed: bool


def _orthogonal_basis(
    dimension: int, generator: torch.Generator, dtype: torch.dtype
) -> Tensor:
    raw = torch.randn(
        (dimension, dimension),
        generator=generator,
        dtype=dtype,
    )
    basis, _ = torch.linalg.qr(raw)
    return basis.transpose(0, 1)


def make_binding_problem(
    *,
    batch_size: int,
    binding_count: int,
    prefix_length: int,
    seed: int,
    split: str,
    phase_shift: float = 0.0,
    query_scale: float = 4.0,
    masked_prefix_count: int = 0,
) -> BindingProblem:
    if (
        batch_size < 1
        or binding_count < 2
        or prefix_length < 2 * binding_count
        or masked_prefix_count < 0
        or prefix_length - masked_prefix_count < 2 * binding_count
    ):
        raise ValueError("binding problem dimensions are insufficient")
    if split not in {"pilot", "train", "test"}:
        raise ValueError("split must be pilot, train, or test")
    dtype = torch.float64
    dimension = 2 * binding_count
    positions = (
        torch.arange(prefix_length, dtype=dtype)
        * (2.0 * math.pi / float(prefix_length))
    )
    prefix_values = []
    incidents = []
    payload_bases = []
    targets = []
    permutations = []
    fingerprints = []
    split_offset = {"pilot": 1000, "train": 2000, "test": 3000}[
        split
    ]
    for item in range(batch_size):
        generator = torch.Generator().manual_seed(
            split_offset + seed * 1009 + item
        )
        basis = _orthogonal_basis(
            dimension, generator, dtype
        )
        keys = basis[:binding_count]
        payloads = basis[binding_count:]
        permutation = torch.randperm(
            binding_count, generator=generator
        )
        bindings = keys + payloads[permutation]
        samples = []
        for coordinate in positions:
            value = torch.zeros(dimension, dtype=dtype)
            for index in range(binding_count):
                value += (
                    math.cos(
                        float(index + 1)
                        * (float(coordinate) - phase_shift)
                    )
                    / math.sqrt(math.pi)
                ) * bindings[index]
            samples.append(value)
        prefix_values.append(torch.stack(samples))
        incidents.append(query_scale * keys)
        payload_bases.append(payloads)
        targets.append(permutation)
        permutations.append(permutation)
        fingerprints.append(basis.reshape(-1))
    prefix_valid = torch.ones(
        (batch_size, prefix_length), dtype=torch.bool
    )
    if masked_prefix_count:
        prefix_valid[:, -masked_prefix_count:] = False
    incident_valid = torch.ones(
        (batch_size, binding_count), dtype=torch.bool
    )
    incident_positions = torch.zeros(
        (batch_size, binding_count), dtype=dtype
    )
    return BindingProblem(
        attention_input=AttentionInput(
            prefix_values=torch.stack(prefix_values),
            incident_values=torch.stack(incidents),
            prefix_positions=positions.unsqueeze(0).expand(
                batch_size, -1
            ),
            incident_positions=incident_positions,
            prefix_valid=prefix_valid,
            incident_valid=incident_valid,
        ),
        key_bases=torch.stack(
            tuple(
                # Incident vectors are query_scale times the keys.
                value / query_scale
                for value in incidents
            )
        ),
        payload_bases=torch.stack(payload_bases),
        target_payload_indices=torch.stack(targets),
        binding_permutations=torch.stack(permutations),
        basis_fingerprints=torch.stack(fingerprints),
        split=split,
        binding_count=binding_count,
    )


def reassign_bindings(
    problem: BindingProblem, *, shift: int = 1
) -> BindingProblem:
    """Change bindings while preserving bases, positions, and multisets."""

    count = problem.binding_count
    if shift % count == 0:
        raise ValueError("reassignment shift must change bindings")
    new_permutations = (
        problem.binding_permutations + int(shift)
    ) % count
    positions = problem.attention_input.prefix_positions
    prefixes = []
    for item in range(positions.shape[0]):
        bindings = (
            problem.key_bases[item]
            + problem.payload_bases[item][
                new_permutations[item]
            ]
        )
        samples = []
        for coordinate in positions[item]:
            value = torch.zeros_like(bindings[0])
            for index in range(count):
                value += (
                    math.cos(
                        float(index + 1) * float(coordinate)
                    )
                    / math.sqrt(math.pi)
                ) * bindings[index]
            samples.append(value)
        prefixes.append(torch.stack(samples))
    current = problem.attention_input
    return BindingProblem(
        attention_input=AttentionInput(
            prefix_values=torch.stack(prefixes),
            incident_values=current.incident_values,
            prefix_positions=current.prefix_positions,
            incident_positions=current.incident_positions,
            prefix_valid=current.prefix_valid,
            incident_valid=current.incident_valid,
        ),
        key_bases=problem.key_bases,
        payload_bases=problem.payload_bases,
        target_payload_indices=new_permutations,
        binding_permutations=new_permutations,
        basis_fingerprints=problem.basis_fingerprints,
        split=problem.split,
        binding_count=count,
    )


def score_binding_result(
    problem: BindingProblem,
    result: AttentionResult,
    thresholds: BenchmarkThresholds = FROZEN_THRESHOLDS,
) -> BindingScore:
    scores = torch.abs(
        torch.einsum(
            "bqd,bkd->bqk",
            result.responses,
            problem.payload_bases,
        )
    ).detach()
    predictions = torch.argmax(scores, dim=-1)
    accuracy = float(
        torch.mean(
            (
                predictions
                == problem.target_payload_indices
            ).to(scores.dtype)
        )
    )
    targets = torch.gather(
        scores,
        2,
        problem.target_payload_indices.unsqueeze(-1),
    ).squeeze(-1)
    return BindingScore(
        accuracy=accuracy,
        predictions=predictions,
        target_scores=targets,
        passed=accuracy >= thresholds.retrieval_accuracy,
    )


def evaluate_binding(
    adapter: AttentionAdapter,
    problem: BindingProblem,
    thresholds: BenchmarkThresholds = FROZEN_THRESHOLDS,
) -> BindingScore:
    return score_binding_result(
        problem, adapter(problem.attention_input), thresholds
    )


def make_routing_problem() -> tuple[AttentionInput, Tensor]:
    dtype = torch.float64
    width = 4
    prefix_length = 8
    positions = (
        torch.arange(prefix_length, dtype=dtype)
        * (2.0 * math.pi / float(prefix_length))
    )
    basis = torch.eye(width, dtype=dtype)
    bindings = (basis[0] + basis[2], basis[1] + basis[3])
    prefix = []
    for binding in bindings:
        prefix.append(
            torch.stack(
                tuple(
                    math.cos(float(position))
                    * binding
                    / math.sqrt(math.pi)
                    for position in positions
                )
            )
        )
    incidents = 4.0 * torch.stack(
        (
            torch.stack((basis[0], basis[1])),
            torch.stack((basis[0], basis[1])),
        )
    )
    return (
        AttentionInput(
            prefix_values=torch.stack(prefix),
            incident_values=incidents,
            prefix_positions=positions.unsqueeze(0).expand(2, -1),
            incident_positions=torch.zeros((2, 2), dtype=dtype),
            prefix_valid=torch.ones((2, prefix_length), dtype=torch.bool),
            incident_valid=torch.ones((2, 2), dtype=torch.bool),
        ),
        torch.stack((basis[2], basis[3])),
    )


def evaluate_routing(
    adapter: AttentionAdapter,
    thresholds: BenchmarkThresholds = FROZEN_THRESHOLDS,
) -> RoutingScore:
    problem, payloads = make_routing_problem()
    response = adapter(problem).responses
    influences = torch.abs(
        torch.einsum("bqd,kd->bqk", response, payloads)
    ).detach()
    # Each row's two incidents are scored against its matching payload.
    matrix = torch.stack(
        (influences[0, :, 0], influences[1, :, 1])
    )
    reversal = bool(
        matrix[0, 0] > matrix[0, 1] + thresholds.routing_margin
        and matrix[1, 1]
        > matrix[1, 0] + thresholds.routing_margin
    )
    return RoutingScore(matrix, reversal, reversal)


def evaluate_distractor_scaling(
    adapter_factory,
    binding_counts: Sequence[int],
    *,
    seed: int,
    thresholds: BenchmarkThresholds = FROZEN_THRESHOLDS,
) -> ScalingScore:
    levels = tuple(int(value) for value in binding_counts)
    accuracies = []
    for count in levels:
        problem = make_binding_problem(
            batch_size=12,
            binding_count=count,
            prefix_length=max(4 * count, 16),
            seed=seed,
            split="test",
        )
        accuracies.append(
            evaluate_binding(
                adapter_factory(count), problem, thresholds
            ).accuracy
        )
    minimum = min(accuracies)
    return ScalingScore(
        tuple(float(level) for level in levels),
        tuple(accuracies),
        minimum,
        minimum >= thresholds.distractor_accuracy,
    )


def evaluate_long_range(
    adapter_factory,
    phase_shifts: Sequence[float],
    *,
    binding_count: int,
    seed: int,
    thresholds: BenchmarkThresholds = FROZEN_THRESHOLDS,
) -> ScalingScore:
    shifts = tuple(float(value) for value in phase_shifts)
    accuracies = []
    for shift in shifts:
        problem = make_binding_problem(
            batch_size=12,
            binding_count=binding_count,
            prefix_length=max(4 * binding_count, 16),
            seed=seed,
            split="test",
            phase_shift=shift,
        )
        accuracies.append(
            evaluate_binding(
                adapter_factory(binding_count),
                problem,
                thresholds,
            ).accuracy
        )
    minimum = min(accuracies)
    return ScalingScore(
        shifts,
        tuple(accuracies),
        minimum,
        minimum >= thresholds.long_range_accuracy,
    )


def evaluate_causality(
    adapter: AttentionAdapter,
    batch: AttentionInput,
    *,
    expect_leakage: bool = False,
    thresholds: BenchmarkThresholds = FROZEN_THRESHOLDS,
) -> CausalityScore:
    initial_state = adapter.initialize_state(batch)
    first = adapter(batch, initial_state)
    changed_values = batch.incident_values.clone()
    changed_values[:, -1] += 17.0
    changed = AttentionInput(
        batch.prefix_values,
        changed_values,
        batch.prefix_positions,
        batch.incident_positions,
        batch.prefix_valid,
        batch.incident_valid,
    )
    second = adapter(changed, initial_state)
    earlier_change = float(
        torch.max(
            torch.abs(
                first.responses[:, :-1]
                - second.responses[:, :-1]
            )
        ).detach()
    )
    pieces = []
    incremental_state = initial_state
    for incident in range(batch.incident_values.shape[1]):
        single = AttentionInput(
            batch.prefix_values,
            batch.incident_values[:, incident : incident + 1],
            batch.prefix_positions,
            batch.incident_positions[:, incident : incident + 1],
            batch.prefix_valid,
            batch.incident_valid[:, incident : incident + 1],
        )
        current = adapter(single, incremental_state)
        pieces.append(current.responses)
        incremental_state = current.state
    incremental = torch.cat(pieces, dim=1)
    incremental_error = float(
        torch.max(
            torch.abs(first.responses - incremental)
        ).detach()
    )
    leakage = earlier_change > thresholds.causal_tolerance
    passed = (
        leakage
        if expect_leakage
        else (
            not leakage
            and incremental_error <= thresholds.causal_tolerance
        )
    )
    return CausalityScore(
        earlier_change,
        incremental_error,
        leakage,
        passed,
    )


def splits_are_disjoint(
    first: BindingProblem, second: BindingProblem
) -> bool:
    if first.split == second.split:
        return False
    for left in first.basis_fingerprints:
        for right in second.basis_fingerprints:
            if torch.equal(left, right):
                return False
    return True
