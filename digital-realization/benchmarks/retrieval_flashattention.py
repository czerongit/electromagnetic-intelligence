"""Matched relation-coordinate and FlashAttention retrieval benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional

from information_field.reduction import select_static_executor
from information_field.retrieval import (
    RelationIncidents,
    RelationNativeSource,
    dense_quotient_response,
    determine_canonical_fixture,
    make_binding_problem,
    move_problem,
)
from information_field.retrieval.cuda import CudaQuotientResponse, source_path


@dataclass(frozen=True)
class Condition:
    identifier: str
    batch: int
    prefix: int
    bindings: int


CONDITIONS = (
    Condition("l64-k4", 8, 64, 4),
    Condition("l128-k8", 8, 128, 8),
    Condition("l256-k16", 4, 256, 16),
    Condition("l512-k64", 1, 512, 64),
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_digest() -> str:
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    paths = sorted((root / "src" / "information_field").rglob("*"))
    for path in paths:
        if not path.is_file() or path.suffix not in {".py", ".cu"}:
            continue
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _samples(call, repetitions: int) -> tuple[float, ...]:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        call()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return tuple(values)


def _host_samples(call, repetitions: int = 20) -> tuple[float, ...]:
    for _ in range(3):
        call()
    values = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        call()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return tuple(values)


def _tensor_bytes(*values: torch.Tensor) -> int:
    return sum(value.numel() * value.element_size() for value in values)


def _accuracy(problem, response: torch.Tensor) -> float:
    scores = torch.abs(
        torch.einsum("bqd,bkd->bqk", response, problem.payload_bases)
    )
    prediction = torch.argmax(scores, dim=-1)
    return float(
        torch.mean(
            (prediction == problem.target_payload_indices).to(scores.dtype)
        ).item()
    )


def _validate_machine(declared: dict) -> dict:
    properties = torch.cuda.get_device_properties(0)
    actual = properties.name
    expected = set(
        str(declared["expected_gpu_model"]).casefold().replace("-", " ").split()
    ) - {"nvidia", "geforce", "gpu"}
    actual_tokens = set(actual.casefold().replace("-", " ").split())
    if expected and not expected.issubset(actual_tokens):
        raise RuntimeError(
            f"detected GPU {actual!r} does not match "
            f"{declared['expected_gpu_model']!r}"
        )
    memory_gib = properties.total_memory / (1024**3)
    if memory_gib + 0.25 < float(declared["expected_memory_gib"]):
        raise RuntimeError("detected GPU memory is below the declared boundary")
    return {
        "device_name": actual,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "memory_gib": memory_gib,
        "torch_cuda": torch.version.cuda,
    }


def _execute(condition: Condition, kernel, repetitions: int, output: Path) -> dict:
    cpu_problem = move_problem(
        make_binding_problem(
            batch_size=condition.batch,
            binding_count=condition.bindings,
            prefix_length=condition.prefix,
            seed=7601,
            split="test",
        ),
        "cpu",
        torch.float32,
    )
    construction_samples = _host_samples(
        lambda: determine_canonical_fixture(
            cpu_problem.attention_input, condition.bindings
        )
    )
    cpu_source = determine_canonical_fixture(
        cpu_problem.attention_input, condition.bindings
    )
    reference = dense_quotient_response(cpu_source)
    field_accuracy = _accuracy(cpu_problem, reference)
    selections = select_static_executor(
        cpu_source.observed_columns[0],
        int(cpu_source.incidents.indices.shape[-1]),
    )
    if any(selection.executor != "columns" for selection in selections):
        raise RuntimeError("retrieval condition did not select indexed columns")

    source = cpu_source.to("cuda", torch.float16)
    field_output = kernel(source)
    torch.cuda.synchronize()
    field_error = float(
        torch.max(torch.abs(field_output.float().cpu() - reference)).item()
    )
    field_samples = _samples(lambda: kernel(source), repetitions)
    field_incremental = _samples(lambda: kernel(source.first()), repetitions)

    reversed_incidents = RelationIncidents(
        torch.flip(cpu_source.incidents.indices, dims=(1,)),
        cpu_source.incidents.amplitudes,
        cpu_source.incidents.valid,
    )
    changed = dense_quotient_response(
        RelationNativeSource(cpu_source.observed_columns, reversed_incidents)
    )
    intervention_change = float(torch.max(torch.abs(changed - reference)).item())

    device_problem = move_problem(cpu_problem, "cuda", torch.float16)
    memory = device_problem.attention_input.prefix_values
    query = device_problem.attention_input.incident_values
    valid = device_problem.attention_input.prefix_valid

    def flash(call_query):
        return functional.scaled_dot_product_attention(
            call_query.unsqueeze(1),
            memory.unsqueeze(1),
            memory.unsqueeze(1),
            attn_mask=valid[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        ).squeeze(1)

    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        flash_output = flash(query)
        torch.cuda.synchronize()
        flash_accuracy = _accuracy(device_problem, flash_output)
        flash_samples = _samples(lambda: flash(query), repetitions)
        flash_incremental = _samples(
            lambda: flash(query[:, :1].contiguous()), repetitions
        )

    field_trace = output / f"{condition.identifier}-field-trace.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        kernel(source)
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(field_trace))
    field_events = sorted(event.key for event in profile.key_averages())

    flash_trace = output / f"{condition.identifier}-flash-trace.json"
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            flash(query)
            torch.cuda.synchronize()
    profile.export_chrome_trace(str(flash_trace))
    flash_events = sorted(event.key for event in profile.key_averages())

    field_verified = any("quotient_response" in event for event in field_events)
    flash_verified = any("flash" in event.casefold() for event in flash_events)
    eligible = (
        field_accuracy >= 0.95
        and flash_accuracy >= 0.95
        and field_error <= 5e-2
        and intervention_change > 0.0
        and field_verified
        and flash_verified
    )
    field_median = statistics.median(field_samples)
    flash_median = statistics.median(flash_samples)
    field_incremental_median = statistics.median(field_incremental)
    flash_incremental_median = statistics.median(flash_incremental)
    return {
        "condition": condition.identifier,
        "shape": asdict(condition),
        "field_accuracy": field_accuracy,
        "flash_accuracy": flash_accuracy,
        "comparison_eligible": eligible,
        "field_response_error_vs_fp32_cpu": field_error,
        "query_intervention_maximum_change": intervention_change,
        "field_source_determination_samples_ms": construction_samples,
        "field_source_determination_ms": statistics.median(construction_samples),
        "field_device_samples_ms": field_samples,
        "field_device_median_ms": field_median,
        "field_incremental_device_samples_ms": field_incremental,
        "field_incremental_device_median_ms": field_incremental_median,
        "flash_device_samples_ms": flash_samples,
        "flash_device_median_ms": flash_median,
        "flash_incremental_device_samples_ms": flash_incremental,
        "flash_incremental_device_median_ms": flash_incremental_median,
        "warm_speedup_over_flash": flash_median / field_median if eligible else None,
        "incremental_speedup_over_flash": (
            flash_incremental_median / field_incremental_median
            if eligible
            else None
        ),
        "field_retained_state_bytes": _tensor_bytes(
            source.observed_columns,
            source.incidents.indices,
            source.incidents.amplitudes,
            source.incidents.valid,
        ),
        "flash_retained_state_bytes": _tensor_bytes(memory, valid),
        "field_trace": field_trace.name,
        "flash_trace": flash_trace.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    machine = (
        json.loads(arguments.machine.read_text())
        if arguments.machine is not None
        else {"status": "not-supplied"}
    )
    manifest = {
        "schema": "information-field.retrieval-flashattention.v1",
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "machine": machine,
        "implementation_sha256": _implementation_digest(),
        "cuda_source_sha256": _digest(source_path()),
        "field_input": "target-independent sparse relation coordinate",
        "flash_input": "equivalent target-independent ambient query vector",
        "field_dtype": "float16 with float32 accumulation",
        "flash_dtype": "float16",
        "minimum_accuracy": 0.95,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if not arguments.execute:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if arguments.machine is None:
        raise SystemExit("--machine is required with --execute")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    manifest["detected_machine"] = _validate_machine(machine)
    kernel = CudaQuotientResponse()
    records = [
        _execute(condition, kernel, arguments.repetitions, arguments.output_dir)
        for condition in CONDITIONS
    ]
    with (arguments.output_dir / "records.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))
    (arguments.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
