from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from information_field.reduction.temporal import TEMPORAL_CONDITIONS, make_temporal_fixture, matrix_powers, observe
from information_field.reduction.temporal_cuda import TemporalCuda, source_path


def samples(call, repetitions: int) -> tuple[float, ...]:
    values = []
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        call()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return tuple(values)


def host_samples(call, repetitions: int = 20) -> tuple[float, ...]:
    for _ in range(3):
        call()
    values = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        call()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return tuple(values)


def validate_machine(machine: dict) -> dict:
    properties = torch.cuda.get_device_properties(0)
    actual_name = properties.name
    expected_tokens = set(
        str(machine["expected_gpu_model"]).casefold().replace("-", " ").split()
    ) - {"nvidia", "geforce", "gpu"}
    actual_tokens = set(actual_name.casefold().replace("-", " ").split())
    if expected_tokens and not expected_tokens.issubset(actual_tokens):
        raise RuntimeError(
            f"detected GPU {actual_name!r} does not match {machine['expected_gpu_model']!r}"
        )
    memory_gib = properties.total_memory / (1024 ** 3)
    if memory_gib + 0.25 < float(machine["expected_memory_gib"]):
        raise RuntimeError("detected GPU memory is below the declared qualification boundary")
    return {
        "device_name": actual_name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "memory_gib": memory_gib,
        "torch_cuda": torch.version.cuda,
    }


def execute_condition(condition, kernel: TemporalCuda, repetitions: int) -> dict:
    transitions64, lengths, initial64, readout64 = make_temporal_fixture(condition)
    transitions = transitions64.float().contiguous()
    initial = initial64.float().contiguous()
    powers = matrix_powers(transitions, lengths)
    power_preparation_samples = host_samples(
        lambda: matrix_powers(transitions, lengths)
    )
    power_preparation_ms = statistics.median(power_preparation_samples)
    transfer_started = time.perf_counter_ns()
    device_transitions = transitions.cuda()
    device_lengths = lengths.cuda()
    device_initial = initial.cuda()
    device_powers = powers.cuda()
    torch.cuda.synchronize()
    transfer_ms = (time.perf_counter_ns() - transfer_started) / 1e6

    regular = kernel.regular(device_transitions, device_lengths, device_initial)
    event = kernel.event(device_powers, device_initial)
    torch.cuda.synchronize()
    regular_cpu = regular.cpu().double()
    event_cpu = event.cpu().double()
    state_error = float(torch.max(torch.abs(event_cpu - regular_cpu)).item())
    observation_error = float(torch.max(torch.abs(
        observe(event_cpu, readout64) - observe(regular_cpu, readout64)
    )).item())
    regular_samples = samples(
        lambda: kernel.regular(device_transitions, device_lengths, device_initial), repetitions
    )
    event_samples = samples(lambda: kernel.event(device_powers, device_initial), repetitions)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profile:
        kernel.regular(device_transitions, device_lengths, device_initial)
        kernel.event(device_powers, device_initial)
        torch.cuda.synchronize()
    events = sorted(event.key for event in profile.key_averages())
    regular_verified = any("regular_grid_temporal_response_kernel" in event for event in events)
    event_verified = any("event_composed_temporal_response_kernel" in event for event in events)
    regular_median = statistics.median(regular_samples)
    event_median = statistics.median(event_samples)
    return {
        "condition": condition.identifier,
        "shape": {
            "batch": condition.batch,
            "modes": condition.modes,
            "run_lengths": condition.run_lengths,
            "steps": condition.steps,
            "events": len(condition.run_lengths),
        },
        "comparison_contract": "same affine recurrence, source events, durations, initial state, and final-state observation",
        "regular_execution": "native fused regular-grid recurrence",
        "event_execution": "native fused application of exact per-event transition powers",
        "state_max_abs": state_error,
        "observation_max_abs": observation_error,
        "quality_matched": state_error <= 2e-3 and observation_error <= 2e-3,
        "power_preparation_ms": power_preparation_ms,
        "power_preparation_samples_ms": power_preparation_samples,
        "input_transfer_ms": transfer_ms,
        "regular_samples_ms": regular_samples,
        "regular_median_ms": regular_median,
        "event_samples_ms": event_samples,
        "event_median_ms": event_median,
        "response_speedup": regular_median / event_median,
        "event_preparation_plus_response_ms": power_preparation_ms + event_median,
        "regular_verified": regular_verified,
        "event_verified": event_verified,
        "comparison_eligible": regular_verified and event_verified and state_error <= 2e-3 and observation_error <= 2e-3,
        "claim_boundary": "temporal specialization only; not a FlashAttention comparison",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    machine = json.loads(arguments.machine.read_text(encoding="utf-8"))
    manifest = {
        "schema": "information-field.temporal-event-cuda.v1",
        "conditions": [condition.__dict__ for condition in TEMPORAL_CONDITIONS],
        "machine": machine,
        "cuda_source_sha256": hashlib.sha256(source_path().read_bytes()).hexdigest(),
        "comparison_boundary": "matched stateful recurrence; no FlashAttention ratio",
        "dtype": "float32",
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    (arguments.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not arguments.execute:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    manifest["detected_machine"] = validate_machine(machine)
    (arguments.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    kernel = TemporalCuda()
    records = [execute_condition(condition, kernel, arguments.repetitions) for condition in TEMPORAL_CONDITIONS]
    with (arguments.output_dir / "records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({
                "condition": record["condition"],
                "quality_matched": record["quality_matched"],
                "response_speedup": record["response_speedup"],
                "claim_boundary": record["claim_boundary"],
            }))


if __name__ == "__main__":
    main()
