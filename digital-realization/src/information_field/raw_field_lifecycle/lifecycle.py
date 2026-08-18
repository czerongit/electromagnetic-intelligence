from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import statistics
import time
from pathlib import Path

import torch

from information_field.geometric_observation import determine_declarative_field
from information_field.geometric_observation.declarative import _held_out_occurrences
from information_field.geometric_observation.wikipedia import load_wikitext_split
from information_field.native_response_kernels import (
    CPUResponseKernels,
    MPSResponseKernels,
    calibrate_executor,
    execute_selected,
)
from information_field.profiled_response import execute_prepared, prepare_backend_request
from information_field.response_backends import BackendKind, BackendRequest, backend_capability, lower_backend_plan
from information_field.response_ir import lower_static_response

from .bridge import lower_declarative_source, prepare_context_batch
from .static import compile_identity_static_response


@dataclass(frozen=True)
class BackendLifecycleRecord:
    backend: str
    dtype: str
    plan_lowering_seconds: float
    request_preparation_seconds: float
    executor_initialization_seconds: float
    calibration_seconds: float
    response_median_seconds: float
    output_transfer_seconds: float
    readout_median_seconds: float
    selected_executor: str
    tensor_median_microseconds: float
    native_median_microseconds: float
    maximum_absolute_error: float
    plan_retained_bytes: int
    input_transfer_bytes: int
    output_bytes: int
    estimated_column_read_bytes: int
    estimated_weight_index_read_bytes: int
    estimated_output_write_bytes: int
    response_scalar_multiply_adds: int
    tensor_host_operator_calls: int
    native_device_dispatches: int
    synchronized: bool


def _timed(operation):
    started = time.perf_counter()
    result = operation()
    return time.perf_counter() - started, result


def _synchronize(backend: BackendKind) -> None:
    if backend is BackendKind.MPS:
        torch.mps.synchronize()
    elif backend is BackendKind.CUDA:
        torch.cuda.synchronize()


def _median_seconds(operation, backend: BackendKind, iterations: int) -> float:
    samples = []
    for _ in range(iterations):
        _synchronize(backend)
        started = time.perf_counter()
        operation()
        _synchronize(backend)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _direct_response(field, contexts) -> torch.Tensor:
    term_index = {term: index for index, term in enumerate(field.terms)}
    output = torch.zeros((len(contexts), len(field.terms)), dtype=torch.float64)
    for row, context in enumerate(contexts):
        for term, value in field.source_covector(context).items():
            output[row, term_index[term]] = value
    return output


def _decode(response: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    maximum = min(5, response.shape[1])
    top = torch.argsort(response, dim=1, descending=True, stable=True)[:, :maximum]
    target_scores = response[torch.arange(response.shape[0]), targets]
    return {
        "coverage": float(torch.mean((target_scores > 0).to(torch.float64)).item()),
        "top_1_accuracy": float(torch.mean((top[:, 0] == targets).to(torch.float64)).item()),
        "top_5_accuracy": float(
            torch.mean(torch.any(top == targets[:, None], dim=1).to(torch.float64)).item()
        ),
    }


def _backend_lifecycle(
    ir,
    request,
    reference,
    *,
    backend: BackendKind,
    calibration_iterations: int,
    response_iterations: int,
) -> tuple[BackendLifecycleRecord, torch.Tensor, float]:
    dtype = torch.float64 if backend is BackendKind.CPU else torch.float32
    lowering_seconds, plan = _timed(lambda: lower_backend_plan(ir, backend, dtype))
    preparation_seconds, prepared = _timed(
        lambda: prepare_backend_request(plan, request)
    )
    kernels_type = CPUResponseKernels if backend is BackendKind.CPU else MPSResponseKernels
    initialization_seconds, kernels = _timed(kernels_type)
    absolute = 1e-12 if backend is BackendKind.CPU else 2e-5
    relative = 1e-12 if backend is BackendKind.CPU else 2e-4
    calibration_seconds, selection = _timed(
        lambda: calibrate_executor(
            ir,
            plan,
            kernels,
            request,
            absolute_tolerance=absolute,
            relative_tolerance=relative,
            warmup=2,
            iterations=calibration_iterations,
        )
    )
    response_seconds = _median_seconds(
        lambda: execute_selected(ir, plan, kernels, selection, request),
        backend,
        response_iterations,
    )
    execution = execute_selected(ir, plan, kernels, selection, request)
    _synchronize(backend)
    output_bytes = execution.result.output.numel() * execution.result.output.element_size()
    transfer_seconds, output = _timed(
        lambda: execution.result.output.detach().cpu().to(torch.float64)
    )
    difference = float(torch.max(torch.abs(output - reference)).item())
    if difference > absolute + relative * float(torch.max(torch.abs(reference)).item()):
        raise ValueError("selected backend response failed the lifecycle oracle")
    readout_seconds = _median_seconds(lambda: torch.topk(output, k=5, dim=1), BackendKind.CPU, response_iterations)
    valid_entries = prepared.local_indices.numel()
    multiply_adds = valid_entries * plan.output_dimension
    scalar_bytes = torch.tensor([], dtype=plan.dtype).element_size()
    column_reads = multiply_adds * scalar_bytes
    weight_index_reads = valid_entries * (scalar_bytes + 8)
    tensor_execution = execute_prepared(plan, prepared)
    native_execution = kernels.execute(plan, prepared)
    _synchronize(backend)
    record = BackendLifecycleRecord(
        backend.value,
        str(dtype),
        lowering_seconds,
        preparation_seconds,
        initialization_seconds,
        calibration_seconds,
        response_seconds,
        transfer_seconds,
        readout_seconds,
        selection.selected.value,
        selection.tensor_median_microseconds,
        selection.native_median_microseconds,
        difference,
        plan.retained_bytes,
        prepared.input_transfer_bytes,
        output_bytes,
        column_reads,
        weight_index_reads,
        output_bytes,
        multiply_adds,
        tensor_execution.host_operator_calls,
        native_execution.record.dispatches,
        backend is not BackendKind.CPU,
    )
    return record, output, readout_seconds


def _replacement_lifecycle(
    train,
    contexts,
    old_ir,
    *,
    radius: int,
    minimum_occurrences: int,
    calibration_iterations: int,
) -> dict:
    replacement_train = train + (
        "The information field relates information through local field relations.",
    )
    started = time.perf_counter()
    determination_seconds, field = _timed(
        lambda: determine_declarative_field(
            replacement_train,
            radius=radius,
            minimum_occurrences=minimum_occurrences,
            normalization="joint",
        )
    )
    bridge_seconds, bridge = _timed(lambda: lower_declarative_source(field))
    incident_seconds, incidents = _timed(
        lambda: prepare_context_batch(bridge, contexts)
    )
    compile_seconds, compiled = _timed(
        lambda: compile_identity_static_response(
            bridge.source, incidents.admitted_features()
        )
    )
    compiled_preparation_seconds, source_prepared = _timed(
        lambda: compiled.prepare(incidents)
    )
    request = BackendRequest(
        source_prepared.amplitudes,
        local_indices=source_prepared.local_indices,
        valid=source_prepared.valid,
    )
    ir_seconds, ir = _timed(lambda: lower_static_response(compiled))
    if ir.artifact_digest == old_ir.artifact_digest:
        raise AssertionError("changed raw declarations did not invalidate response IR")
    plan_seconds, plan = _timed(
        lambda: lower_backend_plan(ir, BackendKind.CPU, torch.float64)
    )
    preparation_seconds, _ = _timed(lambda: prepare_backend_request(plan, request))
    initialization_seconds, kernels = _timed(CPUResponseKernels)
    calibration_seconds, selection = _timed(
        lambda: calibrate_executor(
            ir,
            plan,
            kernels,
            request,
            absolute_tolerance=1e-12,
            relative_tolerance=1e-12,
            warmup=1,
            iterations=max(2, min(calibration_iterations, 5)),
        )
    )
    response_seconds, execution = _timed(
        lambda: execute_selected(ir, plan, kernels, selection, request)
    )
    oracle_seconds, reference = _timed(lambda: _direct_response(field, contexts))
    maximum_error = float(
        torch.max(torch.abs(execution.result.output - reference)).item()
    )
    if maximum_error > 1e-12:
        raise ValueError("source-replacement execution failed direct response oracle")
    validation_total = time.perf_counter() - started
    execution_lifecycle = validation_total - oracle_seconds
    return {
        "execution_lifecycle_seconds": execution_lifecycle,
        "validation_total_seconds": validation_total,
        "determine_source_seconds": determination_seconds,
        "lower_sparse_source_seconds": bridge_seconds,
        "prepare_relation_incidents_seconds": incident_seconds,
        "prepare_compiled_incidents_seconds": compiled_preparation_seconds,
        "compile_identity_response_seconds": compile_seconds,
        "lower_response_ir_seconds": ir_seconds,
        "lower_backend_plan_seconds": plan_seconds,
        "prepare_backend_request_seconds": preparation_seconds,
        "executor_initialization_seconds": initialization_seconds,
        "calibration_seconds": calibration_seconds,
        "response_seconds": response_seconds,
        "direct_oracle_seconds": oracle_seconds,
        "selected_executor": selection.selected.value,
        "source_digest_changed": compiled.source_digest
        != dict(old_ir.invalidation_key.entries)["source"],
        "artifact_digest_changed": ir.artifact_digest != old_ir.artifact_digest,
        "maximum_absolute_error": maximum_error,
    }


def run_lifecycle(
    *,
    train_path: Path,
    test_path: Path,
    maximum_queries: int = 512,
    radius: int = 3,
    minimum_occurrences: int = 50,
    calibration_iterations: int = 10,
    response_iterations: int = 20,
) -> dict:
    load_train_seconds, train_data = _timed(
        lambda: load_wikitext_split("train", train_path)
    )
    load_test_seconds, test_data = _timed(
        lambda: load_wikitext_split("test", test_path)
    )
    train, train_digest = train_data
    test, test_digest = test_data
    determination_seconds, field = _timed(
        lambda: determine_declarative_field(
            train,
            radius=radius,
            minimum_occurrences=minimum_occurrences,
            normalization="joint",
        )
    )
    bridge_seconds, bridge = _timed(lambda: lower_declarative_source(field))
    extraction_seconds, occurrences = _timed(
        lambda: _held_out_occurrences(
            test, frozenset(field.terms), maximum_queries
        )
    )
    term_index = {term: index for index, term in enumerate(field.terms)}
    context_seconds, context_data = _timed(
        lambda: (
            tuple(field.incident(unit, position) for unit, position in occurrences),
            torch.tensor(
                [term_index[unit[position]] for unit, position in occurrences],
                dtype=torch.int64,
            ),
        )
    )
    contexts, targets = context_data
    incident_seconds, incidents = _timed(
        lambda: prepare_context_batch(bridge, contexts)
    )
    compile_seconds, compiled = _timed(
        lambda: compile_identity_static_response(
            bridge.source, incidents.admitted_features()
        )
    )
    compiled_preparation_seconds, source_prepared = _timed(
        lambda: compiled.prepare(incidents)
    )
    request = BackendRequest(
        source_prepared.amplitudes,
        local_indices=source_prepared.local_indices,
        valid=source_prepared.valid,
    )
    ir_seconds, ir = _timed(lambda: lower_static_response(compiled))
    direct_seconds = _median_seconds(
        lambda: _direct_response(field, contexts), BackendKind.CPU, 3
    )
    oracle_seconds, reference = _timed(lambda: _direct_response(field, contexts))
    backend_records = []
    outputs = {}
    for backend in (BackendKind.CPU, BackendKind.MPS):
        if not backend_capability(backend).available:
            continue
        record, output, _ = _backend_lifecycle(
            ir,
            request,
            reference,
            backend=backend,
            calibration_iterations=calibration_iterations,
            response_iterations=response_iterations,
        )
        backend_records.append(record)
        outputs[backend.value] = output
    quality = _decode(outputs.get("cpu", reference), targets)
    replacement = _replacement_lifecycle(
        train,
        contexts,
        ir,
        radius=radius,
        minimum_occurrences=minimum_occurrences,
        calibration_iterations=calibration_iterations,
    )
    common_seconds = sum(
        (
            load_train_seconds,
            load_test_seconds,
            determination_seconds,
            bridge_seconds,
            extraction_seconds,
            context_seconds,
            incident_seconds,
            compile_seconds,
            compiled_preparation_seconds,
            ir_seconds,
        )
    )
    shared_source_seconds = sum(
        (
            load_train_seconds,
            load_test_seconds,
            determination_seconds,
            extraction_seconds,
            context_seconds,
        )
    )
    field_specific_seconds = common_seconds - shared_source_seconds
    amortization = {}
    matched_lifecycle = {}
    for record in backend_records:
        complete_once = (
            common_seconds
            + record.plan_lowering_seconds
            + record.request_preparation_seconds
            + record.executor_initialization_seconds
            + record.calibration_seconds
        )
        optimized_extra_once = (
            field_specific_seconds
            + record.plan_lowering_seconds
            + record.request_preparation_seconds
            + record.executor_initialization_seconds
            + record.calibration_seconds
        )
        repeated_response = (
            record.response_median_seconds
            + record.output_transfer_seconds
            + record.readout_median_seconds
        )
        amortization[record.backend] = {
            str(batches): (
                complete_once + batches * repeated_response
            ) / (batches * maximum_queries)
            for batches in (1, 10, 100, 1000)
        }
        advantage = direct_seconds - (
            record.response_median_seconds + record.output_transfer_seconds
        )
        amortization[record.backend]["break_even_batches_against_direct_response"] = (
            None if advantage <= 0 else optimized_extra_once / advantage
        )
        readout = record.readout_median_seconds
        matched_lifecycle[record.backend] = {
            str(batches): {
                "compiled_total_seconds": complete_once
                + batches * repeated_response,
                "direct_total_seconds": shared_source_seconds
                + batches * (direct_seconds + readout),
            }
            for batches in (1, 10, 100, 1000)
        }
    source_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            bridge.source.rows,
            bridge.source.columns,
            bridge.source.values,
            bridge.source.quantity_metric,
            bridge.source.relation_metric,
        )
    )
    return {
        "contract": "raw-text source and response lifecycle",
        "train_sha256": train_digest,
        "test_sha256": test_digest,
        "radius": radius,
        "minimum_occurrences": minimum_occurrences,
        "queries": len(contexts),
        "terms": len(field.terms),
        "relation_features": field.relation_feature_count,
        "operator_nonzeros": field.nonzero_operator_entries,
        "selected_relation_features": compiled.accounting.compiled_features,
        "selected_operator_nonzeros": compiled.accounting.compiled_operator_nonzeros,
        "source_bytes": source_bytes,
        "compiled_response_bytes": compiled.accounting.retained_bytes,
        "phase_seconds": {
            "load_train": load_train_seconds,
            "load_test": load_test_seconds,
            "determine_source": determination_seconds,
            "lower_sparse_source": bridge_seconds,
            "extract_held_out_incidents": extraction_seconds,
            "encode_contexts_and_targets": context_seconds,
            "prepare_relation_incidents": incident_seconds,
            "compile_identity_response": compile_seconds,
            "prepare_compiled_incidents": compiled_preparation_seconds,
            "lower_response_ir": ir_seconds,
            "common_lifecycle": common_seconds,
            "shared_source_lifecycle": shared_source_seconds,
            "compiled_field_specific_lifecycle": field_specific_seconds,
            "direct_response_median": direct_seconds,
            "direct_oracle_once": oracle_seconds,
        },
        "quality": quality,
        "backends": [asdict(record) for record in backend_records],
        "source_replacement": replacement,
        "amortized_seconds_per_query": amortization,
        "matched_complete_lifecycle": matched_lifecycle,
        "boundaries": {
            "source_determination": "raw WikiText-2 local occurrence law",
            "semantic_nonlocal_relations": "not determined",
            "generative_language_model": "not established",
            "downloads": "forbidden by explicit existing parquet paths",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--maximum-queries", type=int, default=512)
    parser.add_argument("--calibration-iterations", type=int, default=10)
    parser.add_argument("--response-iterations", type=int, default=20)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_lifecycle(
                train_path=arguments.train_parquet,
                test_path=arguments.test_parquet,
                maximum_queries=arguments.maximum_queries,
                calibration_iterations=arguments.calibration_iterations,
                response_iterations=arguments.response_iterations,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
