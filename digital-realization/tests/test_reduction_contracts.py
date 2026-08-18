import torch

from information_field.reduction.static_selector import select_static_executor
from information_field.reduction.temporal import (
    TemporalCondition,
    correctness_record,
    event_composed_response,
    make_temporal_fixture,
    matrix_powers,
    regular_grid_response,
)


def test_static_selector_retains_columns_for_sparse_full_rank_source():
    columns = torch.eye(8, dtype=torch.float64).expand(3, -1, -1).clone()
    selections = select_static_executor(columns, support=1)
    assert all(selection.rank == 8 for selection in selections)
    assert all(selection.executor == "columns" for selection in selections)
    assert all(selection.column_work == 8 for selection in selections)
    assert all(selection.rank_factor_work == 72 for selection in selections)


def test_static_selector_uses_factor_only_when_it_removes_work():
    left = torch.arange(1.0, 17.0, dtype=torch.float64).reshape(8, 2)
    right = torch.arange(1.0, 17.0, dtype=torch.float64).reshape(2, 8)
    columns = (left @ right).unsqueeze(0)
    dense = select_static_executor(columns, support=8)[0]
    sparse = select_static_executor(columns, support=1)[0]
    assert dense.rank == sparse.rank == 2
    assert dense.executor == "rank-factor"
    assert sparse.executor == "columns"


def test_event_composition_preserves_matched_stateful_response():
    condition = TemporalCondition("test", 2, 4, (3, 5, 2))
    transitions, lengths, initial, _ = make_temporal_fixture(condition)
    regular = regular_grid_response(transitions, lengths, initial)
    event = event_composed_response(matrix_powers(transitions, lengths), initial)
    assert torch.max(torch.abs(event - regular)).item() < 1e-12


def test_temporal_correctness_record_keeps_timing_claim_separate():
    result = correctness_record(TemporalCondition("test", 2, 4, (3, 5, 2)))
    assert result["state_max_abs"] < 1e-12
    assert result["observation_max_abs"] < 1e-12
    assert result["state_application_reduction"] == 10 / 3
