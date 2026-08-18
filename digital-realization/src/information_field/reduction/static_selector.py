from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class BatchSelection:
    batch: int
    relations: int
    outputs: int
    support: int
    rank: int
    column_work: int
    rank_factor_work: int
    executor: str


def numerical_rank(matrix: Tensor) -> int:
    """Return the standard scale-aware numerical rank of one response matrix."""

    value = matrix.detach().to(device="cpu", dtype=torch.float64)
    singular = torch.linalg.svdvals(value)
    if singular.numel() == 0:
        return 0
    tolerance = (
        max(value.shape)
        * torch.finfo(value.dtype).eps
        * float(singular.max().item())
    )
    return int(torch.count_nonzero(singular > tolerance).item())


def select_static_executor(observed_columns: Tensor, support: int) -> tuple[BatchSelection, ...]:
    """Choose the cheaper exact response realization for every batch member.

    ``observed_columns[b]`` has relation-by-output shape.  A support-k incident
    costs k*z scalar products under column gathering.  A rank-r factor costs
    r*(k+z).  Ties remain on columns because a factor adds retained state and a
    second execution stage without removing arithmetic.
    """

    if observed_columns.ndim != 3:
        raise ValueError("observed columns require batch by relation by output")
    if support < 0:
        raise ValueError("support must be nonnegative")
    batches, relations, outputs = observed_columns.shape
    selections = []
    for batch in range(batches):
        rank = numerical_rank(observed_columns[batch])
        column_work = support * outputs
        factor_work = rank * (support + outputs)
        executor = "rank-factor" if factor_work < column_work else "columns"
        selections.append(
            BatchSelection(
                batch=batch,
                relations=relations,
                outputs=outputs,
                support=support,
                rank=rank,
                column_work=column_work,
                rank_factor_work=factor_work,
                executor=executor,
            )
        )
    return tuple(selections)


def render_static_selection(observed_columns: Tensor, support: int) -> dict:
    selections = select_static_executor(observed_columns, support)
    executors = {selection.executor for selection in selections}
    return {
        "batch_selections": [asdict(selection) for selection in selections],
        "selected_executor": next(iter(executors)) if len(executors) == 1 else "mixed",
        "all_full_relation_rank": all(
            selection.rank == selection.relations for selection in selections
        ),
        "rank_factor_selected": any(
            selection.executor == "rank-factor" for selection in selections
        ),
    }
