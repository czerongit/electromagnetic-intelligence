from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkloadKind(str, Enum):
    STATIC = "static"
    FIXED_TIME_CONSTANT = "fixed-time-constant"
    FIXED_TIME_IMPULSE = "fixed-time-impulse"
    SAMPLED_CONSTANT = "sampled-constant"
    REGULAR_GRID_PIECEWISE_CONSTANT = "regular-grid-piecewise-constant"
    IRREGULAR_PIECEWISE_CONSTANT = "irregular-piecewise-constant"
    TIME_DEPENDENT_GEOMETRY = "time-dependent-geometry"


@dataclass(frozen=True)
class TemporalWorkload:
    kind: WorkloadKind
    times: tuple[float, ...] = ()
    step: float | None = None
    zero_past: bool = True

    def validate(self) -> None:
        if any(time < 0 for time in self.times):
            raise ValueError("response times must be nonnegative")
        if self.kind in {
            WorkloadKind.FIXED_TIME_CONSTANT,
            WorkloadKind.FIXED_TIME_IMPULSE,
        } and len(self.times) != 1:
            raise ValueError("fixed-time execution requires exactly one response time")
        if self.kind == WorkloadKind.SAMPLED_CONSTANT and not self.times:
            raise ValueError("sampled execution requires at least one response time")
        if self.kind == WorkloadKind.REGULAR_GRID_PIECEWISE_CONSTANT:
            if self.step is None or self.step <= 0:
                raise ValueError("regular-grid execution requires a positive step")
        elif self.step is not None:
            raise ValueError("a grid step belongs only to regular-grid execution")

    @property
    def source_fixed_compilation_supported(self) -> bool:
        return self.kind not in {
            WorkloadKind.IRREGULAR_PIECEWISE_CONSTANT,
            WorkloadKind.TIME_DEPENDENT_GEOMETRY,
        }

    @property
    def requires_state(self) -> bool:
        return self.kind in {
            WorkloadKind.REGULAR_GRID_PIECEWISE_CONSTANT,
            WorkloadKind.IRREGULAR_PIECEWISE_CONSTANT,
            WorkloadKind.TIME_DEPENDENT_GEOMETRY,
        }
