"""Digital realization of finite information-field response laws.

Public names at this level cover source construction, exact static response,
minimal causal response, observable temporal response, and matrix-free
compilation.  Specialized reductions and processor realizations remain
available through their namespaced modules.
"""

from .causal_minimal import (
    CausalMinimalRealization,
    compile_minimal_realization,
    compile_relation_field,
)
from .matrix_free_field import (
    FactorizedIntrinsicOperator,
    MatrixFreeCompilation,
    compile_matrix_free_relation_field,
)
from .observable_response import (
    ExactGridRecurrence,
    FixedTimeGreenMap,
    ObservableSpectrum,
    SampledGreenFamily,
    compile_fixed_time_green,
    compile_grid_recurrence,
    compile_observable_spectrum,
    compile_sampled_green,
)
from .quotient_response import (
    CausalState,
    CompiledStaticResponse,
    SparseIncident,
    SparseIncidentBatch,
    SparseRelationSource,
    compile_exact_modal,
    compile_static_response,
)

__version__ = "0.1.0"

__all__ = [
    "CausalMinimalRealization",
    "CausalState",
    "CompiledStaticResponse",
    "ExactGridRecurrence",
    "FactorizedIntrinsicOperator",
    "FixedTimeGreenMap",
    "MatrixFreeCompilation",
    "ObservableSpectrum",
    "SampledGreenFamily",
    "SparseIncident",
    "SparseIncidentBatch",
    "SparseRelationSource",
    "compile_exact_modal",
    "compile_fixed_time_green",
    "compile_grid_recurrence",
    "compile_matrix_free_relation_field",
    "compile_minimal_realization",
    "compile_observable_spectrum",
    "compile_relation_field",
    "compile_sampled_green",
    "compile_static_response",
]
