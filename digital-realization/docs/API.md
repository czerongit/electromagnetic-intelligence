# Package map

`information_field` exposes the common finite response path directly. More
specialized constructions live in namespaced modules so their mathematical
preconditions remain visible.

| Mathematical object or operation | Public module |
|---|---|
| Sparse finite relation source and incident coordinates | `information_field.quotient_response` |
| Exact static relation-coordinate response | `information_field.quotient_response` |
| Exact modal evolution for the intrinsic operator | `information_field.quotient_response` |
| Minimal reachable and observable carrier | `information_field.causal_minimal` |
| Spectral residues, fixed-time maps, and grid recurrence | `information_field.observable_response` |
| Factorized intrinsic operator and block-Krylov construction | `information_field.matrix_free_field` |
| Invariant and carrier-augmenting source updates | `information_field.incremental_field`, `information_field.augmented_update` |
| Connected-component restriction | `information_field.local_field` |
| Involutive symmetry sectors and Noether checks | `information_field.symmetry_field` |
| Algebraic lower bounds | `information_field.field_lower_bounds` |
| Processor-independent response representation | `information_field.response_ir` |
| CPU, Metal, and CUDA lowering | `information_field.response_backends` |
| Tensor and processor-specific execution | `information_field.profiled_response`, `information_field.native_response_kernels`, `information_field.wide_response_kernels` |
| Corpus-determined local-occurrence source | `information_field.geometric_observation` |
| Sparse corpus lifecycle | `information_field.raw_field_lifecycle` |
| Retrieval source and matched input generator | `information_field.retrieval` |
| Static executor selection and event composition | `information_field.reduction` |

Every reduction checks its own structural conditions. A presently zero response
does not authorize removal of a source direction; complete response at the
declared incident and readout ports determines admissible reduction.
