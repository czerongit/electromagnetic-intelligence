# Published result records

Files in this directory are the unchanged machine-readable records underlying
the digital measurements reported in the manuscript. `CHECKSUMS.sha256` binds
their bytes. Historical schema strings inside these records identify the
internal experiment in which a result was produced; they are provenance, not
public package names.

| Record | Manuscript result |
|---|---|
| `retrieval-flashattention/` | Relation-coordinate response versus forced FP16 FlashAttention |
| `minimal-causal-carrier.json` | Reachable and observable carrier reduction |
| `fixed-time-response.json` | Fixed-time Green response |
| `matrix-free-construction.json` | Factorized block-Krylov construction |
| `invariant-source-update.json` | Exact update inside an invariant carrier |
| `augmented-source-update.json` | Exact update after small carrier expansion |
| `component-restriction.json` | Disconnected-component restriction |
| `symmetry-reduction.json` | Involutive symmetry-sector reduction |
| `processor-independent-qualification.json` | Cross-backend response-map accuracy |
| `execution-traces.json` | Tensor execution traces |
| `cpu-kernels.json`, `metal-kernels.json`, `cuda-kernels.json` | Narrow processor-specific recurrence kernels |
| `metal-wide-kernels.json`, `cuda-wide-kernels.json` | Wide recurrence crossover |
| `wikitext-lifecycle.json` | WikiText source construction and response lifecycle |
| `temporal-event-composition/` | Constant-source event composition |

Latency records are observations of their declared hardware and software
environment, not processor-independent constants. Exact response identities
and reduction conditions are tested independently of those timings.
