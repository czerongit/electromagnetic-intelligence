# Published digital measurements

This table records the numerical claims supported by
`results/published`. Ratios divide comparison time by reduced-response time;
larger values favor the field realization. Exact identities and preconditions
are established separately by the test suite.

| Experiment | Reported result | Record |
|---|---:|---|
| Relation-coordinate retrieval, batch | 2.19–3.01× over FlashAttention | `retrieval-flashattention/records.jsonl` |
| Relation-coordinate retrieval, incremental | 3.13–3.64× over FlashAttention | `retrieval-flashattention/records.jsonl` |
| Retrieval accuracy | 100% for both realizations in all four conditions | `retrieval-flashattention/records.jsonl` |
| Minimal causal carrier | 1.59–3.97× response reduction | `minimal-causal-carrier.json` |
| Fixed-time response | 2.28–26.55× response reduction across reported CPU and accelerator workloads | `fixed-time-response.json` |
| Matrix-free construction | 2.83–261.31× construction reduction | `matrix-free-construction.json` |
| Invariant source update | 1.64–2.90× construction reduction | `invariant-source-update.json` |
| Carrier-augmenting update | 1.25–2.29× construction reduction | `augmented-source-update.json` |
| Component restriction | Up to 3.94× construction reduction | `component-restriction.json` |
| Symmetry decomposition | 1.15× at dimension 512; slower below crossover | `symmetry-reduction.json` |
| Processor numerical error | CPU FP64 ≤ 2.22e-16; Metal/CUDA FP32 ≤ 1.51e-7 | `processor-independent-qualification.json` |
| Narrow recurrence kernels | Metal 3.10–4.40×; CUDA 2.15–7.88×; CPU 1.08–8.82× in favorable tested widths | processor kernel records |
| Constant-source event composition | 1.49×, 2.49×, and 6.90× at 256, 1,024, and 4,096 steps | `temporal-event-composition/records.jsonl` |
| WikiText held-out response | 99.22% coverage, 15.23% top-1, 34.57% top-5 | `wikitext-lifecycle.json` |

Latency measurements apply to the declared input boundary, device, precision,
and workload. Source-construction and transfer measurements remain separate
where the manuscript reports them separately. Structural reductions remain
exact only while their stated source, incident, readout, time, locality, or
symmetry conditions hold.
