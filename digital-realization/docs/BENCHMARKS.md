# Reproducing the computational experiments

Run commands from this directory after installing the package. Output paths
should point outside `results/published`; those files are immutable records of
the reported runs.

## Structural reductions

```bash
python benchmarks/minimal_causal_carrier.py --dimensions 128,512,2048 --output runs/minimal.json
python benchmarks/fixed_time_response.py --frequencies 4,16,64 --output runs/fixed-time.json
python benchmarks/matrix_free_construction.py --dimensions 128,512,2048 --output runs/matrix-free.json
python benchmarks/invariant_source_update.py --dimensions 128,512,2048 --output runs/invariant-update.json
python benchmarks/augmented_source_update.py --dimensions 128,512,2048,8192 --output runs/augmented-update.json
python benchmarks/component_restriction.py --components 8,32,128,512 --output runs/components.json
python benchmarks/symmetry_reduction.py --pairs 8,16,32,64,128,256 --output runs/symmetry.json
```

## Processor execution

```bash
python benchmarks/cpu_kernels.py
python benchmarks/processor_kernels.py
python benchmarks/cuda_kernels.py
python benchmarks/wide_processor_kernels.py --backend cuda
python benchmarks/wide_processor_kernels.py --backend mps
```

Unavailable processors are rejected or skipped by the corresponding runner.
CPU uses FP64 in the reported accuracy comparison; Metal and CUDA use FP32
except for the FP16 retrieval comparison.

## Retrieval and FlashAttention

Inspect the protocol without executing CUDA kernels:

```bash
python benchmarks/retrieval_flashattention.py --output-dir runs/retrieval
```

Execute on a CUDA host after editing `machine.example.json`:

```bash
python benchmarks/retrieval_flashattention.py \
  --machine machine.example.json \
  --output-dir runs/retrieval \
  --execute
```

Both paths must attain at least 95 percent retrieval accuracy before a latency
ratio is emitted. Reported runs attained 100 percent. Timings begin after the
relation coordinates and the query, key, and value arrays have been formed;
field-source construction and transfer are recorded separately. PyTorch is
forced to its FlashAttention scaled-dot-product-attention backend.

## Constant-source temporal composition

```bash
python benchmarks/temporal_event_composition.py \
  --machine machine.example.json \
  --output-dir runs/temporal \
  --execute
```

Regular-grid propagation and event composition receive the same affine
transitions, interval lengths, initial state, and final-state readout. Matrix
power construction is measured separately.

## WikiText-2 lifecycle

```bash
python benchmarks/wikitext_lifecycle.py \
  --train-parquet /path/to/train-00000-of-00001.parquet \
  --test-parquet /path/to/test-00000-of-00001.parquet
```

Training text alone determines the sparse relation source. Held-out targets are
used only for evaluation. Both parquet files are supplied explicitly and
checked against the pinned digests in
`information_field.geometric_observation.wikipedia`.
