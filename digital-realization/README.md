# Information Field: Digital Realization

This directory contains the reusable digital realization and the computational
experiments reported in *A Dynamical Field Theory of Information and
Attention*. One Python distribution provides finite source construction,
relation-coordinate response, causal reduction, temporal response maps,
matrix-free compilation, exact source updates, locality and symmetry
reductions, and processor-specific execution.

Field response requires no fitted parameters. Its inputs are relation operators,
incident coordinates, readouts, carrier metrics, and, for temporal response,
initial data and clock parameters.

## Install

Python 3.11 or later and PyTorch are required.

```bash
python -m pip install -e .
```

WikiText experiments additionally require:

```bash
python -m pip install -e '.[corpus]'
```

Processor-specific C++ and CUDA extensions additionally require a compatible
compiler toolchain and Ninja:

```bash
python -m pip install -e '.[native]'
```

## Minimal response

```python
import torch
from information_field import (
    SparseIncidentBatch,
    SparseRelationSource,
    compile_static_response,
)

D = SparseRelationSource.from_dense(
    torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
)
q = SparseIncidentBatch(
    torch.tensor([[0], [1]]),
    torch.tensor([[3.0], [4.0]], dtype=torch.float64),
    torch.ones((2, 1), dtype=torch.bool),
)
response = compile_static_response(D, torch.eye(2, dtype=torch.float64), q.admitted_features())
print(response.run(q))
```

The complete example is in
[`examples/basic_static_response.py`](examples/basic_static_response.py).

## Contents

- `src/information_field` contains the installable package.
- `benchmarks` contains independent consumers of its public modules.
- `results/published` contains the records underlying the paper's reported
  digital measurements, with SHA-256 checksums.
- `tests` verifies exact identities, response equivalence, update conditions,
  numerical precision, and public release behavior.
- `docs/API.md` maps the mathematical objects to package modules.
- `docs/BENCHMARKS.md` gives reproduction commands and measurement boundaries.
- `docs/PUBLISHED_RESULTS.md` maps the paper's numerical claims to records.

## Verify

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m build
```

Tests requiring unavailable CUDA hardware are skipped. Benchmark result files
are never overwritten by the test suite.

## Data and network policy

No benchmark downloads data automatically. WikiText-2 runners require explicit
paths to the pinned parquet files and verify their SHA-256 digests before use.
CUDA runners require an explicit machine description when execution is
requested. Dry runs write only protocol manifests.

## Published measurements

Published records include the relation-coordinate/FlashAttention retrieval
comparison, minimal causal carrier, fixed-time response, matrix-free
construction, source updates, component and symmetry reductions, processor
kernels, WikiText source construction and response, and constant-source event
composition. See [`results/published/README.md`](results/published/README.md)
for the correspondence between files and manuscript claims.

## Citation

Theory and release archive:
[https://doi.org/10.5281/zenodo.21986663](https://doi.org/10.5281/zenodo.21986663)

Repository-wide publication and attribution terms are stated in the parent
[`README.md`](../README.md).
