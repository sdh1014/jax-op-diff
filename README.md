# jax-op-diff

[中文文档](README.zh.md)

## Introduction
jax-op-diff is a JAX-first low-level operator diff tool: it compares JAX vs PyTorch outputs across numeric precisions and input shapes on CPU/GPU (and TPU on the JAX side), and dumps reproducible artifacts for deterministic replay.

## Key Features
- JAX-first operator output comparison against PyTorch.
- Precision and shape sweeps (`float32`, `bfloat16`, `float8_*`, etc.).
- Three run modes: **compare**, **jax-only dump**, **replay**.
- Single-file `dump.h5` + replay workflow for deterministic re-checks.
- Cross-backend workflow: dump on GPU, replay on TPU (or vice versa).
- Op schema validation at import time — configuration errors fail fast.
- Automatic `CSV` and `Markdown` report generation.

## Quick Start
```bash
pip install -e .
jax-op-diff --help
```

### Three Run Modes

#### 1. Compare (default)
JAX vs PyTorch precision comparison, with optional dump.
```bash
# GPU
jax-op-diff --mode compare --jax-backend gpu --torch-device cuda

# CPU
jax-op-diff --mode compare --jax-backend cpu --torch-device cpu

# Skip dump
jax-op-diff --mode compare --no-dump
```

#### 2. JAX-only Dump
Run only JAX and dump outputs. No PyTorch needed — useful for ops without torch equivalents.
```bash
jax-op-diff --mode jax-only --jax-backend gpu --dump-dir output/dumps_gpu
```

#### 3. Replay
Replay from an existing dump, comparing stored outputs against fresh JAX execution on a (possibly different) backend.
```bash
jax-op-diff --mode replay --from-dumps output/dumps_gpu --jax-backend tpu
```

### Typical Cross-Backend Workflow
```bash
# Step 1: On GPU machine, dump JAX results (no torch needed)
jax-op-diff --mode jax-only --jax-backend gpu --dump-dir output/dumps_gpu

# Step 2: On TPU machine, replay and compare GPU outputs vs TPU outputs
jax-op-diff --mode replay --from-dumps output/dumps_gpu --jax-backend tpu
```

### Filtering
```bash
# Filter by category
jax-op-diff --categories basic exp_trig

# Filter by operator name
jax-op-diff --ops add mul exp

# Filter by dtype
jax-op-diff --dtypes float32 bfloat16
```

## Default Output
- `output/dumps/dump.h5` — HDF5 dump of all JAX outputs + inputs
- `output/reports/precision_report.csv` — Full precision report
- `output/reports/precision_report.md` — Markdown summary

## Project Structure

```
python/jax_op_diff/
  __main__.py          # Entry point: parse → config → pipeline
  cli.py               # argparse + build_config + build_filters
  core.py              # Shared data types: PrecisionResult, RunFilters, CaseData
  config.py            # TestConfig, dtype maps, get_shapes_for_op
  op_registry.py       # OpSpec, register(), schema validation, generate_inputs
  pipeline.py          # Flow orchestration: run_compare / run_jax_only / run_replay / report
  dump_store.py        # DumpStore: HDF5 read/write with lazy filtering
  reporter.py          # CSV, Markdown, console summary generation
  executor/
    __init__.py        # Strict re-export + __all__
    engine.py          # execute_single_test, execute_jax_only, replay_single_case
    metrics.py         # compute_metrics, compute_all_close (pure numpy)
  ops/
    all_ops.py         # Operator registration manifest

tests/
  unit/                # Pure numpy, no GPU, runs in seconds
    test_metrics.py
    test_op_schema.py
    test_config.py
  integration/         # Needs JAX+Torch, CPU sufficient
    test_dump_store.py
    test_direct_dtype_comparison.py
  e2e/                 # Full pipeline tests
    test_pipeline.py
```

## Testing

```bash
# Unit tests only (pure numpy, fast)
pytest tests/unit/ -v

# Integration tests (needs JAX + Torch)
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

## Architecture

```
__main__.py
  ├─→ cli.py            (parse_args, build_config, build_filters)
  └─→ pipeline.py       (run_compare, run_jax_only, run_replay, report)
        ├─→ core.py      (PrecisionResult, CaseData, RunFilters)
        ├─→ config.py    (TestConfig, get_shapes_for_op)
        ├─→ op_registry.py (get_all_ops, schema validation)
        ├─→ executor/    (execute_single_test, execute_jax_only, replay_single_case)
        ├─→ dump_store.py (DumpStore)
        └─→ reporter.py  (ReportGenerator)
```

Dependency direction is strictly top-down, no cycles. `core.py` sits at the bottom, depended on by all business modules.
