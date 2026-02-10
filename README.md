# jax-op-diff

[中文文档](README.zh.md)

## Introduction
jax-op-diff is a JAX-first low-level operator diff tool: it compares JAX vs PyTorch outputs across numeric precisions and input shapes on CPU/GPU (and TPU on the JAX side), and dumps reproducible artifacts for deterministic replay.

## Key Features
- JAX-first operator output comparison against PyTorch.
- Precision and shape sweeps (`float32`, `bfloat16`, `float8_*`, etc.).
- Dump + replay workflow for deterministic re-checks.
- Automatic `CSV` and `Markdown` report generation.

## Quick Start
```bash
pip install -e .
jax-op-diff --help
```

```bash
# Standard compare (GPU)
jax-op-diff --jax-backend gpu --torch-device cuda

# Standard compare (CPU)
jax-op-diff --jax-backend cpu --torch-device cpu

# Replay mode: deterministic verification from existing dumps
jax-op-diff --from-dumps output/dumps --jax-backend gpu --report-dir output/replay_reports
```
