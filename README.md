# jax-op-diff

[English](#en) | [中文](#zh)

<a id="en"></a>
## English

### Introduction
jax-op-diff is a JAX-first low-level operator diff tool: it compares JAX vs PyTorch outputs across numeric precisions and input shapes on CPU/GPU (and TPU on the JAX side), and dumps reproducible artifacts for deterministic replay.

### Key Features
- JAX-first operator output comparison against PyTorch.
- Precision and shape sweeps (`float32`, `bfloat16`, `float8_*`, etc.).
- Dump + replay workflow for deterministic re-checks.
- Automatic `CSV` and `Markdown` report generation.

### Quick Start
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

Jump to [中文](#zh).

<a id="zh"></a>
## 中文

### 简介
jax-op-diff 是一个以 JAX 为核心的底层算子差异对比工具：在不同数值精度和输入形状下，对比 JAX 与 PyTorch 在 CPU/GPU（以及 JAX 侧 TPU）上的输出差异，并将输入与结果 dump 为可复现数据，用于确定性复测。

### Key Features
- 以 JAX 为主，对比同一算子在 JAX 与 PyTorch 的输出差异。
- 覆盖多精度（如 `float32`、`bfloat16`、`float8_*`）与多输入形状。
- 支持 dump 与 replay：复用同一输入进行确定性重放对比。
- 自动生成 `CSV` 和 `Markdown` 报告。

### Quick Start
```bash
pip install -e .
jax-op-diff --help
```

```bash
# 常规对比（GPU）
jax-op-diff --jax-backend gpu --torch-device cuda

# 常规对比（CPU）
jax-op-diff --jax-backend cpu --torch-device cpu

# 回放模式：使用已有 dump 做确定性复测
jax-op-diff --from-dumps output/dumps --jax-backend gpu --report-dir output/replay_reports
```

Back to [English](#en).
