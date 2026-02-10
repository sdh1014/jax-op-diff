# jax-op-diff

[English README](README.md)

## 简介
jax-op-diff 是一个以 JAX 为核心的底层算子差异对比工具：在不同数值精度和输入形状下，对比 JAX 与 PyTorch 在 CPU/GPU（以及 JAX 侧 TPU）上的输出差异，并将输入与结果 dump 为可复现数据，用于确定性复测。

## Key Features
- 以 JAX 为主，对比同一算子在 JAX 与 PyTorch 的输出差异。
- 覆盖多精度（如 `float32`、`bfloat16`、`float8_*`）与多输入形状。
- 支持 dump 与 replay：复用同一输入进行确定性重放对比。
- 自动生成 `CSV` 和 `Markdown` 报告。

## Quick Start
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
