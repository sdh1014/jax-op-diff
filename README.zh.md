# jax-op-diff

[English README](README.md)

## 简介
jax-op-diff 是一个以 JAX 为核心的底层算子差异对比工具：在不同数值精度和输入形状下，对比 JAX 与 PyTorch 在 CPU/GPU（以及 JAX 侧 TPU）上的输出差异，并将输入与结果 dump 为可复现数据，用于确定性复测。

## 核心特性
- 以 JAX 为主，对比同一算子在 JAX 与 PyTorch 的输出差异。
- 覆盖多精度（`float32`、`bfloat16`、`float8_*`）与多输入形状。
- 三种运行模式：**compare**（对比）、**jax-only**（仅 JAX dump）、**replay**（回放）。
- 单文件 `dump.h5` + replay：复用同一输入进行确定性重放对比。
- 跨后端工作流：在 GPU 上 dump，在 TPU 上 replay（反之亦然）。
- Op Schema 注册期校验 —— 配置错误在 import 时立即暴露。
- 自动生成 `CSV` 和 `Markdown` 报告。

## 快速开始
```bash
pip install -e .
jax-op-diff --help
```

### 三种运行模式

#### 1. Compare（默认）
JAX vs PyTorch 精度对比，可选 dump。
```bash
# GPU
jax-op-diff --mode compare --jax-backend gpu --torch-device cuda

# CPU
jax-op-diff --mode compare --jax-backend cpu --torch-device cpu

# 不生成 dump
jax-op-diff --mode compare --no-dump
```

#### 2. JAX-only Dump
仅运行 JAX 并 dump 输出。不需要 PyTorch —— 适用于没有 torch 对应实现的算子。
```bash
jax-op-diff --mode jax-only --jax-backend gpu --dump-dir output/dumps_gpu
```

#### 3. Replay（回放）
从已有 dump 回放，将存储的输出与当前 JAX 后端的新计算结果对比。
```bash
jax-op-diff --mode replay --from-dumps output/dumps_gpu --jax-backend tpu
```

### 典型跨后端工作流
```bash
# 步骤 1：在 GPU 机器上，只跑 JAX 并 dump 结果（不需要 torch）
jax-op-diff --mode jax-only --jax-backend gpu --dump-dir output/dumps_gpu

# 步骤 2：在 TPU 机器上，从 dump 回放，对比 GPU 输出 vs TPU 输出
jax-op-diff --mode replay --from-dumps output/dumps_gpu --jax-backend tpu
```

### 过滤选项
```bash
# 按类别过滤
jax-op-diff --categories basic exp_trig

# 按算子名称过滤
jax-op-diff --ops add mul exp

# 按数据类型过滤
jax-op-diff --dtypes float32 bfloat16
```

## 默认输出文件
- `output/dumps/dump.h5` — 所有 JAX 输出 + 输入的 HDF5 dump
- `output/reports/precision_report.csv` — 完整精度报告
- `output/reports/precision_report.md` — Markdown 摘要

## 项目结构

```
python/jax_op_diff/
  __main__.py          # 入口：parse → config → pipeline
  cli.py               # argparse + build_config + build_filters
  core.py              # 跨模块共享数据类型：PrecisionResult, RunFilters, CaseData
  config.py            # TestConfig, dtype 映射, get_shapes_for_op
  op_registry.py       # OpSpec, register(), Schema 校验, generate_inputs
  pipeline.py          # 流程编排：run_compare / run_jax_only / run_replay / report
  dump_store.py        # DumpStore: HDF5 读写 + 惰性过滤
  reporter.py          # CSV、Markdown、控制台摘要生成
  executor/
    __init__.py        # 严格 re-export + __all__
    engine.py          # execute_single_test, execute_jax_only, replay_single_case
    metrics.py         # compute_metrics, compute_all_close（纯 numpy）
  ops/
    all_ops.py         # 算子注册清单

tests/
  unit/                # 纯 numpy，无 GPU 依赖，秒级完成
    test_metrics.py
    test_op_schema.py
    test_config.py
  integration/         # 需要 JAX+Torch，CPU 即可
    test_dump_store.py
    test_direct_dtype_comparison.py
  e2e/                 # 完整 pipeline 测试
    test_pipeline.py
```

## 测试

```bash
# 仅单元测试（纯 numpy，快速）
pytest tests/unit/ -v

# 集成测试（需要 JAX + Torch）
pytest tests/integration/ -v

# 全量测试
pytest tests/ -v
```

## 架构

```
__main__.py
  ├─→ cli.py            (parse_args, build_config, build_filters)
  └─→ pipeline.py       (run_compare, run_jax_only, run_replay, report)
        ├─→ core.py      (PrecisionResult, CaseData, RunFilters)
        ├─→ config.py    (TestConfig, get_shapes_for_op)
        ├─→ op_registry.py (get_all_ops, Schema 校验)
        ├─→ executor/    (execute_single_test, execute_jax_only, replay_single_case)
        ├─→ dump_store.py (DumpStore)
        └─→ reporter.py  (ReportGenerator)
```

依赖方向严格自上而下，无循环。`core.py` 位于最底层，被所有业务模块引用。
