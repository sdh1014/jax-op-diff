"""Flow orchestration. The single entry point for all run modes.

Does not depend on cli.py or argparse. Only accepts domain objects
(TestConfig, RunFilters).
"""

import dataclasses
from typing import List, Optional

from .core import PrecisionResult, RunFilters
from .config import TestConfig, get_shapes_for_op
from .dump_store import DumpStore
from .executor import execute_single_test, execute_jax_only, replay_single_case
from .op_registry import get_all_ops
from .reporter import ReportGenerator


# ---- compare mode ----


def run_compare(config: TestConfig, filters: RunFilters,
                dump: bool = True) -> List[PrecisionResult]:
    """JAX vs Torch precision comparison. Optionally dumps JAX outputs."""
    ops = _select_ops(get_all_ops(), filters)
    dumper = None
    if dump:
        dumper = DumpStore(config.dump_dir)
        dumper.create()
    results: List[PrecisionResult] = []
    total = sum(len(get_shapes_for_op(op, config)) * len(config.dtypes) for op in ops)
    done = 0

    print(f"Running {total} test cases across {len(ops)} operators...")

    for op in ops:
        for dtype_key in config.dtypes:
            for shape in get_shapes_for_op(op, config):
                result, jax_output = execute_single_test(
                    op, shape, dtype_key, config, config.seed)
                results.append(result)
                if dumper and jax_output is not None and not result.error_msg:
                    try:
                        dumper.append_case(op, shape, dtype_key, config.seed, jax_output)
                    except Exception:
                        pass  # dump failure is non-critical
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  Progress: {done}/{total} ({100 * done // total}%)")
    return results


# ---- jax-only dump mode ----


@dataclasses.dataclass
class DumpStats:
    """Statistics for jax-only dump mode (no metrics, just counts)."""
    total: int = 0
    dumped: int = 0
    errors: int = 0


def run_jax_only(config: TestConfig, filters: RunFilters) -> DumpStats:
    """Run only JAX and dump outputs. No torch needed, no PrecisionResult."""
    ops = _select_ops(get_all_ops(), filters)
    dumper = DumpStore(config.dump_dir)
    dumper.create()
    stats = DumpStats()

    for op in ops:
        for dtype_key in config.dtypes:
            for shape in get_shapes_for_op(op, config):
                stats.total += 1
                jax_output = execute_jax_only(op, shape, dtype_key, config)
                if jax_output is not None:
                    dumper.append_case(op, shape, dtype_key, config.seed, jax_output)
                    stats.dumped += 1
                else:
                    stats.errors += 1
    return stats


def print_dump_stats(stats: DumpStats) -> None:
    print(f"JAX-only dump complete: {stats.dumped}/{stats.total} cases dumped"
          f" ({stats.errors} errors)")


# ---- replay mode ----


def run_replay(config: TestConfig, filters: RunFilters,
               dump_dir: str) -> List[PrecisionResult]:
    """Replay from dump, compare stored output vs fresh JAX output."""
    op_map = {op.name: op for op in get_all_ops()}
    store = DumpStore(dump_dir)
    results: List[PrecisionResult] = []
    cases = list(store.iter_cases(filters=filters))

    if not cases:
        print(f"  No dump cases found in {dump_dir}")
        return []

    print(f"Found {len(cases)} dump cases in {dump_dir}")
    print(f"Replaying on JAX backend: {config.jax_backend}")

    for i, case in enumerate(cases, 1):
        result = replay_single_case(case, op_map, config)
        results.append(result)
        if i % 50 == 0 or i == len(cases):
            print(f"  Progress: {i}/{len(cases)} ({100 * i // len(cases)}%)")
    return results


# ---- report ----


def report(config: TestConfig, results: List[PrecisionResult],
           title_override: Optional[str] = None) -> None:
    """Generate CSV, Markdown, and console summary."""
    reporter = ReportGenerator(
        config.report_dir,
        jax_backend=config.jax_backend,
        torch_device=config.torch_device,
        title_override=title_override,
    )
    csv_path = reporter.generate_csv(results)
    md_path = reporter.generate_markdown(results)
    reporter.print_console_summary(results)
    print(f"CSV report:      {csv_path}")
    print(f"Markdown report: {md_path}")


# ---- internal helpers ----


def _select_ops(ops, filters: RunFilters):
    if filters.categories:
        ops = [op for op in ops if op.category in filters.categories]
    if filters.op_names:
        ops = [op for op in ops if op.name in filters.op_names]
    return ops
