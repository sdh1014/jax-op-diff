#!/usr/bin/env python
"""Main entry point for jax-op-diff."""

import time

from .cli import parse_args, build_config, build_filters
from . import pipeline


def main():
    args = parse_args()
    config = build_config(args)
    filters = build_filters(args)

    # Trigger op registration + schema validation
    import jax_op_diff.ops.all_ops  # noqa: F401

    # --- Banner ---
    print("=" * 70)
    if args.mode == "jax-precision":
        print(f"JAX Precision: CPU fp32 (ground truth) vs {config.jax_backend}")
    elif args.mode == "replay":
        print(f"JAX Replay: stored -> JAX ({config.jax_backend}) Precision Test")
    elif args.mode == "jax-only":
        print(f"JAX-only dump mode (backend: {config.jax_backend})")
    else:
        print(f"JAX ({config.jax_backend}) vs PyTorch ({config.torch_device}) Precision Test")
    print("=" * 70)
    print(f"  JAX backend:  {config.jax_backend}")
    if args.mode == "jax-precision" and args.from_dumps:
        print(f"  From dumps:   {args.from_dumps}")
    elif args.mode == "replay":
        print(f"  From dumps:   {args.from_dumps}")
    elif args.mode not in ("jax-only", "jax-precision"):
        print(f"  Torch device: {config.torch_device}")
    print(f"  Dtypes:       {config.dtypes}")
    print(f"  Seed:         {config.seed}")
    print()

    # --- Run ---
    start = time.time()

    if args.mode == "jax-only":
        stats = pipeline.run_jax_only(config, filters)
        pipeline.print_dump_stats(stats)
    elif args.mode == "jax-precision":
        title = f"JAX Precision: CPU fp32 (ground truth) vs {config.jax_backend}"
        if args.from_dumps:
            results = pipeline.run_jax_precision_replay(
                config, filters, args.from_dumps)
        else:
            results = pipeline.run_jax_precision(
                config, filters, dump=not args.no_dump)
        pipeline.report(config, results, title_override=title)
    elif args.mode == "replay":
        results = pipeline.run_replay(config, filters, args.from_dumps)
        pipeline.report(
            config, results,
            title_override=f"JAX Replay: stored vs JAX ({config.jax_backend}) Precision",
        )
    else:
        results = pipeline.run_compare(config, filters, dump=not args.no_dump)
        pipeline.report(config, results)

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")
    if args.mode in ("compare", "jax-precision") and not args.no_dump and not args.from_dumps:
        print(f"Data dump file:  {config.dump_dir}/dump.h5")


if __name__ == "__main__":
    main()
