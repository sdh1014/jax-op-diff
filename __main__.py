#!/usr/bin/env python
"""Main entry point for jax-op-diff."""

import argparse
import time
import sys


def main():
    parser = argparse.ArgumentParser(
        description="jax-op-diff"
    )
    parser.add_argument(
        "--dtypes", nargs="+",
        default=["float32", "bfloat16", "float8_e4m3fn", "float8_e5m2"],
        help="Dtypes to test",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Filter to specific categories (e.g. basic exp_trig)",
    )
    parser.add_argument(
        "--ops", nargs="+", default=None,
        help="Filter to specific op names",
    )
    parser.add_argument("--dump-dir", default="output/dumps")
    parser.add_argument("--report-dir", default="output/reports")
    parser.add_argument("--no-dump", action="store_true")
    parser.add_argument("--jax-backend", default="gpu")
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--from-dumps", default=None, metavar="DIR",
        help="Replay mode: load inputs from existing .npz dump files and "
             "compare stored outputs against fresh JAX results on --jax-backend",
    )
    args = parser.parse_args()

    # Import after argparse to allow --help without GPU init
    from precision_test.config import TestConfig
    from precision_test.runner import run_all_tests, run_replay_tests
    from precision_test.reporter import ReportGenerator

    # Import all ops to trigger registration
    import precision_test.ops.all_ops  # noqa: F401

    config = TestConfig(
        seed=args.seed,
        jax_backend=args.jax_backend,
        torch_device=args.torch_device,
        dtypes=tuple(args.dtypes),
        dump_dir=args.dump_dir,
        report_dir=args.report_dir,
    )

    replay_mode = args.from_dumps is not None
    categories = set(args.categories) if args.categories else None
    op_names = set(args.ops) if args.ops else None

    # --- Banner ---
    print("=" * 70)
    if replay_mode:
        print(f"JAX Replay: stored -> JAX ({config.jax_backend}) Precision Test")
    else:
        print(f"JAX ({config.jax_backend}) vs PyTorch ({config.torch_device}) Precision Test")
    print("=" * 70)
    print(f"  JAX backend:  {config.jax_backend}")
    if replay_mode:
        print(f"  From dumps:   {args.from_dumps}")
    else:
        print(f"  Torch device: {config.torch_device}")
    print(f"  Dtypes:       {config.dtypes}")
    print(f"  Seed:         {config.seed}")
    if args.categories:
        print(f"  Categories:   {args.categories}")
    if args.ops:
        print(f"  Ops:          {args.ops}")
    print()

    # --- Run tests ---
    start = time.time()
    if replay_mode:
        results = run_replay_tests(
            config=config,
            dump_dir=args.from_dumps,
            categories=categories,
            op_names=op_names,
        )
    else:
        results = run_all_tests(
            config=config,
            categories=categories,
            op_names=op_names,
            dump=not args.no_dump,
        )
    elapsed = time.time() - start

    # --- Reports ---
    if replay_mode:
        reporter = ReportGenerator(
            config.report_dir,
            jax_backend=config.jax_backend,
            torch_device=config.torch_device,
            title_override=f"JAX Replay: stored vs JAX ({config.jax_backend}) Precision",
        )
    else:
        reporter = ReportGenerator(
            config.report_dir,
            jax_backend=config.jax_backend,
            torch_device=config.torch_device,
        )
    csv_path = reporter.generate_csv(results)
    md_path = reporter.generate_markdown(results)
    reporter.print_console_summary(results)

    print(f"Completed {len(results)} tests in {elapsed:.1f}s")
    print(f"CSV report:      {csv_path}")
    print(f"Markdown report: {md_path}")
    if not replay_mode and not args.no_dump:
        print(f"Data dumps:      {config.dump_dir}/")


if __name__ == "__main__":
    main()
