"""CLI argument parsing and domain object construction.

args lifecycle stops at __main__.py. Pipeline only sees TestConfig + RunFilters.
"""

import argparse

from .core import RunFilters


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="jax-op-diff")
    parser.add_argument(
        "--mode", choices=("compare", "jax-only", "replay"),
        default="compare",
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
    parser.add_argument("--jax-backend", default="gpu")
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dump-dir", default="output/dumps")
    parser.add_argument("--report-dir", default="output/reports")
    parser.add_argument("--no-dump", action="store_true")
    parser.add_argument(
        "--from-dumps", default=None, metavar="DIR",
        help="Replay mode: load inputs from dump.h5 and "
             "compare stored outputs against fresh JAX results on --jax-backend",
    )
    args = parser.parse_args(argv)

    # Strict mode constraints: no silent fallback
    if args.mode == "replay" and not args.from_dumps:
        parser.error("--mode replay requires --from-dumps DIR")
    if args.mode in ("compare", "jax-only") and args.from_dumps is not None:
        parser.error("--from-dumps is only valid when --mode replay")
    if args.mode == "jax-only" and args.no_dump:
        parser.error("--no-dump is invalid when --mode jax-only")

    # Backwards compat: --from-dumps without --mode implies replay
    if args.from_dumps is not None and args.mode == "compare":
        args.mode = "replay"

    return args


def build_config(args: argparse.Namespace):
    """args -> TestConfig. The single conversion point."""
    from .config import TestConfig  # Lazy import: --help doesn't trigger GPU init
    return TestConfig(
        seed=args.seed,
        jax_backend=args.jax_backend,
        torch_device=args.torch_device,
        dtypes=tuple(args.dtypes),
        dump_dir=args.dump_dir,
        report_dir=args.report_dir,
    )


def build_filters(args: argparse.Namespace) -> RunFilters:
    """args -> RunFilters. The single conversion point."""
    return RunFilters(
        categories=set(args.categories) if args.categories else None,
        op_names=set(args.ops) if args.ops else None,
    )
