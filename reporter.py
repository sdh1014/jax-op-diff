"""Report generation: CSV, Markdown, console summary."""

import csv
import dataclasses
import os
from typing import List

from .executor import PrecisionResult


class ReportGenerator:
    def __init__(self, report_dir: str, jax_backend: str = "gpu",
                 torch_device: str = "cuda", title_override: str = None):
        self.report_dir = report_dir
        self.jax_backend = jax_backend
        self.torch_device = torch_device
        self._title_override = title_override
        os.makedirs(report_dir, exist_ok=True)

    @property
    def title(self) -> str:
        if self._title_override:
            return self._title_override
        return f"JAX ({self.jax_backend}) vs PyTorch ({self.torch_device}) Precision"

    def generate_csv(self, results: List[PrecisionResult],
                     filename: str = "precision_report.csv"):
        filepath = os.path.join(self.report_dir, filename)
        fieldnames = [
            "op_name", "category", "dtype", "shape",
            "max_abs_error", "mean_abs_error",
            "max_rel_error", "mean_rel_error",
            "max_ulp_diff", "mean_ulp_diff",
            "matrix_rel_fro_error",
            "all_close", "jax_has_nan", "torch_has_nan",
            "torch_missing", "error_msg", "notes",
        ]
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(dataclasses.asdict(r))
        return filepath

    def generate_markdown(self, results: List[PrecisionResult],
                          filename: str = "precision_report.md"):
        filepath = os.path.join(self.report_dir, filename)

        with open(filepath, "w") as f:
            f.write(f"# {self.title} Report\n\n")

            # Group by category
            categories = {}
            for r in results:
                categories.setdefault(r.category, []).append(r)

            for cat_name, cat_results in categories.items():
                f.write(f"## {cat_name}\n\n")
                f.write("| Op | Dtype | Shape | Max Abs Err | Mean Abs Err | Max ULP | Matrix Rel Fro | Status |\n")
                f.write("|:---|:------|:------|------------:|-------------:|--------:|---------------:|:-------|\n")

                for r in cat_results:
                    if r.torch_missing:
                        status = "MISSING"
                    elif r.error_msg and r.error_msg.startswith("SKIP"):
                        status = "SKIP"
                    elif r.error_msg:
                        status = "ERROR"
                    elif r.all_close:
                        status = "PASS"
                    else:
                        status = "**DIFF**"

                    f.write(
                        f"| {r.op_name} | {r.dtype} | {r.shape} "
                        f"| {r.max_abs_error:.2e} | {r.mean_abs_error:.2e} "
                        f"| {r.max_ulp_diff:.1f} | {r.matrix_rel_fro_error:.2e} | {status} |\n"
                    )
                f.write("\n")

            # Summary
            total = len(results)
            passed = sum(1 for r in results
                         if r.all_close and not r.torch_missing and not r.error_msg)
            missing = sum(1 for r in results if r.torch_missing)
            errors = sum(1 for r in results
                         if r.error_msg and not r.error_msg.startswith("SKIP"))
            skipped = sum(1 for r in results
                          if r.error_msg and r.error_msg.startswith("SKIP"))
            diffs = total - passed - missing - errors - skipped

            f.write("## Summary\n\n")
            f.write(f"- **Total tests**: {total}\n")
            f.write(f"- **Passed (all_close)**: {passed}\n")
            f.write(f"- **Significant differences**: {diffs}\n")
            f.write(f"- **Missing torch equivalent**: {missing}\n")
            f.write(f"- **Errors**: {errors}\n")
            f.write(f"- **Skipped (dtype unsupported)**: {skipped}\n")

        return filepath

    def print_console_summary(self, results: List[PrecisionResult]):
        print("\n" + "=" * 90)
        print(f"{self.title.upper()} SUMMARY")
        print("=" * 90)

        total = len(results)
        passed = sum(1 for r in results
                     if r.all_close and not r.torch_missing and not r.error_msg)
        missing = sum(1 for r in results if r.torch_missing)
        errors = sum(1 for r in results
                     if r.error_msg and not r.error_msg.startswith("SKIP"))
        skipped = sum(1 for r in results
                      if r.error_msg and r.error_msg.startswith("SKIP"))
        diffs = total - passed - missing - errors - skipped

        print(f"\nTotal: {total} | Pass: {passed} | Diff: {diffs} | "
              f"Missing: {missing} | Error: {errors} | Skip: {skipped}")

        # Worst offenders
        real_results = [r for r in results
                        if not r.torch_missing and not r.error_msg]
        if real_results:
            worst_abs = max(real_results, key=lambda r: r.max_abs_error)
            worst_ulp = max(real_results, key=lambda r: r.max_ulp_diff)
            worst_fro = max(real_results, key=lambda r: r.matrix_rel_fro_error)

            print(f"\nWorst max abs error: {worst_abs.op_name} "
                  f"({worst_abs.dtype}, {worst_abs.shape}): {worst_abs.max_abs_error:.6e}")
            print(f"Worst max ULP diff:  {worst_ulp.op_name} "
                  f"({worst_ulp.dtype}, {worst_ulp.shape}): {worst_ulp.max_ulp_diff:.1f}")
            print(f"Worst matrix rel fro: {worst_fro.op_name} "
                  f"({worst_fro.dtype}, {worst_fro.shape}): {worst_fro.matrix_rel_fro_error:.6e}")

        # Missing ops
        missing_ops = sorted(set(r.op_name for r in results if r.torch_missing))
        if missing_ops:
            print(f"\nMISSING torch equivalents: {', '.join(missing_ops)}")

        # Errors
        error_results = [r for r in results
                         if r.error_msg and not r.error_msg.startswith("SKIP")]
        if error_results:
            print(f"\nERRORS ({len(error_results)}):")
            for r in error_results[:20]:
                print(f"  {r.op_name} ({r.dtype}, {r.shape}): {r.error_msg[:100]}")

        # Significant diffs
        diff_results = [r for r in results
                        if not r.all_close and not r.torch_missing and not r.error_msg]
        if diff_results:
            print(f"\nSIGNIFICANT DIFFERENCES ({len(diff_results)}):")
            for r in sorted(diff_results, key=lambda r: -r.max_abs_error)[:20]:
                print(f"  {r.op_name} ({r.dtype}, {r.shape}): "
                      f"max_abs={r.max_abs_error:.2e}, max_ulp={r.max_ulp_diff:.1f}")

        print()
