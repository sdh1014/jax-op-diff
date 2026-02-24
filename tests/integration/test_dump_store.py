"""DumpStore HDF5 read/write tests. Needs h5py, no GPU required."""

import numpy as np
import pytest

from jax_op_diff.dump_store import DumpStore
from jax_op_diff.core import RunFilters, CaseData
from jax_op_diff.op_registry import OpSpec, OpArity, InputDomain


def _make_op(name="test_add", category="basic", **kw) -> OpSpec:
    defaults = dict(
        name=name,
        category=category,
        arity=OpArity.UNARY,
        jax_fn=lambda x: x,
        shape_type="elementwise",
    )
    defaults.update(kw)
    return OpSpec(**defaults)


class TestDumpStore:
    def test_write_read_roundtrip(self, tmp_path):
        store = DumpStore(str(tmp_path))
        store.create()
        op = _make_op()
        output = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        store.append_case(op, (3,), "float32", 42, output)

        cases = list(store.iter_cases())
        assert len(cases) == 1
        case = cases[0]
        assert isinstance(case, CaseData)
        assert case.op_name == "test_add"
        assert case.dtype_key == "float32"
        np.testing.assert_array_equal(case.stored_output, output)

    def test_filter_by_category(self, tmp_path):
        store = DumpStore(str(tmp_path))
        store.create()
        store.append_case(_make_op(category="basic"), (3,), "float32", 42,
                          np.array([1.0], dtype=np.float32))
        store.append_case(_make_op(name="other_op", category="trig"), (3,), "float32", 42,
                          np.array([2.0], dtype=np.float32))

        basic_cases = list(store.iter_cases(RunFilters(categories={"basic"})))
        assert len(basic_cases) == 1
        assert basic_cases[0].category == "basic"

    def test_filter_by_op_name(self, tmp_path):
        store = DumpStore(str(tmp_path))
        store.create()
        store.append_case(_make_op(name="add"), (3,), "float32", 42,
                          np.array([1.0], dtype=np.float32))
        store.append_case(_make_op(name="mul"), (3,), "float32", 42,
                          np.array([2.0], dtype=np.float32))

        add_cases = list(store.iter_cases(RunFilters(op_names={"add"})))
        assert len(add_cases) == 1
        assert add_cases[0].op_name == "add"

    def test_empty_dump_returns_no_cases(self, tmp_path):
        store = DumpStore(str(tmp_path))
        store.create()
        cases = list(store.iter_cases())
        assert len(cases) == 0

    def test_nonexistent_dump_returns_no_cases(self, tmp_path):
        store = DumpStore(str(tmp_path / "nonexistent"))
        cases = list(store.iter_cases())
        assert len(cases) == 0

    def test_case_count(self, tmp_path):
        store = DumpStore(str(tmp_path))
        store.create()
        assert store.case_count() == 0
        store.append_case(_make_op(), (3,), "float32", 42,
                          np.array([1.0], dtype=np.float32))
        assert store.case_count() == 1
