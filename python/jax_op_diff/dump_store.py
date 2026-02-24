"""HDF5 dump storage: write test cases and read them back as CaseData."""

import ast
import os
from typing import Iterator, Optional

import h5py
import numpy as np

from .config import safe_shape_str
from .core import CaseData, RunFilters
from .op_registry import OpSpec, generate_inputs


class DumpStore:
    """Dumps JAX test cases into one HDF5 file and reads them back."""

    def __init__(self, dump_dir: str):
        self.dump_dir = dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        self.dump_path = os.path.join(dump_dir, "dump.h5")
        self.case_index = 0

    def create(self) -> None:
        """Create a fresh dump file (overwrites existing)."""
        with h5py.File(self.dump_path, "w") as h5:
            h5.attrs["format"] = "jax-op-diff-h5-v1"

    def _make_case_name(self, op_name: str, dtype_key: str, shape_str: str) -> str:
        safe = shape_str.replace(" ", "").replace(",", "x").replace("(", "").replace(")", "")
        return f"{self.case_index:06d}__{op_name}__{dtype_key}__{safe}"

    @staticmethod
    def _encode_array(arr: np.ndarray):
        a = np.asarray(arr)
        dtype_module = getattr(a.dtype.type, "__module__", "")
        if dtype_module.startswith("ml_dtypes"):
            encoded = np.frombuffer(a.tobytes(), dtype=np.uint8)
            return encoded, str(a.dtype), a.shape
        return a, "", ()

    @staticmethod
    def _decode_array(dset):
        arr = np.array(dset)
        original_dtype = dset.attrs.get("original_dtype", "")
        if original_dtype:
            original_shape = dset.attrs.get("original_shape", None)
            if original_shape is not None:
                flat = np.frombuffer(
                    np.asarray(arr, dtype=np.uint8).tobytes(),
                    dtype=np.dtype(original_dtype),
                )
                arr = flat.reshape(tuple(original_shape))
            else:
                arr = arr.view(np.dtype(original_dtype))
        return arr

    def append_case(self, op: OpSpec, shape, dtype_key: str,
                    seed: int, jax_output: np.ndarray) -> str:
        """Append a single test case into dump.h5."""
        shape_str = safe_shape_str(shape)
        case_name = self._make_case_name(op.name, dtype_key, shape_str)

        inputs = generate_inputs(op, shape, dtype_key, seed)

        with h5py.File(self.dump_path, "a") as h5:
            grp = h5.create_group(case_name)
            grp.attrs["op_name"] = op.name
            grp.attrs["category"] = op.category
            grp.attrs["dtype"] = dtype_key
            grp.attrs["shape"] = str(shape)
            grp.attrs["notes"] = op.notes
            grp.attrs["arity"] = op.arity.value

            for key, arr in inputs.items():
                if isinstance(arr, (str, bytes)):
                    continue
                if not (isinstance(arr, np.ndarray) or np.isscalar(arr)):
                    continue
                encoded, original_dtype, original_shape = self._encode_array(arr)
                kwargs = {"data": encoded}
                if np.asarray(encoded).ndim > 0:
                    kwargs["compression"] = "gzip"
                dset = grp.create_dataset(f"input_{key}", **kwargs)
                if original_dtype:
                    dset.attrs["original_dtype"] = original_dtype
                    dset.attrs["original_shape"] = original_shape

            encoded, original_dtype, original_shape = self._encode_array(jax_output)
            kwargs = {"data": encoded}
            if np.asarray(encoded).ndim > 0:
                kwargs["compression"] = "gzip"
            dset = grp.create_dataset("jax_output", **kwargs)
            if original_dtype:
                dset.attrs["original_dtype"] = original_dtype
                dset.attrs["original_shape"] = original_shape

        self.case_index += 1
        return self.dump_path

    def iter_cases(self, filters: Optional[RunFilters] = None) -> Iterator[CaseData]:
        """Lazily iterate HDF5 cases, filtering by attrs before decoding data."""
        if not os.path.exists(self.dump_path):
            return

        with h5py.File(self.dump_path, "r") as h5:
            for name in sorted(h5.keys()):
                grp = h5[name]
                op_name = str(grp.attrs["op_name"])
                category = str(grp.attrs.get("category", ""))
                dtype_key = str(grp.attrs["dtype"])
                shape_str = str(grp.attrs["shape"])
                arity = str(grp.attrs.get("arity", ""))

                # Filter early before decoding data
                if filters and filters.categories and category not in filters.categories:
                    continue
                if filters and filters.op_names and op_name not in filters.op_names:
                    continue

                yield self._decode_case(name, grp, op_name, category,
                                        dtype_key, shape_str, arity)

    def _decode_case(self, name: str, grp, op_name: str, category: str,
                     dtype_key: str, shape_str: str, arity: str) -> CaseData:
        """Decode an h5py group into a CaseData NamedTuple."""
        inputs = {
            key[len("input_"):]: self._decode_array(grp[key])
            for key in grp.keys()
            if key.startswith("input_")
        }

        stored_output = (
            self._decode_array(grp["jax_output"])
            if "jax_output" in grp
            else np.array([])
        )

        return CaseData(
            case_name=name,
            op_name=op_name,
            category=category,
            dtype_key=dtype_key,
            shape_str=shape_str,
            arity=arity,
            inputs=inputs,
            stored_output=stored_output,
        )

    def case_count(self) -> int:
        """Return the number of cases in the dump file."""
        if not os.path.exists(self.dump_path):
            return 0
        with h5py.File(self.dump_path, "r") as h5:
            return len(h5.keys())
