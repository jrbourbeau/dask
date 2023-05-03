import numpy as np
import pandas as pd

from dask.dataframe._compat import PANDAS_GT_150, PANDAS_GT_200
from dask.dataframe.utils import is_dataframe_like, is_index_like, is_series_like

try:
    import pyarrow as pa
except ImportError:
    pa = None


def is_pyarrow_dtype(dtype):
    return isinstance(dtype, pd.ArrowDtype) or dtype == pd.StringDtype("pyarrow")


def to_pyarrow_dtype(dtype):
    import pyarrow as pa
    from pandas.core.arrays.arrow.array import to_pyarrow_type
    from pandas.core.dtypes.dtypes import BaseMaskedDtype, PandasExtensionDtype

    # Already a pyarrow-backed dtype
    if is_pyarrow_dtype(dtype):
        return dtype

    if isinstance(dtype, PandasExtensionDtype):
        base_dtype = dtype.base
    elif isinstance(dtype, BaseMaskedDtype):
        base_dtype = dtype.numpy_dtype
    elif isinstance(dtype, pd.StringDtype):
        base_dtype = np.dtype(str)
    else:
        base_dtype = dtype

    if base_dtype == object:
        # Convert objects to strings
        pa_type = pa.string()
    else:
        pa_type = to_pyarrow_type(base_dtype)
    if pa_type is None:
        raise TypeError(f"Encountered {dtype} which is not compatible with pyarrow")

    if pa_type == pa.string():
        return pd.StringDtype("pyarrow")
    else:
        return pd.ArrowDtype(pa_type)


def is_pyarrow_string_dtype(dtype):
    """Is the input dtype a pyarrow string?"""
    if pa is None:
        return False

    if PANDAS_GT_150:
        pa_string_types = [pd.StringDtype("pyarrow"), pd.ArrowDtype(pa.string())]
    else:
        pa_string_types = [pd.StringDtype("pyarrow")]
    return dtype in pa_string_types


def is_object_string_dtype(dtype):
    """Determine if input is a non-pyarrow string dtype"""
    # in pandas < 2.0, is_string_dtype(DecimalDtype()) returns True
    return not is_pyarrow_dtype(dtype)
    # return (
    #     pd.api.types.is_string_dtype(dtype)
    #     and not is_pyarrow_string_dtype(dtype)
    #     and not pd.api.types.is_dtype_equal(dtype, "decimal")
    # )


def is_object_string_index(x):
    if isinstance(x, pd.MultiIndex):
        return any(is_object_string_index(level) for level in x.levels)
    return isinstance(x, pd.Index) and is_object_string_dtype(x.dtype)


def is_object_string_series(x):
    return isinstance(x, pd.Series) and (
        is_object_string_dtype(x.dtype) or is_object_string_index(x.index)
    )


def is_object_string_dataframe(x):
    return isinstance(x, pd.DataFrame) and (
        any(is_object_string_series(s) for _, s in x.items())
        or is_object_string_index(x.index)
    )


def to_pyarrow_string(df):
    if not (is_dataframe_like(df) or is_series_like(df) or is_index_like(df)):
        return df

    # Possibly convert DataFrame/Series/Index to `string[pyarrow]`
    dtypes = None
    if is_dataframe_like(df):
        dtypes = {
            col: to_pyarrow_dtype(dtype)
            for col, dtype in df.dtypes.items()
            if is_object_string_dtype(dtype)
        }
    elif is_object_string_dtype(df.dtype):
        dtypes = to_pyarrow_dtype(df.dtype)

    if dtypes:
        df = df.astype(dtypes, copy=False)

    # Convert DataFrame/Series index too
    if (is_dataframe_like(df) or is_series_like(df)) and is_object_string_index(
        df.index
    ):
        if isinstance(df.index, pd.MultiIndex):
            levels = {
                i: level.astype(to_pyarrow_dtype(level.dtype))
                for i, level in enumerate(df.index.levels)
                if is_object_string_dtype(level.dtype)
            }
            # set verify_integrity=False to preserve index codes
            df.index = df.index.set_levels(
                levels.values(), level=levels.keys(), verify_integrity=False
            )
        else:
            df.index = df.index.astype(to_pyarrow_dtype(df.index.dtype))
    return df


def check_pyarrow_string_supported():
    """Make sure we have all the required versions"""
    if pa is None:
        raise RuntimeError(
            "Using dask's `dataframe.convert-string` configuration "
            "option requires `pyarrow` to be installed."
        )
    if not PANDAS_GT_200:
        raise RuntimeError(
            "Using dask's `dataframe.convert-string` configuration "
            "option requires `pandas>=2.0` to be installed."
        )
