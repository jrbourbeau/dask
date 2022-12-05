import decimal
import signal
import sys
import threading

import pytest

from dask.datasets import timeseries

dd = pytest.importorskip("dask.dataframe")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pyspark = pytest.importorskip("pyspark")
pa = pytest.importorskip("pyarrow")
pytest.importorskip("fastparquet")

from dask.dataframe.utils import assert_eq

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="Unnecessary, and hard to get spark working on non-linux platforms",
)

# pyspark auto-converts timezones -- round-tripping timestamps is easier if
# we set everything to UTC.
pdf = timeseries(freq="1H").compute()
pdf.index = pdf.index.tz_localize("UTC")
pdf = pdf.reset_index()


@pytest.fixture(scope="module")
def spark_session():
    # Spark registers a global signal handler that can cause problems elsewhere
    # in the test suite. In particular, the handler fails if the spark session
    # is stopped (a bug in pyspark).
    prev = signal.getsignal(signal.SIGINT)
    # Create a spark session. Note that we set the timezone to UTC to avoid
    # conversion to local time when reading parquet files.
    spark = (
        pyspark.sql.SparkSession.builder.master("local")
        .appName("Dask Testing")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield spark

    spark.stop()
    # Make sure we get rid of the signal once we leave stop the session.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, prev)


@pytest.mark.parametrize("npartitions", (1, 5, 10))
@pytest.mark.parametrize("engine", ("pyarrow", "fastparquet"))
def test_roundtrip_parquet_spark_to_dask(spark_session, npartitions, tmpdir, engine):
    tmpdir = str(tmpdir)

    sdf = spark_session.createDataFrame(pdf)
    # We are not overwriting any data, but spark complains if the directory
    # already exists (as tmpdir does) and we don't set overwrite
    sdf.repartition(npartitions).write.parquet(tmpdir, mode="overwrite")

    ddf = dd.read_parquet(tmpdir, engine=engine)
    # Papercut: pandas TZ localization doesn't survive roundtrip
    ddf = ddf.assign(timestamp=ddf.timestamp.dt.tz_localize("UTC"))
    assert ddf.npartitions == npartitions

    assert_eq(ddf, pdf, check_index=False)


@pytest.mark.parametrize("engine", ("pyarrow", "fastparquet"))
def test_roundtrip_hive_parquet_spark_to_dask(spark_session, tmpdir, engine):
    tmpdir = str(tmpdir)

    sdf = spark_session.createDataFrame(pdf)
    # not overwriting any data, but spark complains if the directory
    # already exists and we don't set overwrite
    sdf.write.parquet(tmpdir, mode="overwrite", partitionBy="name")

    ddf = dd.read_parquet(tmpdir, engine=engine)
    # Papercut: pandas TZ localization doesn't survive roundtrip
    ddf = ddf.assign(timestamp=ddf.timestamp.dt.tz_localize("UTC"))

    # Partitioning can change the column order. This is mostly okay,
    # but we sort them here to ease comparison
    ddf = ddf.compute().sort_index(axis=1)
    # Dask automatically converts hive-partitioned columns to categories.
    # This is fine, but convert back to strings for comparison.
    ddf = ddf.assign(name=ddf.name.astype("str"))

    assert_eq(ddf, pdf.sort_index(axis=1), check_index=False)


@pytest.mark.parametrize("npartitions", (1, 5, 10))
@pytest.mark.parametrize("engine", ("pyarrow", "fastparquet"))
def test_roundtrip_parquet_dask_to_spark(spark_session, npartitions, tmpdir, engine):
    tmpdir = str(tmpdir)
    ddf = dd.from_pandas(pdf, npartitions=npartitions)

    # Papercut: https://github.com/dask/fastparquet/issues/646#issuecomment-885614324
    kwargs = {"times": "int96"} if engine == "fastparquet" else {}

    ddf.to_parquet(tmpdir, engine=engine, write_index=False, **kwargs)

    sdf = spark_session.read.parquet(tmpdir)
    sdf = sdf.toPandas()

    # Papercut: pandas TZ localization doesn't survive roundtrip
    sdf = sdf.assign(timestamp=sdf.timestamp.dt.tz_localize("UTC"))

    assert_eq(sdf, ddf, check_index=False)


def test_roundtrip_parquet_spark_to_dask_extension_dtypes(spark_session, tmpdir):
    tmpdir = str(tmpdir)
    npartitions = 5

    size = 20
    pdf = pd.DataFrame(
        {
            "a": range(size),
            "b": np.random.random(size=size),
            "c": [True, False] * (size // 2),
            "d": ["alice", "bob"] * (size // 2),
        }
    )
    # Note: since we set use_nullable_dtypes=True below, we are expecting *all*
    # of the resulting series to use those dtypes. If there is a mix of nullable
    # and non-nullable dtypes here, then that will result in dtype mismatches
    # in the finale frame.
    pdf = pdf.astype(
        {
            "a": "Int64",
            "b": "Float64",
            "c": "boolean",
            "d": "string",
        }
    )
    # # Ensure all columns are extension dtypes
    assert all([pd.api.types.is_extension_array_dtype(dtype) for dtype in pdf.dtypes])

    sdf = spark_session.createDataFrame(pdf)
    # We are not overwriting any data, but spark complains if the directory
    # already exists (as tmpdir does) and we don't set overwrite
    sdf.repartition(npartitions).write.parquet(tmpdir, mode="overwrite")

    ddf = dd.read_parquet(tmpdir, engine="pyarrow", use_nullable_dtypes=True)
    assert all(
        [pd.api.types.is_extension_array_dtype(dtype) for dtype in ddf.dtypes]
    ), ddf.dtypes
    assert_eq(ddf, pdf, check_index=False)


def test_read_decimal_dtype(spark_session, tmpdir):
    tmpdir = "test.parquet"
    npartitions = 3

    size = 6

    decimal_data = [
        decimal.Decimal("8093.234"),
        decimal.Decimal("8094.234"),
        decimal.Decimal("8095.234"),
        decimal.Decimal("8096.234"),
        decimal.Decimal("8097.234"),
        decimal.Decimal("8098.234"),
    ]
    pdf = pd.DataFrame(
        {
            "a": range(size),
            "b": np.random.random(size=size),
            "c": [True, False] * (size // 2),
            "d": ["alice", "bob"] * (size // 2),
            "e": decimal_data,
        }
    )
    pa_decimal_type = pa.decimal128(7, 3)
    pdf = pdf.astype(
        {
            "a": "int64[pyarrow]",
            "b": "float64[pyarrow]",
            "c": "boolean[pyarrow]",
            "d": "string[pyarrow]",
            "e": pd.ArrowDtype(pa_decimal_type),
        }
    )

    sdf = spark_session.createDataFrame(pdf)
    sdf.printSchema()
    # sdf = sdf.withColumn("b", sdf["b"].cast(pyspark.sql.types.DecimalType(10, 0)))
    # sdf.printSchema()
    # We are not overwriting any data, but spark complains if the directory
    # already exists (as tmpdir does) and we don't set overwrite
    sdf.repartition(npartitions).write.parquet(tmpdir, mode="overwrite")

    # types_mapper = {pa_decimal_type: pd.ArrowDtype(pa_decimal_type)}
    ddf = dd.read_parquet(tmpdir, engine="pyarrow", use_nullable_dtypes="pyarrow")
    assert ddf.e.dtype == "decimal128(7, 3)[pyarrow]", ddf.e.dtype
    assert ddf.e.compute().dtype == "decimal128(7, 3)[pyarrow]", ddf.e.compute().dtype
    assert_eq(ddf, sdf.toPandas(), check_index=False)
