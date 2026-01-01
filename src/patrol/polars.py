"""Polars backend for type-parameterized DataFrame with Protocol-based schema validation."""

from typing import Optional
from typing import Generic, TypeVar, get_type_hints

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install patrol[polars]")


SchemaT_co = TypeVar("SchemaT_co", covariant=True)


# Polars type checkers
TYPE_CHECKERS = {
    int: lambda dtype: dtype
    in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64),
    float: lambda dtype: dtype in (pl.Float32, pl.Float64),
    str: lambda dtype: dtype == pl.Utf8,
    bool: lambda dtype: dtype == pl.Boolean,
}


def validate_dataframe(df: pl.DataFrame, schema: type) -> None:
    """
    Validate that a Polars DataFrame conforms to a Protocol schema.

    Args:
        df: Polars DataFrame to validate
        schema: Protocol type defining the expected schema

    Raises:
        ValueError: If a required column is missing or type is unsupported
        TypeError: If a column has the wrong type
    """
    expected_cols = get_type_hints(schema)

    for col_name, col_type in expected_cols.items():
        _check_column_exists(df, col_name)
        _check_column_type(df, col_name, col_type)


def _check_column_exists(df: pl.DataFrame, col_name: str) -> None:
    """Check if a column exists in the DataFrame."""
    if col_name not in df.columns:
        raise ValueError(f"Missing column: {col_name}")


def _check_column_type(df: pl.DataFrame, col_name: str, expected_type: type) -> None:
    """Check if a column has the expected type."""
    if expected_type not in TYPE_CHECKERS:
        raise ValueError(f"Unsupported type: {expected_type}")

    type_checker = TYPE_CHECKERS[expected_type]
    col_dtype = df[col_name].dtype
    if not type_checker(col_dtype):
        raise TypeError(f"Column '{col_name}' expected {expected_type.__name__}, got {col_dtype}")


class DataFrame(pl.DataFrame, Generic[SchemaT_co]):
    """
    Type-parameterized DataFrame with runtime validation for Polars.

    Usage:
        # Static type checking only
        def process(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
            return df

        # Runtime validation
        validated = DataFrame[UserSchema](raw_df)

    The type parameter is covariant, allowing structural subtyping:
        DataFrame[ChildSchema] is compatible with DataFrame[ParentSchema]
        when ChildSchema has all columns of ParentSchema.
    """

    _schema: Optional[type] = None

    def __class_getitem__(cls, schema: type):
        """Create a new DataFrame class with schema validation."""

        class TypedDataFrame(DataFrame):
            _schema = schema

        return TypedDataFrame

    def __new__(cls, data, *args, **kwargs):
        """
        Create DataFrame with optional schema validation.

        Args:
            data: Data to create DataFrame from (pl.DataFrame or dict/list)
            *args: Additional arguments passed to pl.DataFrame
            **kwargs: Additional keyword arguments passed to pl.DataFrame

        Raises:
            ValueError: If required column is missing
            TypeError: If column has wrong type
        """
        # Create DataFrame instance
        if isinstance(data, pl.DataFrame):
            instance = data
        else:
            instance = pl.DataFrame(data, *args, **kwargs)

        # Validate if schema is set
        if cls._schema is not None:
            validate_dataframe(instance, cls._schema)

        return instance
