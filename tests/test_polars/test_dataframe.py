from typing import Protocol

import pytest

try:
    import polars as pl

    from patrol.polars import DataFrame

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


pytestmark = pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")


class SimpleSchema(Protocol):
    a: int


class MultiTypeSchema(Protocol):
    int_col: int
    float_col: float
    str_col: str
    bool_col: bool


def test_dataframe_class_getitem_returns_class():
    """DataFrame[Schema] returns a class"""
    type_of = DataFrame[SimpleSchema]
    assert isinstance(type_of, type)


def test_dataframe_with_schema_validates_correct_dataframe():
    """DataFrame[Schema](df) passes validation for correct DataFrame"""
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = DataFrame[SimpleSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_schema_raises_on_missing_column():
    """DataFrame[Schema](df) raises error for missing column"""
    df = pl.DataFrame({"b": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing column: a"):
        DataFrame[SimpleSchema](df)


def test_dataframe_with_schema_raises_on_wrong_type():
    """DataFrame[Schema](df) raises error for wrong type"""
    df = pl.DataFrame({"a": ["x", "y", "z"]})
    with pytest.raises(TypeError, match="Column 'a' expected int"):
        DataFrame[SimpleSchema](df)


def test_dataframe_multiple_types():
    """Support multiple types"""
    df = pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.5, 3.7],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )
    result = DataFrame[MultiTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_extra_columns():
    """Extra columns are ignored during validation"""
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = DataFrame[SimpleSchema](df)
    assert result.equals(df)
