from typing import Annotated, Protocol

import polars as pl
import pytest

from patrol.polars import DataFrame
from patrol.validators import Range


class AgeSchema(Protocol):
    age: Annotated[int, Range(0, 150)]


def test_range_validator_accepts_values_within_range():
    """Range validator accepts values within the specified range"""
    df = pl.DataFrame({"age": [25, 30, 45, 100]})
    result = DataFrame[AgeSchema](df)
    assert isinstance(result, pl.DataFrame)


def test_range_validator_rejects_values_below_minimum():
    """Range validator rejects values below the minimum"""
    df = pl.DataFrame({"age": [25, -5, 30]})
    with pytest.raises(ValueError, match="age.*range"):
        DataFrame[AgeSchema](df)


def test_range_validator_rejects_values_above_maximum():
    """Range validator rejects values above the maximum"""
    df = pl.DataFrame({"age": [25, 200, 30]})
    with pytest.raises(ValueError, match="age.*range"):
        DataFrame[AgeSchema](df)
