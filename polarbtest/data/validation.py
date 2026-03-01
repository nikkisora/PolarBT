"""Data validation functions for OHLCV DataFrames."""

from dataclasses import dataclass, field

import polars as pl

OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}
PRICE_COLUMNS = {"open", "high", "low", "close"}
NUMERIC_DTYPES = {
    pl.Float32,
    pl.Float64,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}
TIMESTAMP_DTYPES = {pl.Date, pl.Datetime, pl.Time}


@dataclass
class ValidationResult:
    """Result of a data validation check.

    Attributes:
        valid: Whether all checks passed.
        errors: List of critical issues that will cause problems.
        warnings: List of non-critical issues worth noting.
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_columns(
    df: pl.DataFrame,
    required: set[str] | None = None,
    ohlcv: bool = False,
) -> ValidationResult:
    """Check that required columns exist in the DataFrame.

    Args:
        df: The DataFrame to validate.
        required: Set of column names that must be present.
        ohlcv: If True, require standard OHLCV columns (open, high, low, close, volume).

    Returns:
        ValidationResult with any missing column errors.
    """
    result = ValidationResult()
    cols = set(df.columns)

    if ohlcv:
        required = (required or set()) | OHLCV_COLUMNS

    if required:
        missing = required - cols
        if missing:
            result.valid = False
            result.errors.append(f"Missing required columns: {sorted(missing)}")

    return result


def validate_dtypes(df: pl.DataFrame) -> ValidationResult:
    """Check that known columns have appropriate data types.

    Validates that price/volume columns are numeric and timestamp columns
    are temporal types.

    Args:
        df: The DataFrame to validate.

    Returns:
        ValidationResult with dtype mismatch errors.
    """
    result = ValidationResult()
    for col in df.columns:
        dtype = df[col].dtype
        if col in PRICE_COLUMNS | {"volume"} and dtype not in NUMERIC_DTYPES:
            result.valid = False
            result.errors.append(f"Column '{col}' has non-numeric dtype {dtype}")
        if col == "timestamp" and dtype not in TIMESTAMP_DTYPES and dtype != pl.Utf8 and dtype != pl.String:
            result.warnings.append(f"Column 'timestamp' has dtype {dtype}, expected temporal type")
    return result


def validate_timestamps(df: pl.DataFrame, column: str = "timestamp") -> ValidationResult:
    """Check that timestamps are sorted and contain no duplicates.

    Args:
        df: The DataFrame to validate.
        column: Name of the timestamp column.

    Returns:
        ValidationResult with sorting/duplicate errors.
    """
    result = ValidationResult()
    if column not in df.columns:
        result.warnings.append(f"Timestamp column '{column}' not found, skipping timestamp validation")
        return result

    ts = df[column]

    # Check sorted
    if not ts.is_sorted():
        result.valid = False
        result.errors.append(f"Column '{column}' is not sorted in ascending order")

    # Check duplicates
    n_dupes = ts.len() - ts.n_unique()
    if n_dupes > 0:
        result.valid = False
        result.errors.append(f"Column '{column}' contains {n_dupes} duplicate values")

    return result


def validate_ohlc_integrity(df: pl.DataFrame) -> ValidationResult:
    """Check OHLC data integrity (high >= low, high >= open/close, low <= open/close).

    Args:
        df: The DataFrame to validate (must have open, high, low, close columns).

    Returns:
        ValidationResult with integrity errors.
    """
    result = ValidationResult()
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        result.warnings.append("OHLC columns not all present, skipping integrity check")
        return result

    violations = df.filter(pl.col("high") < pl.col("low"))
    if len(violations) > 0:
        result.valid = False
        result.errors.append(f"Found {len(violations)} rows where high < low")

    violations = df.filter((pl.col("high") < pl.col("open")) | (pl.col("high") < pl.col("close")))
    if len(violations) > 0:
        result.valid = False
        result.errors.append(f"Found {len(violations)} rows where high < open or high < close")

    violations = df.filter((pl.col("low") > pl.col("open")) | (pl.col("low") > pl.col("close")))
    if len(violations) > 0:
        result.valid = False
        result.errors.append(f"Found {len(violations)} rows where low > open or low > close")

    return result


def validate_no_nulls(df: pl.DataFrame, columns: list[str] | None = None) -> ValidationResult:
    """Check for null values in specified columns.

    Args:
        df: The DataFrame to validate.
        columns: Columns to check. If None, checks all columns.

    Returns:
        ValidationResult with null value errors.
    """
    result = ValidationResult()
    check_cols = columns or df.columns
    for col in check_cols:
        if col not in df.columns:
            continue
        null_count = df[col].null_count()
        if null_count > 0:
            result.valid = False
            result.errors.append(f"Column '{col}' contains {null_count} null values")
    return result


def validate_no_negative_prices(df: pl.DataFrame) -> ValidationResult:
    """Check that price columns contain no negative values.

    Args:
        df: The DataFrame to validate.

    Returns:
        ValidationResult with negative price errors.
    """
    result = ValidationResult()
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            continue
        if df[col].dtype not in NUMERIC_DTYPES:
            continue
        neg_count = df.filter(pl.col(col) < 0).height
        if neg_count > 0:
            result.valid = False
            result.errors.append(f"Column '{col}' contains {neg_count} negative values")
    return result


def validate(
    df: pl.DataFrame,
    required_columns: set[str] | None = None,
    ohlcv: bool = False,
    timestamp_column: str = "timestamp",
) -> ValidationResult:
    """Run all validation checks on a DataFrame.

    Combines column presence, dtype, timestamp ordering, OHLC integrity,
    null checks on price columns, and negative price checks.

    Args:
        df: The DataFrame to validate.
        required_columns: Additional required columns beyond OHLCV.
        ohlcv: If True, require standard OHLCV columns.
        timestamp_column: Name of the timestamp column.

    Returns:
        Combined ValidationResult from all checks.
    """
    combined = ValidationResult()

    checks = [
        validate_columns(df, required=required_columns, ohlcv=ohlcv),
        validate_dtypes(df),
        validate_timestamps(df, column=timestamp_column),
        validate_no_nulls(df, columns=[c for c in PRICE_COLUMNS if c in df.columns]),
        validate_no_negative_prices(df),
    ]

    if {"open", "high", "low", "close"}.issubset(set(df.columns)):
        checks.append(validate_ohlc_integrity(df))

    for check in checks:
        if not check.valid:
            combined.valid = False
        combined.errors.extend(check.errors)
        combined.warnings.extend(check.warnings)

    return combined
