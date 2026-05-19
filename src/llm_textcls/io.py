"""Parquet R/W with schema validation, path resolution, .env loading."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


@dataclass(frozen=True)
class FNNRow:
    id: str
    label: str
    title: str
    text: str
    url: str
    token_count: int
    held_out_fs: bool


@dataclass(frozen=True)
class ERUnlabeledRow:
    id: str
    text: str
    keyword_class: str
    keyword_match_count: int
    token_count: int
    held_out_fs: bool


@dataclass(frozen=True)
class ERLabeledRow:
    id: str
    text: str
    label: str
    token_count: int
    held_out_fs: bool


_DTYPE_MAP: dict[type, set[str]] = {
    str: {"object", "string", "str"},
    int: {"int64", "int32", "Int64", "Int32"},
    bool: {"bool", "boolean"},
    float: {"float64", "float32"},
}


def project_root() -> Path:
    """Return the repo root (parent of `src/`).

    Returns:
        Absolute path to project root.
    """
    return Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load `.env` from project root if present; no error if missing."""
    env_path = project_root() / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def assert_schema(df: pd.DataFrame, schema: type) -> None:
    """Validate that df columns + dtypes match the dataclass annotations.

    Args:
        df: DataFrame to validate.
        schema: dataclass type whose annotations define the expected schema.

    Raises:
        ValueError: on column-set mismatch or dtype mismatch.
    """
    expected = {f.name: f.type for f in fields(schema)}
    actual_cols = set(df.columns)
    expected_cols = set(expected.keys())
    if actual_cols != expected_cols:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        raise ValueError(
            f"Schema mismatch for {schema.__name__}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for col, py_type in expected.items():
        # Dataclass annotations may be strings under `from __future__ import annotations`.
        if isinstance(py_type, str):
            py_type = {"str": str, "int": int, "bool": bool, "float": float}.get(py_type, str)
        allowed = _DTYPE_MAP.get(py_type, set())
        actual_dtype = str(df[col].dtype)
        if actual_dtype not in allowed:
            raise ValueError(
                f"Schema mismatch for {schema.__name__}.{col}: "
                f"expected one of {sorted(allowed)}, got {actual_dtype}"
            )


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame.

    Args:
        path: Path to .parquet file.

    Returns:
        DataFrame.
    """
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path, schema: type) -> None:
    """Validate against schema, then write Parquet (creating parent dirs).

    Args:
        df: DataFrame to write.
        path: Output .parquet path.
        schema: dataclass type to validate against.

    Raises:
        ValueError: if schema validation fails.
    """
    assert_schema(df, schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
