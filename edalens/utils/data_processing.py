"""Single implementation of the Home and Missing Data transformations.

Views call ``run_operation`` for both direct and tracked execution, and replay
uses ``apply_operations``; both go through ``apply_operation`` so the two paths
cannot drift. Every operation returns a new DataFrame and never mutates its
input. Failures raise ``OperationError`` and leave session state untouched.
"""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import os
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from sklearn.impute import KNNImputer

Operation = Mapping[str, Any]

DATE_FEATURES = ("Year", "Month", "Day", "DayOfWeek", "Hour")
SAMPLE_SEED = 42
KNN_NEIGHBORS = 5
# Set by ``edalens.walk(df)``: path of a pickled DataFrame to preload in every session.
PRESET_DATA_ENV = "EDALENS_DATA"
PRESET_DATA_LABEL = "DataFrame from notebook"


class OperationError(ValueError):
    """A transformation could not be applied; the message is shown to users."""


def describe_operation(operation: Operation) -> str:
    """Return a short human-readable label for an operation-history entry."""
    op_type = operation.get("type")
    if op_type == "select_columns":
        return f"Selected columns: {', '.join(map(str, operation['columns']))}"
    if op_type == "cast_types":
        numeric = ", ".join(map(str, operation["numeric_cols"]))
        categorical = ", ".join(map(str, operation["categorical_cols"]))
        return f"Cast types - Numeric: {numeric}, Categorical: {categorical}"
    if op_type == "subsample":
        return f"Subsampled to {operation['sample_pct']}% of data"
    if op_type == "melt":
        return f"Melted with ID vars: {', '.join(map(str, operation['id_vars']))}"
    if op_type == "date_features":
        return (
            f"Extracted date features from {operation['date_column']} "
            f"(dayfirst: {operation['dayfirst']})"
        )
    if op_type == "remove_nan":
        return "Removed rows with missing values"
    if op_type == "impute":
        return "Imputed missing values (KNN numeric, mode categorical)"
    return f"Unknown operation {op_type!r}"


def _require_columns(df: pd.DataFrame, columns: Sequence[Any], role: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise OperationError(
            f"{role} refer to columns missing from the current data: "
            f"{', '.join(map(str, missing))}."
        )


def _select_columns(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    columns = list(operation["columns"])
    if not columns:
        raise OperationError("Select at least one column to keep.")
    _require_columns(df, columns, "Selected columns")
    wanted = set(columns)
    # Keep the frame's own column order so selection never reorders data.
    return df.loc[:, [column for column in df.columns if column in wanted]].copy()


def _as_text(series: pd.Series) -> pd.Series:
    """Convert values to text while keeping missing values missing."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.copy()
    return series.astype(object).map(str).where(series.notna(), np.nan).astype(object)


def _as_numeric(series: pd.Series) -> pd.Series:
    if ptypes.is_datetime64_any_dtype(series) or ptypes.is_timedelta64_dtype(series):
        raise OperationError(
            f"Column '{series.name}' is datetime-like and cannot be cast to numeric; "
            "use date feature extraction instead."
        )
    if ptypes.is_numeric_dtype(series):
        return series.copy()
    converted = pd.to_numeric(series.astype(object), errors="coerce")
    if series.notna().any() and converted.isna().all():
        raise OperationError(
            f"Column '{series.name}' has no values that can be read as numbers."
        )
    return converted


def _cast_types(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    numeric = list(operation["numeric_cols"])
    categorical = list(operation["categorical_cols"])
    overlap = set(numeric) & set(categorical)
    if overlap:
        raise OperationError(
            f"Columns cannot be both numeric and categorical: {', '.join(map(str, overlap))}."
        )
    _require_columns(df, numeric + categorical, "Variable types")
    result = df.copy()
    for column in numeric:
        result[column] = _as_numeric(result[column])
    for column in categorical:
        result[column] = _as_text(result[column])
    return result


def _subsample(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    pct = operation["sample_pct"]
    if not 0 < pct <= 100:
        raise OperationError("Sample percentage must be greater than 0 and at most 100.")
    if pct == 100:
        return df.copy()
    # Same rows as ``df.sample(frac, random_state=42)``, kept in original order.
    positions = (
        pd.RangeIndex(len(df)).to_series().sample(frac=pct / 100, random_state=SAMPLE_SEED)
    )
    return df.iloc[np.sort(positions.to_numpy())].copy()


def _melt(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    id_vars = list(operation["id_vars"])
    if not id_vars:
        raise OperationError("Select at least one ID variable before melting.")
    _require_columns(df, id_vars, "ID variables")
    value_vars = [column for column in df.columns if column not in id_vars]
    if not value_vars:
        raise OperationError("Melting needs at least one non-ID column to turn into rows.")
    # Melting reshapes rows, so the result has a new RangeIndex by design.
    return pd.melt(
        df, id_vars=id_vars, value_vars=value_vars, var_name="variable", value_name="value"
    )


def _date_features(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    column = operation["date_column"]
    _require_columns(df, [column], "Date column")
    collisions = [name for name in DATE_FEATURES if name in df.columns]
    if collisions:
        raise OperationError(
            "Date features would overwrite existing columns: "
            f"{', '.join(collisions)}. Remove or rename them first."
        )
    source = df[column]
    parsed = pd.to_datetime(source, errors="coerce", dayfirst=operation["dayfirst"])
    if len(df) and parsed.isna().all():
        raise OperationError(f"No values in column '{column}' could be parsed as dates.")
    result = df.copy()
    has_missing = bool(parsed.isna().any())
    parts = {
        "Year": parsed.dt.year,
        "Month": parsed.dt.month,
        "Day": parsed.dt.day,
        "DayOfWeek": parsed.dt.day_of_week,
        "Hour": parsed.dt.hour,
    }
    for name in DATE_FEATURES:
        result[name] = parts[name].astype("float64" if has_missing else "int64")
    result[column] = parsed.astype(str).where(parsed.notna(), np.nan).astype(object)
    return result


def _remove_nan(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    result = df.dropna()
    if len(df) and result.empty:
        raise OperationError(
            "Removing rows with missing values would remove every row; "
            "drop mostly-empty columns or impute instead."
        )
    return result.copy()


def _imputed_dtype(dtype: Any) -> Any:
    """Integer columns receiving KNN means become float to keep fractions."""
    if ptypes.is_float_dtype(dtype):
        return dtype
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        return "Float64"
    return "float64"


def _is_knn_numeric(series: pd.Series) -> bool:
    return ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)


def _is_mode_imputed(series: pd.Series) -> bool:
    return not _is_knn_numeric(series) and not (
        ptypes.is_datetime64_any_dtype(series) or ptypes.is_timedelta64_dtype(series)
    )


def _impute(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    """KNN-impute numeric columns and mode-impute text/categorical/boolean ones.

    Datetime-like and all-missing columns are left unchanged. Columns without
    missing values keep their exact dtype; column order and index never change.
    """
    result = df.copy()
    features = [
        column
        for column in result.columns
        if _is_knn_numeric(result[column]) and result[column].notna().any()
    ]
    targets = [column for column in features if result[column].isna().any()]
    if targets:
        matrix = result[features].to_numpy(dtype="float64", na_value=np.nan)
        imputer = KNNImputer(
            n_neighbors=KNN_NEIGHBORS, weights="uniform", metric="nan_euclidean"
        )
        imputed = imputer.fit_transform(matrix)
        if imputed.shape != matrix.shape:
            raise OperationError("KNN imputation changed the data shape unexpectedly.")
        for position, column in enumerate(features):
            if column in targets:
                result[column] = pd.Series(
                    imputed[:, position], index=result.index, name=column
                ).astype(_imputed_dtype(result[column].dtype))
    for column in result.columns:
        series = result[column]
        if _is_mode_imputed(series) and series.isna().any():
            mode = series.mode(dropna=True)
            if not mode.empty:
                result[column] = series.fillna(mode.iloc[0])
    return result


_HANDLERS: dict[str, Callable[[pd.DataFrame, Operation], pd.DataFrame]] = {
    "select_columns": _select_columns,
    "cast_types": _cast_types,
    "subsample": _subsample,
    "melt": _melt,
    "date_features": _date_features,
    "remove_nan": _remove_nan,
    "impute": _impute,
}


def apply_operation(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    """Apply one operation to ``df`` and return a new frame; never mutates ``df``."""
    op_type = operation.get("type") if isinstance(operation, Mapping) else None
    handler = _HANDLERS.get(op_type)  # type: ignore[arg-type]
    if handler is None:
        raise OperationError(f"Unknown operation type: {op_type!r}.")
    if not df.columns.is_unique:
        raise OperationError("Column names must be unique before applying operations.")
    try:
        return handler(df, operation)
    except OperationError:
        raise
    except Exception as exc:  # Normalize pandas/sklearn errors for the UI.
        raise OperationError(f"Could not apply '{op_type}': {exc}") from exc


def apply_operations(raw_df: pd.DataFrame, operations: Sequence[Operation]) -> pd.DataFrame:
    """Replay ``operations`` on a copy of ``raw_df``; raise on the first failure."""
    df = raw_df.copy()
    for step, operation in enumerate(operations, 1):
        try:
            df = apply_operation(df, operation)
        except OperationError as exc:
            raise OperationError(
                f"Step {step} ({describe_operation(operation)}) failed: {exc}"
            ) from exc
    return df


def column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return the numeric and categorical column lists the views use."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return numeric, categorical


def set_working_df(
    state: MutableMapping[str, Any], df: pd.DataFrame, operations: list[dict]
) -> None:
    """Store a validated working frame together with its operation history.

    ``data_version`` changes with every new frame; views put it in widget keys
    so widgets restart from the new columns instead of keeping selections of
    columns that no longer exist (which Streamlit would show as empty).
    """
    numeric, categorical = column_types(df)
    state["data_version"] = state.get("data_version", 0) + 1
    state["filtered_df"] = df
    state["operations"] = operations
    state["current_numeric_cols"] = numeric
    state["current_categorical_cols"] = categorical
    state["is_imputed"] = any(op.get("type") == "impute" for op in operations)


def load_preset_data(
    state: MutableMapping[str, Any], environ: Mapping[str, str] = os.environ
) -> bool:
    """Load the notebook's DataFrame into a new session; return whether it loaded.

    Runs once per browser session, so "Reset All" or a new upload replaces it.
    """
    path = environ.get(PRESET_DATA_ENV)
    if not path or state.get("preset_data_loaded"):
        return False
    if path.endswith(".json"):  # In-browser mode: pickles are not portable to Pyodide's pandas.
        raw_df = pd.read_json(path, orient="table")
    else:
        raw_df = pd.read_pickle(path)
    for key in ("filtered_df", "current_numeric_cols", "current_categorical_cols", "is_imputed", "imputed_df"):
        state.pop(key, None)
    state["raw_df"] = raw_df.copy()
    set_working_df(state, raw_df.copy(), [])
    state["last_uploaded_file"] = PRESET_DATA_LABEL
    state["data_loaded"] = True
    state["preset_data_loaded"] = True
    return True


def run_operation(state: MutableMapping[str, Any], operation: Operation) -> pd.DataFrame:
    """Apply ``operation`` to the session's working frame atomically.

    Direct mode applies it to ``filtered_df``; tracking mode replays the full
    history from ``raw_df``. Both use ``apply_operation``, so results agree.
    History is always recorded so enabling tracking later replays correctly.
    On ``OperationError`` no session key is changed.
    """
    operations = [dict(op) for op in state.get("operations", [])]
    candidate = [*operations, dict(operation)]
    if state.get("enable_operation_tracking", False):
        result = apply_operations(state["raw_df"], candidate)
    else:
        result = apply_operation(state["filtered_df"], operation)
    set_working_df(state, result, candidate)
    return result


def undo_last_operation(state: MutableMapping[str, Any]) -> pd.DataFrame:
    """Drop the last operation and rebuild the working frame from ``raw_df``."""
    operations = [dict(op) for op in state.get("operations", [])]
    if not operations:
        raise OperationError("There is no operation to undo.")
    remaining = operations[:-1]
    result = apply_operations(state["raw_df"], remaining)
    set_working_df(state, result, remaining)
    return result
