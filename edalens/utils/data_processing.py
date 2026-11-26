"""Single implementation of the Home and Missing Data transformations.

Views call ``run_operation`` for both direct and tracked execution, and replay
uses ``apply_operations``; both go through ``apply_operation`` so the two paths
cannot drift. Every operation returns a new DataFrame and never mutates its
input. Failures raise ``OperationError`` and leave session state untouched.
"""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import os
import warnings
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
        numeric = operation.get("numeric", "knn")
        numeric = f"KNN (k={operation.get('k', KNN_NEIGHBORS)})" if numeric == "knn" else numeric
        where = f" in {', '.join(map(str, operation['columns']))}" if operation.get("columns") else ""
        return f"Imputed missing values{where} (numeric: {numeric}, categorical: {operation.get('categorical', 'mode')})"
    if op_type == "drop_columns_missing":
        return f"Dropped columns with more than {operation['max_pct']}% missing"
    if op_type == "rename_columns":
        return "Renamed " + ", ".join(f"{a} → {b}" for a, b in operation["mapping"].items())
    if op_type == "to_datetime":
        return f"Parsed {operation['column']} as dates (dayfirst: {operation['dayfirst']})"
    if op_type == "filter_rows":
        if "values" in operation:
            return f"Kept rows where {operation['column']} is one of {', '.join(map(str, operation['values']))}"
        return f"Kept rows where {operation['column']} is between {operation['min']} and {operation['max']}"
    if op_type == "bin_numeric":
        return f"Binned {operation['column']} into {operation['bins']} {operation['method']} bins as {operation['name']}"
    if op_type == "recode_values":
        return f"Recoded {operation['column']}: " + ", ".join(f"{a} → {b}" for a, b in operation["mapping"].items())
    if op_type == "category_order":
        return f"Ordered {operation['column']}: {', '.join(map(str, operation['order']))}"
    if op_type == "drop_duplicates":
        return "Removed duplicate rows"
    if op_type == "set_index":
        return f"Used {operation['column']} as the row index"
    if op_type == "reset_index":
        return "Moved the row index back to a column"
    if op_type == "remove_outliers":
        return (f"Removed {operation['method']} outliers (threshold {operation['threshold']}) in "
                f"{', '.join(map(str, operation['columns']))}")
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


# Year first (2023-01-05, 2023/1/5 12:00): the day/month order is never ambiguous.
_YEAR_FIRST = r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:\D|$)"


def parse_dates(values: pd.Series, dayfirst: bool) -> pd.Series:
    """Parse ``values`` as dates; unparseable values become NaT.

    ``dayfirst`` only decides ambiguous dates such as 05/01/2023. Year-first
    dates ignore it: pandas would otherwise read 2023-01-05 as 1 May
    (format ``%Y-%d-%m``) and turn 2023-01-20 into NaT.
    """
    if ptypes.is_datetime64_any_dtype(values):
        return values
    text = values.dropna().astype(str)
    with warnings.catch_warnings():  # "Could not infer format": text that is not dates.
        warnings.simplefilter("ignore", UserWarning)
        if len(text) and text.str.match(_YEAR_FIRST).mean() > 0.5:
            parsed = pd.to_datetime(values, errors="coerce", format="ISO8601")
            if parsed.notna().sum() < len(text):  # e.g. 2023/01/05: no single ISO form.
                parsed = pd.to_datetime(values, errors="coerce", format="mixed", dayfirst=False)
            return parsed
        parsed = pd.to_datetime(values, errors="coerce", dayfirst=dayfirst)
    # pandas 3 reads bare month names ("Jan") as dates in year 1; text with no digits is not a date.
    return parsed.where(values.astype(str).str.contains(r"\d", regex=True) | values.isna())


MIN_DATE_SHARE = 0.8  # A text column is a date column when this share of its values parse.


def _parse_share(values: pd.Series, dayfirst: bool) -> float:
    present = values.notna().sum()
    return parse_dates(values, dayfirst).notna().sum() / present if present else 0.0


def guess_dayfirst(values: pd.Series) -> bool:
    """Whether ambiguous dates such as 01/02/2024 are day first.

    The order that parses more values wins. When both parse all of them, the one
    whose dates form a regular series (daily, monthly...) wins: month-first data
    read day-first jumps around inside each month.
    """
    if ptypes.is_datetime64_any_dtype(values):
        return True
    day, month = _parse_share(values, True), _parse_share(values, False)
    if day != month:
        return day > month

    def regular(dayfirst: bool) -> bool:
        dates = pd.DatetimeIndex(parse_dates(values, dayfirst).dropna().drop_duplicates().sort_values())
        return len(dates) >= 3 and pd.infer_freq(dates) is not None

    if regular(False) and not regular(True):
        return False
    return True


def date_columns(df: pd.DataFrame) -> list:
    """Columns that hold dates: datetime columns first, then text that parses as dates.

    Numbers are left out: a column of years or IDs is not a date column.
    """
    found, text = [], []
    for column in df.columns:
        series = df[column]
        if ptypes.is_datetime64_any_dtype(series):
            found.append(column)
        elif not ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series) and series.notna().any():
            sample = series.dropna().astype(str).head(500)
            if sample.str.contains(r"\d").mean() < MIN_DATE_SHARE:
                continue
            share = max(_parse_share(sample, True), _parse_share(sample, False))
            if share >= MIN_DATE_SHARE:
                text.append((-share, column))
    return found + [column for _, column in sorted(text, key=lambda item: item[0])]


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
    parsed = parse_dates(source, operation["dayfirst"])
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


NUMERIC_IMPUTERS = ("knn", "mean", "median", "zero", "interpolate", "none")
CATEGORICAL_IMPUTERS = ("mode", "missing", "none")
MISSING_LABEL = "Missing"


def _impute(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    """Fill missing values; by default KNN for numeric columns and the mode for
    text/categorical/boolean ones (the operation's original behaviour).

    Options: ``numeric`` in NUMERIC_IMPUTERS (``k`` neighbours for KNN, which uses
    every numeric column as features), ``categorical`` in CATEGORICAL_IMPUTERS
    ("missing" fills with the label "Missing"), and ``columns`` to fill only some.
    Datetime-like and all-missing columns are left unchanged. Columns without
    missing values keep their exact dtype; column order and index never change.
    """
    numeric_method = operation.get("numeric", "knn")
    categorical_method = operation.get("categorical", "mode")
    if numeric_method not in NUMERIC_IMPUTERS or categorical_method not in CATEGORICAL_IMPUTERS:
        raise OperationError(f"Unknown imputation method: {numeric_method!r} / {categorical_method!r}.")
    k = int(operation.get("k", KNN_NEIGHBORS))
    if k < 1:
        raise OperationError("KNN needs at least one neighbour.")
    only = operation.get("columns")
    if only:
        _require_columns(df, only, "Columns to impute")
    wanted = set(only) if only else set(df.columns)
    result = df.copy()
    features = [
        column
        for column in result.columns
        if _is_knn_numeric(result[column]) and result[column].notna().any()
    ]
    targets = [column for column in features if result[column].isna().any() and column in wanted]
    if targets and numeric_method in ("mean", "median", "zero", "interpolate"):
        for column in targets:
            series = result[column]
            if numeric_method == "interpolate":
                filled = series.astype("float64").interpolate(limit_direction="both")
            else:
                value = {"mean": series.mean(), "median": series.median(), "zero": 0}[numeric_method]
                filled = series.astype("float64").fillna(value)
            result[column] = filled.astype(_imputed_dtype(series.dtype))
        targets = []
    if numeric_method == "none":
        targets = []
    if targets:
        matrix = result[features].to_numpy(dtype="float64", na_value=np.nan)
        imputer = KNNImputer(
            n_neighbors=k, weights="uniform", metric="nan_euclidean"
        )
        imputed = imputer.fit_transform(matrix)
        if imputed.shape != matrix.shape:
            raise OperationError("KNN imputation changed the data shape unexpectedly.")
        for position, column in enumerate(features):
            if column in targets:
                result[column] = pd.Series(
                    imputed[:, position], index=result.index, name=column
                ).astype(_imputed_dtype(result[column].dtype))
    if categorical_method == "none":
        return result
    for column in result.columns:
        series = result[column]
        if column in wanted and _is_mode_imputed(series) and series.isna().any():
            if categorical_method == "missing" and not ptypes.is_bool_dtype(series):
                if isinstance(series.dtype, pd.CategoricalDtype) and MISSING_LABEL not in series.cat.categories:
                    series = series.cat.add_categories([MISSING_LABEL])
                result[column] = series.fillna(MISSING_LABEL)
                continue
            mode = series.mode(dropna=True)
            if not mode.empty:
                result[column] = series.fillna(mode.iloc[0])
    return result


def _drop_columns_missing(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    limit = float(operation["max_pct"])
    share = df.isna().mean() * 100 if len(df) else df.isna().sum()
    keep = [c for c in df.columns if share[c] <= limit]
    if not keep:
        raise OperationError(f"Every column has more than {limit}% missing values.")
    return df.loc[:, keep].copy()


def _rename_columns(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    mapping = {old: new for old, new in dict(operation["mapping"]).items() if old != new}
    _require_columns(df, list(mapping), "Renamed columns")
    if any(not str(new).strip() for new in mapping.values()):
        raise OperationError("New column names cannot be empty.")
    names = [mapping.get(c, c) for c in df.columns]
    if len(set(names)) != len(names):
        raise OperationError("Renaming would give two columns the same name.")
    return df.rename(columns=mapping)


def _to_datetime(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    column = operation["column"]
    _require_columns(df, [column], "Date column")
    parsed = parse_dates(df[column], operation["dayfirst"])
    if len(df) and parsed.notna().sum() == 0:
        raise OperationError(f"No values in column '{column}' could be parsed as dates.")
    result = df.copy()
    result[column] = parsed
    return result


def _filter_rows(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    column = operation["column"]
    _require_columns(df, [column], "Filter column")
    series = df[column]
    if "values" in operation:
        wanted = set(map(str, operation["values"]))
        keep = series.astype(object).map(str).isin(wanted) & series.notna()
    else:
        numbers = pd.to_numeric(series, errors="coerce") if not ptypes.is_datetime64_any_dtype(series) else series
        low, high = operation.get("min"), operation.get("max")
        if ptypes.is_datetime64_any_dtype(series):
            low, high = pd.Timestamp(low) if low is not None else None, pd.Timestamp(high) if high is not None else None
        keep = numbers.notna()
        if low is not None:
            keep &= numbers >= low
        if high is not None:
            keep &= numbers <= high
    result = df[keep]
    if len(df) and result.empty:
        raise OperationError("The filter would remove every row.")
    return result.copy()


BIN_METHODS = ("equal-width", "quantile")


def _bin_numeric(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    column, bins, name = operation["column"], int(operation["bins"]), operation["name"]
    _require_columns(df, [column], "Binned column")
    if name in df.columns and name != column:
        raise OperationError(f"A column named '{name}' already exists.")
    if bins < 2:
        raise OperationError("Use at least two bins.")
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() < 2:
        raise OperationError(f"Column '{column}' needs numbers to bin.")
    if operation["method"] == "quantile":
        binned = pd.qcut(values, bins, duplicates="drop", precision=3)
    else:
        binned = pd.cut(values, bins, precision=3)
    labels = [str(interval) for interval in binned.cat.categories]
    result = df.copy()
    # Ordered categories keep the bins in numeric order in every plot.
    result[name] = pd.Categorical(binned.astype(object).map(lambda v: str(v) if pd.notna(v) else np.nan),
                                  categories=labels, ordered=True)
    return result


def _recode_values(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    column = operation["column"]
    _require_columns(df, [column], "Recoded column")
    mapping = {str(old): new for old, new in dict(operation["mapping"]).items()}
    series = df[column]
    order = list(series.cat.categories) if isinstance(series.dtype, pd.CategoricalDtype) else None
    text = series.astype(object).map(lambda v: mapping.get(str(v), v) if pd.notna(v) else v)
    result = df.copy()
    if order is not None:  # Keep the order; merged values take the first position.
        categories = list(dict.fromkeys(mapping.get(str(c), c) for c in order))
        result[column] = pd.Categorical(text, categories=categories, ordered=series.cat.ordered)
    else:
        result[column] = text
    return result


def _category_order(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    """Give a text column a fixed category order; plots start from it (values not
    listed go last, in their current order)."""
    column = operation["column"]
    _require_columns(df, [column], "Ordered column")
    series = df[column]
    text = series.astype(object).map(lambda v: str(v) if pd.notna(v) else np.nan)
    present = list(pd.unique(text.dropna()))
    order = [str(v) for v in operation["order"]]
    order = [v for v in order if v in present] + [v for v in present if v not in order]
    result = df.copy()
    result[column] = pd.Categorical(text, categories=order, ordered=True)
    return result


def _drop_duplicates(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    return df.drop_duplicates().copy()


def _remove_outliers(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    from edalens.utils.stats import outlier_mask

    columns = list(operation["columns"])
    _require_columns(df, columns, "Outlier columns")
    flags = outlier_mask(df, columns, operation["method"], float(operation["threshold"])).any(axis=1)
    result = df[~flags]
    if len(df) and result.empty:
        raise OperationError("Every row is an outlier with these settings.")
    return result.copy()


def _set_index(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    column = operation["column"]
    _require_columns(df, [column], "Index column")
    if df[column].isna().any():
        raise OperationError(f"'{column}' has missing values; an index needs a label on every row.")
    if not df[column].is_unique:
        raise OperationError(f"'{column}' has repeated values; an index needs a different label on every row.")
    # The old index is kept as a column when it carries labels (not 0, 1, 2, ...).
    result = df if _default_index(df) else df.rename_axis(df.index.name or "index").reset_index()
    return result.set_index(column)


def _reset_index(df: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    if _default_index(df):
        raise OperationError("The data has no row index to move.")
    name = df.index.name or "index"
    if name in df.columns:
        raise OperationError(f"A column named '{name}' already exists.")
    return df.rename_axis(name).reset_index()


def _default_index(df: pd.DataFrame) -> bool:
    """True for the plain 0, 1, 2, ... index pandas gives (or one left by filtering rows)."""
    return df.index.name is None and ptypes.is_integer_dtype(df.index) and not isinstance(df.index, pd.MultiIndex)


_HANDLERS: dict[str, Callable[[pd.DataFrame, Operation], pd.DataFrame]] = {
    "select_columns": _select_columns,
    "cast_types": _cast_types,
    "subsample": _subsample,
    "melt": _melt,
    "date_features": _date_features,
    "remove_nan": _remove_nan,
    "impute": _impute,
    "drop_columns_missing": _drop_columns_missing,
    "rename_columns": _rename_columns,
    "to_datetime": _to_datetime,
    "filter_rows": _filter_rows,
    "bin_numeric": _bin_numeric,
    "recode_values": _recode_values,
    "category_order": _category_order,
    "drop_duplicates": _drop_duplicates,
    "remove_outliers": _remove_outliers,
    "set_index": _set_index,
    "reset_index": _reset_index,
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
    # By dtype test, not select_dtypes("object"): pandas 3 stores text as "str".
    categorical = [
        column
        for column in df.columns
        if ptypes.is_object_dtype(df[column])
        or isinstance(df[column].dtype, pd.CategoricalDtype)
        or ptypes.is_string_dtype(df[column])
    ]
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
        # convert_dates=False: pandas would parse text columns named like "date" month-first;
        # real datetime columns come back from the table schema either way.
        raw_df = pd.read_json(path, orient="table", convert_dates=False)
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
