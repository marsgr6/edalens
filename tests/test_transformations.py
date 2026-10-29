"""Regression tests for shared transformations, replay, and atomic session updates."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from edalens.utils.data_processing import (
    OperationError,
    apply_operation,
    apply_operations,
    run_operation,
    undo_last_operation,
)


def mixed_frame() -> pd.DataFrame:
    """Small frame covering dtypes that imputation previously dropped."""
    return pd.DataFrame(
        {
            "when": pd.to_datetime(
                ["2024-01-01", "2024-01-02", None, "2024-01-04", "2024-01-05"]
            ),
            "label": pd.Categorical(["a", "b", None, "a", "a"]),
            "small": np.array([1, 2, 3, 4, 5], dtype="int32"),
            "measure": [1.0, np.nan, 3.0, 4.0, 5.0],
            "count": pd.array([1, 2, None, 2, 1], dtype="Int64"),
            "text": ["x", None, "y", "x", "x"],
            "flag": [True, False, True, False, True],
        },
        index=[10, 20, 30, 40, 50],
    )


def frame_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy(deep=True)


# --- column loss -------------------------------------------------------------


def test_impute_preserves_columns_order_index_and_unaffected_dtypes():
    source = mixed_frame()
    original = frame_snapshot(source)

    result = apply_operation(source, {"type": "impute"})

    assert list(result.columns) == list(source.columns)
    assert result.index.equals(source.index)
    # Columns without imputed values keep their exact dtype and values.
    for column in ("small", "flag"):
        assert_series_equal(result[column], source[column])
    # Datetime is not imputed; its missing value is preserved explicitly.
    assert_series_equal(result["when"], source["when"])
    assert result["measure"].dtype == "float64"
    assert result["measure"].notna().all()
    assert result["label"].dtype == "category"
    assert result.loc[30, "label"] == "a"
    assert result.loc[20, "text"] == "x"
    assert_frame_equal(source, original)


def test_impute_turns_integer_columns_with_fractional_results_into_float():
    source = pd.DataFrame(
        {
            "nullable": pd.array([1, 2, None], dtype="Int64"),
            "float_int": [1.0, 2.0, np.nan],
            "complete": [10, 20, 30],
        }
    )

    result = apply_operation(source, {"type": "impute"})

    # Documented rule: an integer column that receives imputed values becomes
    # the matching float dtype so fractional KNN means are not truncated.
    assert result["nullable"].dtype == "Float64"
    assert result["nullable"].tolist() == [1.0, 2.0, 1.5]
    assert result["float_int"].tolist() == [1.0, 2.0, 1.5]
    assert result["complete"].dtype == source["complete"].dtype


def test_impute_keeps_all_missing_columns_instead_of_dropping_them():
    source = pd.DataFrame(
        {
            "empty_num": [np.nan, np.nan, np.nan],
            "value": [1.0, np.nan, 3.0],
            "empty_text": pd.Series([None, None, None], dtype=object),
        }
    )

    result = apply_operation(source, {"type": "impute"})

    assert list(result.columns) == ["empty_num", "value", "empty_text"]
    assert result["empty_num"].isna().all()
    assert result["empty_text"].isna().all()
    assert result["value"].tolist() == [1.0, 2.0, 3.0]


def test_impute_handles_empty_frames():
    source = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": []})

    assert_frame_equal(apply_operation(source, {"type": "impute"}), source)


# --- silent failures ---------------------------------------------------------


def test_replay_raises_instead_of_returning_a_partial_frame():
    source = pd.DataFrame({"value": ["1", "2"], "label": ["a", "b"]})
    operations = [
        {"type": "cast_types", "numeric_cols": ["value"], "categorical_cols": []},
        {"type": "select_columns", "columns": ["missing"]},
    ]

    with pytest.raises(OperationError, match="Step 2"):
        apply_operations(source, operations)


def test_unknown_operation_raises():
    with pytest.raises(OperationError, match="Unknown operation"):
        apply_operation(pd.DataFrame({"a": [1]}), {"type": "explode"})


def test_melt_without_value_columns_raises():
    source = pd.DataFrame({"id": ["a"], "x": [1]})

    with pytest.raises(OperationError, match="non-ID column"):
        apply_operation(source, {"type": "melt", "id_vars": ["id", "x"]})


def test_remove_missing_that_would_empty_the_frame_raises():
    source = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, "x"]})

    with pytest.raises(OperationError, match="every row"):
        apply_operation(source, {"type": "remove_nan"})


def test_date_features_reject_unparseable_and_colliding_columns():
    unparseable = pd.DataFrame({"date": ["soon", "later"]})
    with pytest.raises(OperationError, match="could be parsed"):
        apply_operation(
            unparseable,
            {"type": "date_features", "date_column": "date", "dayfirst": True},
        )

    colliding = pd.DataFrame({"date": ["2024-01-01"], "Year": ["keep me"]})
    with pytest.raises(OperationError, match="Year"):
        apply_operation(
            colliding,
            {"type": "date_features", "date_column": "date", "dayfirst": False},
        )


def test_cast_types_rejects_unknown_columns():
    with pytest.raises(OperationError, match="missing"):
        apply_operation(
            pd.DataFrame({"a": [1]}),
            {"type": "cast_types", "numeric_cols": ["missing"], "categorical_cols": []},
        )


# --- dtype and index handling -----------------------------------------------


def test_cast_to_categorical_keeps_missing_values_missing():
    source = pd.DataFrame(
        {"code": [1, None, 3], "name": ["a", None, "c"]}, index=[5, 6, 7]
    )

    result = apply_operation(
        source,
        {
            "type": "cast_types",
            "numeric_cols": [],
            "categorical_cols": ["code", "name"],
        },
    )

    assert result["name"].isna().tolist() == [False, True, False]
    assert result["code"].tolist()[0] == "1.0"
    assert pd.isna(result.loc[6, "code"])
    assert result.index.equals(source.index)


def test_cast_to_numeric_keeps_nullable_integers_and_rejects_datetimes():
    source = pd.DataFrame(
        {
            "n": pd.array([1, None], dtype="Int64"),
            "when": pd.to_datetime(["2024-01-01", None]),
        }
    )

    result = apply_operation(
        source, {"type": "cast_types", "numeric_cols": ["n"], "categorical_cols": []}
    )
    assert_frame_equal(result, source)

    with pytest.raises(OperationError, match="datetime"):
        apply_operation(
            source,
            {"type": "cast_types", "numeric_cols": ["when"], "categorical_cols": []},
        )


def test_select_columns_keeps_frame_order_index_and_dtypes():
    source = mixed_frame()

    result = apply_operation(
        source, {"type": "select_columns", "columns": ["text", "when"]}
    )

    assert_frame_equal(result, source[["when", "text"]])


def test_date_features_keep_index_and_define_feature_dtypes():
    source = pd.DataFrame(
        {"date": ["31/12/2023", "bad", None], "other": [1, 2, 3]},
        index=["r1", "r2", "r3"],
    )

    result = apply_operation(
        source, {"type": "date_features", "date_column": "date", "dayfirst": True}
    )

    assert list(result.columns) == [
        "date",
        "other",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "Hour",
    ]
    assert result.index.equals(source.index)
    assert result.loc["r1", "date"] == "2023-12-31"
    # Unparseable values become missing, never the literal text "NaT".
    assert result["date"].isna().tolist() == [False, True, True]
    assert result["Year"].dtype == "float64"
    assert result.loc["r1", "Year"] == 2023
    assert result["other"].dtype == source["other"].dtype

    complete = apply_operation(
        source.loc[["r1"]],
        {"type": "date_features", "date_column": "date", "dayfirst": True},
    )
    assert complete["Year"].dtype == "int64"


def test_subsample_is_deterministic_and_keeps_row_order_and_dtypes():
    source = mixed_frame()
    operation = {"type": "subsample", "sample_pct": 60}

    first = apply_operation(source, operation)

    assert len(first) == 3
    assert first.index.is_monotonic_increasing
    assert set(first.index) <= set(source.index)
    assert first.dtypes.equals(source.dtypes)
    assert_frame_equal(first, apply_operation(source, operation))
    assert_frame_equal(
        apply_operation(source, {"type": "subsample", "sample_pct": 100}), source
    )


def test_empty_frames_pass_through_shape_preserving_operations():
    source = pd.DataFrame(
        {"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype=object)}
    )

    for operation in (
        {"type": "remove_nan"},
        {"type": "subsample", "sample_pct": 50},
        {"type": "select_columns", "columns": ["a", "b"]},
    ):
        assert_frame_equal(apply_operation(source, operation), source)


# --- direct execution vs. replay --------------------------------------------


OPERATION_SEQUENCE = [
    {"type": "cast_types", "numeric_cols": ["measure"], "categorical_cols": ["text"]},
    {"type": "impute"},
    {"type": "subsample", "sample_pct": 80},
    {
        "type": "select_columns",
        "columns": ["when", "label", "measure", "count", "text"],
    },
    {"type": "remove_nan"},
]


def test_direct_execution_matches_replay_and_preserves_source():
    source = mixed_frame()
    original = frame_snapshot(source)

    direct = source
    for operation in OPERATION_SEQUENCE:
        direct = apply_operation(direct, operation)

    assert_frame_equal(direct, apply_operations(source, OPERATION_SEQUENCE))
    assert_frame_equal(source, original)


@pytest.mark.parametrize("tracking", [False, True])
def test_run_operation_is_atomic_on_failure(tracking):
    source = mixed_frame()
    state = {
        "raw_df": source,
        "filtered_df": source.copy(),
        "operations": [],
        "enable_operation_tracking": tracking,
    }
    run_operation(state, {"type": "select_columns", "columns": ["measure", "text"]})
    before_df = frame_snapshot(state["filtered_df"])
    before_ops = list(state["operations"])
    before_types = (state["current_numeric_cols"], state["current_categorical_cols"])

    with pytest.raises(OperationError):
        run_operation(state, {"type": "melt", "id_vars": ["measure", "text"]})

    assert_frame_equal(state["filtered_df"], before_df)
    assert state["operations"] == before_ops
    assert (
        state["current_numeric_cols"],
        state["current_categorical_cols"],
    ) == before_types


def test_direct_and_tracked_sessions_agree_and_undo_restores_previous_state():
    direct = {
        "raw_df": mixed_frame(),
        "filtered_df": mixed_frame(),
        "operations": [],
        "enable_operation_tracking": False,
    }
    tracked = {
        **direct,
        "filtered_df": mixed_frame(),
        "enable_operation_tracking": True,
    }
    snapshots = []

    for operation in OPERATION_SEQUENCE:
        snapshots.append(frame_snapshot(tracked["filtered_df"]))
        run_operation(direct, operation)
        run_operation(tracked, operation)
        assert_frame_equal(direct["filtered_df"], tracked["filtered_df"])

    assert direct["operations"] == OPERATION_SEQUENCE
    assert tracked["is_imputed"] is True

    for expected in reversed(snapshots):
        undo_last_operation(tracked)
        assert_frame_equal(tracked["filtered_df"], expected)

    assert tracked["operations"] == []
    assert tracked["is_imputed"] is False
    assert_frame_equal(tracked["filtered_df"], mixed_frame())
    assert_frame_equal(tracked["raw_df"], mixed_frame())
