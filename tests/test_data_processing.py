"""Behavioral baseline for the existing operation replay implementation."""

import pandas as pd
from pandas.testing import assert_frame_equal

from edalens.utils.data_processing import apply_operations


def test_replay_matches_expected_values_and_does_not_mutate_source():
    source = pd.DataFrame({"value": ["1", "2", "3"], "label": ["a", "b", "c"]})
    original = source.copy(deep=True)
    operations = [
        {"type": "cast_types", "numeric_cols": ["value"], "categorical_cols": []},
        {"type": "select_columns", "columns": ["value"]},
    ]
    result = apply_operations(source, operations)
    assert_frame_equal(result, pd.DataFrame({"value": [1, 2, 3]}))
    assert_frame_equal(source, original)
    # Replaying without the final operation restores the selected-away column.
    restored = apply_operations(source, operations[:-1])
    assert list(restored.columns) == ["value", "label"]
    assert restored["label"].tolist() == ["a", "b", "c"]


def test_remove_missing_preserves_index_and_column_order():
    source = pd.DataFrame({"b": [1.0, None, 3.0], "a": ["x", "y", "z"]})
    result = apply_operations(source, [{"type": "remove_nan"}])
    assert_frame_equal(result, source.loc[[0, 2]])


def test_melt_preserves_identifiers_and_values():
    source = pd.DataFrame({"id": ["a", "b"], "x": [1, 2], "y": [3, 4]})
    result = apply_operations(source, [{"type": "melt", "id_vars": ["id"]}])
    expected = pd.DataFrame(
        {
            "id": ["a", "b", "a", "b"],
            "variable": ["x", "x", "y", "y"],
            "value": [1, 2, 3, 4],
        }
    )
    assert_frame_equal(result, expected)


def test_sampling_is_repeatable_and_does_not_modify_original():
    source = pd.DataFrame({"value": range(20)})
    operation = [{"type": "subsample", "sample_pct": 50}]
    first = apply_operations(source, operation)
    assert len(first) == 10
    assert_frame_equal(first, apply_operations(source, operation))
    assert source["value"].tolist() == list(range(20))
