"""Date parsing: the day-first option only decides ambiguous dates."""

import pandas as pd
import pytest

from edalens.utils.data_processing import apply_operations, parse_dates


def days(values, dayfirst):
    return parse_dates(pd.Series(values), dayfirst).dt.strftime("%Y-%m-%d").tolist()


@pytest.mark.parametrize("dayfirst", [True, False])
def test_year_first_dates_ignore_the_day_first_option(dayfirst):
    # pandas alone reads 2023-01-05 as 1 May with dayfirst=True and drops 2023-01-20.
    assert days(["2023-01-05", "2023-01-20", "2023-02-01 10:30"], dayfirst) == [
        "2023-01-05",
        "2023-01-20",
        "2023-02-01",
    ]


def test_day_first_and_month_first_decide_ambiguous_dates():
    assert days(["05/01/2023", "20/01/2023"], dayfirst=True) == [
        "2023-01-05",
        "2023-01-20",
    ]
    assert days(["01/05/2023", "01/20/2023"], dayfirst=False) == [
        "2023-01-05",
        "2023-01-20",
    ]


def test_text_and_datetimes_pass_through():
    assert parse_dates(pd.Series(["a", None]), True).isna().all()
    stamps = pd.Series(pd.to_datetime(["2023-01-05"]))
    assert parse_dates(stamps, True) is stamps


def test_date_features_read_iso_dates_with_day_first_checked():
    frame = pd.DataFrame({"when": ["2023-01-05", "2023-01-20"]})
    result = apply_operations(
        frame, [{"type": "date_features", "date_column": "when", "dayfirst": True}]
    )
    assert result["Month"].tolist() == [1, 1]
    assert result["Day"].tolist() == [5, 20]
