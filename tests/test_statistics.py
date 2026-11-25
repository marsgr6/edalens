"""Statistics page and helpers: summary, group tests, chi-square, correlations, outliers."""

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from edalens.utils import stats
from edalens.utils.data_processing import apply_operation


def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 200
    group = rng.choice(["a", "b", "c"], n)
    value = rng.normal(0, 1, n) + np.where(group == "c", 2.0, 0.0)
    value[:3] = [15, -12, 20]  # Clear outliers.
    return pd.DataFrame(
        {
            "value": value,
            "other": value * 2 + rng.normal(0, 0.1, n),
            "noise": rng.normal(0, 1, n),
            "group": group,
            "kind": np.where(group == "c", "x", rng.choice(["x", "y"], n)),
        }
    )


def test_summary_table_counts_missing_unique_and_outliers():
    data = frame()
    data.loc[5:9, "noise"] = np.nan
    table = stats.summary_table(data).set_index("Column")
    assert table.loc["noise", "Missing"] == 5 and table.loc["noise", "Missing %"] == 2.5
    assert (
        table.loc["group", "Unique"] == 3
        and table.loc["group", "Type"] == "categorical"
    )
    assert table.loc["value", "Outliers"] >= 3
    assert table.loc["value", "Mean"] == pytest.approx(data["value"].mean())


def test_duplicates_are_found_and_removed():
    data = pd.concat([frame().head(5), frame().head(2)], ignore_index=True)
    assert len(stats.duplicate_rows(data)) == 4
    assert len(apply_operation(data, {"type": "drop_duplicates"})) == 5


@pytest.mark.parametrize(
    "groups, test, name",
    [
        (["a", "c"], stats.PARAMETRIC, "Welch's t-test"),
        (["a", "c"], stats.NONPARAMETRIC, "Mann-Whitney U"),
        (["a", "b", "c"], stats.PARAMETRIC, "One-way ANOVA"),
        (["a", "b", "c"], stats.NONPARAMETRIC, "Kruskal-Wallis H"),
    ],
)
def test_group_tests_find_the_shifted_group(groups, test, name):
    data = frame()
    result = stats.compare_groups(
        data[data["group"].isin(groups)], "value", "group", test
    )
    assert result.test == name and result.significant
    assert result.effect_label in ("medium", "large")
    assert sorted(result.groups["group"]) == groups


def test_t_test_matches_scipy():
    from scipy.stats import ttest_ind

    data = frame()
    a, c = (data.loc[data["group"] == g, "value"] for g in ("a", "c"))
    result = stats.compare_groups(
        data[data["group"].isin(["a", "c"])], "value", "group"
    )
    assert abs(result.p_value - ttest_ind(a, c, equal_var=False).pvalue) < 1e-12


def test_chi_square_and_cramers_v():
    result, table = stats.chi_square(frame(), "group", "kind")
    assert result.significant and 0 < result.effect <= 1
    assert table.shape == (3, 2)


def test_correlations_with_p_values_and_top_pairs():
    r, p = stats.correlations(frame(), ["value", "other", "noise"], "spearman")
    assert r.loc["value", "other"] > 0.9 and p.loc["value", "other"] < 0.001
    pairs = stats.top_pairs(r, p)
    assert {pairs.loc[0, "Variable 1"], pairs.loc[0, "Variable 2"]} == {
        "value",
        "other",
    }


@pytest.mark.parametrize("method", stats.OUTLIER_METHODS)
def test_outliers_are_flagged_and_removed(method):
    data = frame()
    flags = stats.outlier_mask(data, ["value"], method)
    assert flags.loc[:2, "value"].all()
    kept = apply_operation(
        data,
        {
            "type": "remove_outliers",
            "columns": ["value"],
            "method": method,
            "threshold": 3.0,
        },
    )
    assert len(kept) < len(data) and not kept.index.isin([0, 1, 2]).any()


def statistics_page():
    from edalens.views.statistics import statistics

    statistics()


def page(data: pd.DataFrame) -> AppTest:
    app = AppTest.from_function(statistics_page, default_timeout=30)
    app.session_state["filtered_df"] = data
    app.session_state["raw_df"] = data
    app.session_state["operations"] = []
    app.session_state["current_numeric_cols"] = ["value", "other", "noise"]
    app.session_state["current_categorical_cols"] = ["group", "kind"]
    return app


@pytest.mark.parametrize(
    "section", ["Summary", "Compare groups", "Correlations", "Outliers"]
)
def test_every_section_runs(section):
    app = page(frame())
    app.run()
    app.radio[0].set_value(section)
    app.run()
    assert not app.exception, app.exception[0].message
    if section == "Compare groups":
        assert any("Welch" in s.value or "ANOVA" in s.value for s in app.success)
        next(r for r in app.radio if r.label == "Compare").set_value(
            "Two categorical columns"
        )
        app.run()
        assert not app.exception and any("Chi-square" in s.value for s in app.success)


def test_summary_goes_to_the_report_and_outliers_can_be_excluded():
    app = page(frame())
    app.run()
    next(b for b in app.button if b.label == "➕ Add table to report").click()
    app.run()
    item = app.session_state["report"].items[0]
    assert item.kind == "table" and "Column" in item.frame().columns
    app.radio[0].set_value("Outliers")
    app.run()
    next(b for b in app.button if b.label.startswith("Exclude these")).click()
    app.run()
    assert not app.exception
    assert len(app.session_state["filtered_df"]) < 200
    assert app.session_state["operations"][-1]["type"] == "remove_outliers"
