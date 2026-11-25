"""Statistics page: summary table, group comparisons, correlations and outliers."""

import pandas as pd
import plotly.express as px
import streamlit as st

from edalens.utils import stats
from edalens.utils.data_processing import OperationError, run_operation
from edalens.utils.ui import plotly_chart, report, start_charts

MAX_CRAMER_LEVELS = 50  # Categorical columns with more levels are left out of Cramér's V.
FLAGGED_ROWS_SHOWN = 500


def _add_table(frame: pd.DataFrame, title: str, note: str = "") -> None:
    report().add_table(frame, title, note)
    st.toast(f"Added “{title}” to the report.", icon="➕")


def _table_to_report(frame: pd.DataFrame, title: str, key: str, note: str = "") -> None:
    st.button("➕ Add table to report", key=key, type="tertiary", on_click=_add_table, args=(frame, title, note))


def _run(operation: dict, message: str) -> None:
    try:
        run_operation(st.session_state, operation)
    except OperationError as exc:
        st.error(f"{exc} The data was left unchanged.")
        return
    st.success(message)


def _numbers(table: pd.DataFrame) -> dict:
    """Column formats for st.dataframe: 4 significant digits for floats."""
    return {c: st.column_config.NumberColumn(format="%.4g") for c in table.columns
            if pd.api.types.is_float_dtype(table[c])}


def _summary(df: pd.DataFrame) -> None:
    method = st.radio("Outliers counted by", stats.OUTLIER_METHODS, horizontal=True, key="stats_summary_outliers",
                      help="IQR: beyond 1.5 × IQR from the quartiles. Z-score: more than 3 SD from the mean.")
    table = stats.summary_table(df, method)
    st.dataframe(table, hide_index=True, column_config=_numbers(table))
    _table_to_report(table, "Summary statistics", "stats_summary_report",
                     f"{len(df)} rows × {df.shape[1]} columns. Outliers by {method}.")

    st.markdown("#### Duplicate rows")
    duplicates = stats.duplicate_rows(df)
    extra = int(df.duplicated().sum())
    if not extra:
        st.success("No duplicate rows.")
        return
    st.warning(f"{extra} row{'s are' if extra != 1 else ' is'} an exact copy of an earlier row "
               f"({len(duplicates)} rows involved).")
    with st.expander("Show the duplicated rows"):
        st.dataframe(duplicates.head(FLAGGED_ROWS_SHOWN))
    if st.button("Remove duplicate rows (keep the first copy)"):
        _run({"type": "drop_duplicates"}, f"Removed {extra} duplicate rows.")
        st.rerun()


def _result(result: stats.TestResult) -> None:
    (st.success if result.significant else st.info)(result.sentence())
    for note in result.notes:
        st.caption(f"ℹ️ {note}")


def _compare(df: pd.DataFrame, numeric: list, categorical: list) -> None:
    kind = st.radio("Compare", ["A numeric column across groups", "Two categorical columns"], horizontal=True,
                    key="stats_compare_kind")
    if kind.startswith("A numeric"):
        if not numeric or not categorical:
            st.info("This needs a numeric column and a categorical one (set types on the Load data page).")
            return
        left, right, test_col = st.columns([1, 1, 1.4])
        value = left.selectbox("Numeric column", numeric, key="stats_value")
        group = right.selectbox("Groups", categorical, key="stats_group")
        test = test_col.radio("Test", stats.GROUP_TESTS, key="stats_test",
                              help="Parametric tests compare means and assume roughly normal data; rank-based "
                                   "tests compare distributions and suit skewed data or outliers.")
        try:
            result = stats.compare_groups(df, value, group, test)
        except ValueError as exc:
            st.warning(str(exc))
            return
        _result(result)
        st.dataframe(result.groups, hide_index=True, column_config=_numbers(result.groups))
        _table_to_report(result.groups, f"{value} by {group}", "stats_groups_report", result.sentence())
        fig = px.box(df, x=group, y=value, color=group, points="outliers",
                     title=f"{value} by {group}: {result.test}, {stats.format_p(result.p_value)}",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False)
        plotly_chart(fig)
        return
    if len(categorical) < 2:
        st.info("This needs two categorical columns.")
        return
    left, right = st.columns(2)
    a = left.selectbox("First column", categorical, key="stats_chi_a")
    b = right.selectbox("Second column", [c for c in categorical if c != a], key="stats_chi_b")
    try:
        result, table = stats.chi_square(df, a, b)
    except ValueError as exc:
        st.warning(str(exc))
        return
    _result(result)
    st.dataframe(table)
    _table_to_report(table.reset_index(), f"{a} × {b}", "stats_chi_report", result.sentence())


def _correlations(df: pd.DataFrame, numeric: list, categorical: list) -> None:
    left, right = st.columns([1, 2])
    method = left.selectbox("Method", stats.CORRELATION_METHODS, key="stats_corr_method",
                            format_func=str.capitalize,
                            help="Pearson: linear relation. Spearman and Kendall: any monotonic relation, "
                                 "robust to outliers.")
    columns = right.multiselect("Numeric columns", numeric, default=numeric[:12], key="stats_corr_columns")
    if len(columns) >= 2:
        r, p = stats.correlations(df, columns, method)
        fig = px.imshow(r, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
                        title=f"{method.capitalize()} correlation")
        plotly_chart(fig)
        pairs = stats.top_pairs(r, p)
        pairs["Significant"] = pairs["p-value"] < stats.ALPHA
        st.markdown("**Strongest pairs**")
        st.dataframe(pairs, hide_index=True, column_config=_numbers(pairs))
        _table_to_report(pairs, f"Strongest {method} correlations", "stats_pairs_report")
    else:
        st.info("Pick at least two numeric columns.")

    usable = [c for c in categorical if 1 < df[c].nunique() <= MAX_CRAMER_LEVELS]
    if len(usable) >= 2:
        st.markdown("**Categorical columns: Cramér's V** (0 = unrelated, 1 = one determines the other)")
        chosen = st.multiselect("Categorical columns", usable, default=usable[:10], key="stats_cramer_columns")
        if len(chosen) >= 2:
            fig = px.imshow(stats.cramers_v_matrix(df, chosen), text_auto=".2f", zmin=0, zmax=1,
                            color_continuous_scale="Blues", aspect="auto", title="Cramér's V")
            plotly_chart(fig)


def _outliers(df: pd.DataFrame, numeric: list) -> None:
    if not numeric:
        st.info("No numeric columns.")
        return
    left, mid, right = st.columns([1, 1, 2])
    method = left.radio("Method", stats.OUTLIER_METHODS, key="stats_outlier_method")
    default = stats.DEFAULT_THRESHOLDS[method]
    threshold = mid.number_input("Threshold (k)", 0.5, 10.0, default, 0.5, key=f"stats_outlier_k_{method}",
                                 help="IQR: k × IQR beyond the quartiles (1.5 usual, 3 extreme). "
                                      "Z-score: k standard deviations from the mean.")
    columns = right.multiselect("Columns", numeric, default=numeric, key="stats_outlier_columns")
    if not columns:
        return
    flags = stats.outlier_mask(df, columns, method, threshold)
    counts = flags.sum().rename("Outliers").to_frame()
    counts["%"] = (100 * counts["Outliers"] / max(len(df), 1)).round(2)
    st.dataframe(counts.T)
    rows = flags.any(axis=1)
    flagged = int(rows.sum())
    if not flagged:
        st.success("No outliers with these settings.")
        return
    st.warning(f"{flagged} row{'s have' if flagged != 1 else ' has'} at least one outlier "
               f"({100 * flagged / len(df):.1f}% of the data). Outlying values are highlighted.")
    shown = df[rows].head(FLAGGED_ROWS_SHOWN)
    marks = flags.loc[shown.index]
    st.dataframe(shown.style.apply(
        lambda col: ["background-color: rgba(214, 39, 40, 0.25)" if marks.at[i, col.name] else "" for i in col.index]
        if col.name in marks.columns else [""] * len(col), axis=0))
    if flagged > FLAGGED_ROWS_SHOWN:
        st.caption(f"First {FLAGGED_ROWS_SHOWN} of {flagged} rows.")
    if st.button(f"Exclude these {flagged} rows from the data", type="primary",
                 help="Applies to every page, like the Load data operations; undo it there."):
        _run({"type": "remove_outliers", "columns": columns, "method": method, "threshold": threshold},
             f"Excluded {flagged} rows with outliers.")
        st.rerun()


def statistics() -> None:
    start_charts()
    st.subheader("📐 Statistics")
    if "filtered_df" not in st.session_state:
        st.warning("Please upload and process a dataset in the Load data section first.")
        return
    df = st.session_state["filtered_df"]
    numeric = [c for c in st.session_state.get("current_numeric_cols", []) if c in df.columns]
    categorical = [c for c in st.session_state.get("current_categorical_cols", []) if c in df.columns]
    section = st.radio("Section", ["Summary", "Compare groups", "Correlations", "Outliers"], horizontal=True,
                       key="stats_section", label_visibility="collapsed")
    if section == "Summary":
        _summary(df)
    elif section == "Compare groups":
        _compare(df, numeric, categorical)
    elif section == "Correlations":
        _correlations(df, numeric, categorical)
    else:
        _outliers(df, numeric)
