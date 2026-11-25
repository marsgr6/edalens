import streamlit as st
import pandas as pd
import numpy as np
from edalens.utils.data_processing import (
    CATEGORICAL_IMPUTERS, KNN_NEIGHBORS, NUMERIC_IMPUTERS, OperationError, run_operation,
)

NUMERIC_LABELS = {
    "knn": "KNN (similar rows)", "mean": "Mean", "median": "Median", "zero": "Zero",
    "interpolate": "Interpolate (ordered data, e.g. time series)", "none": "Leave missing",
}
CATEGORICAL_LABELS = {"mode": "Most frequent value", "missing": 'A "Missing" label', "none": "Leave missing"}


def _missing_info(df):
    missing_data = df.isna().sum()
    missing_percentage = (missing_data / len(df) * 100).round(2) if len(df) else missing_data * 0.0
    missing_info = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_data,
        "Percentage Missing (%)": missing_percentage
    })
    return missing_data, missing_info


def _run(operation):
    """Apply an operation atomically; report failures without changing state."""
    try:
        return run_operation(st.session_state, operation)
    except OperationError as exc:
        st.error(f"{exc} The data and operation history were left unchanged.")
        return None


def missing_data_imputation():
    st.subheader("🔄 Missing Data Imputation")

    if "filtered_df" not in st.session_state:
        st.warning("Please upload and process a dataset in the Home section first.")
        return

    # Display-only copy; the shared working frame changes only via run_operation.
    df = st.session_state["filtered_df"].copy()
    df = df.apply(lambda x: x.replace([None], np.nan))

    st.subheader("📉 Missing Data Overview")
    missing_data, missing_info = _missing_info(df)
    st.dataframe(missing_info)

    if missing_data.sum() == 0:
        st.info("No missing values found in the dataset.")
        return

    st.subheader("🧹 Data Cleaning Options")
    if st.button("Remove Rows with Any Missing Values"):
        original_rows = df.shape[0]
        result = _run({"type": "remove_nan"})
        if result is not None:
            df = result.copy()
            st.success(f"Removed rows with missing values! {original_rows - df.shape[0]} rows dropped. New size: {df.shape[0]} rows.")
            missing_data, missing_info = _missing_info(df)
            st.dataframe(missing_info)
            if missing_data.sum() == 0:
                st.info("No missing values left after removal.")
                return

    st.subheader("🗑️ Drop Mostly-Empty Columns")
    limit = st.slider("Drop columns with more than this % missing", 0, 100, 50, 5, key="missing_drop_pct")
    too_empty = [c for c in df.columns if len(df) and df[c].isna().mean() * 100 > limit]
    st.caption(f"Would drop: {', '.join(map(str, too_empty))}." if too_empty else "No column is above this limit.")
    if st.button("Drop these columns", disabled=not too_empty):
        result = _run({"type": "drop_columns_missing", "max_pct": limit})
        if result is not None:
            st.success(f"Dropped {len(too_empty)} column{'s' if len(too_empty) != 1 else ''}.")
            df = result.copy()

    st.subheader("🔧 Imputation Settings")
    with_missing = [c for c in df.columns if df[c].isna().any()]
    columns = st.multiselect("Columns to fill", with_missing, default=with_missing, key="impute_columns",
                             help="Only these columns are filled; KNN still uses every numeric column to find similar rows.")
    left, right = st.columns(2)
    numeric = left.selectbox("Numeric columns", NUMERIC_IMPUTERS, format_func=NUMERIC_LABELS.get, key="impute_numeric",
                             help="KNN fills a value from the k most similar rows (by the other numeric columns). "
                                  "Mean/median are quick but shrink the spread. Interpolate follows the row order.")
    k = left.number_input("Neighbours (k)", 1, 50, KNN_NEIGHBORS, key="impute_k") if numeric == "knn" else KNN_NEIGHBORS
    categorical = right.selectbox("Text / categorical columns", CATEGORICAL_IMPUTERS,
                                  format_func=CATEGORICAL_LABELS.get, key="impute_categorical")
    if st.button("Impute missing values", type="primary", disabled=not columns):
        operation = {"type": "impute", "numeric": numeric, "k": int(k), "categorical": categorical}
        if set(columns) != set(with_missing):
            operation["columns"] = columns
        result = _run(operation)
        if result is not None:
            df = result.copy()
            st.success("Missing values have been imputed!")
            remaining = df.isna().sum()
            remaining = remaining[remaining > 0]
            if not remaining.empty:
                st.warning(
                    "Some values are still missing (columns not selected, left missing on purpose, "
                    f"datetime-like or entirely empty): {', '.join(map(str, remaining.index))}."
                )
            st.write("📊 Preview of the imputed dataset:")
            st.dataframe(df.head())

    if st.session_state.get("is_imputed"):
        st.info("Currently using imputed dataset for visualization.")
    else:
        st.info("Currently using original dataset for visualization. Click the button above to impute missing values.")
