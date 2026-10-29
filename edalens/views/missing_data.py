import streamlit as st
import pandas as pd
import numpy as np
from edalens.utils.data_processing import OperationError, run_operation


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

    st.subheader("🔧 Imputation Settings")
    if st.button("Impute Missing Values with KNN Imputer (Numeric) and Mode (Categorical)"):
        result = _run({"type": "impute"})
        if result is not None:
            df = result.copy()
            st.success("Missing values have been imputed!")
            remaining = df.isna().sum()
            remaining = remaining[remaining > 0]
            if not remaining.empty:
                st.warning(
                    "Some values are still missing because their columns are datetime-like "
                    f"or entirely empty: {', '.join(map(str, remaining.index))}."
                )
            st.write("📊 Preview of the imputed dataset:")
            st.dataframe(df.head())

    if st.session_state.get("is_imputed"):
        st.info("Currently using imputed dataset for visualization.")
    else:
        st.info("Currently using original dataset for visualization. Click the button above to impute missing values.")
