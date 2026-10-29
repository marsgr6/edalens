import streamlit as st
import pandas as pd
import requests
import io
from utils.data_processing import (
    OperationError,
    describe_operation,
    run_operation,
    set_working_df,
    undo_last_operation,
)


def _run(operation, success_message):
    """Apply an operation atomically; report failures without changing state."""
    try:
        df = run_operation(st.session_state, operation)
    except OperationError as exc:
        st.error(f"{exc} The data and operation history were left unchanged.")
        return None
    st.success(success_message(df) if callable(success_message) else success_message)
    return df

def home():
    st.subheader("🏠 Data Upload & Selection")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.markdown("**Or load a CSV from a GitHub link:**")
    github_link = st.text_input(
        "Paste the GitHub link to your CSV file",
        placeholder="e.g., https://raw.githubusercontent.com/marsgr6/r-scripts/master/data/viz_data/sample.csv"
    )
    load_from_github = st.button("Load from GitHub")

    st.markdown("""
    ⚠️ **Important:**  
    Your CSV file must be comma-separated, use a dot (`.`) as the decimal separator, and must not include thousands separators or special symbols.  
    Be aware that Excel often modifies CSV files when saving. Please ensure your file is as standard as possible before uploading.

    📎 **You can also try using sample CSV files:**  
    [Example Datasets on GitHub](https://github.com/marsgr6/r-scripts/tree/master/data/viz_data)
    """)

    if "operations" not in st.session_state:
        st.session_state["operations"] = []
    if "enable_operation_tracking" not in st.session_state:
        st.session_state["enable_operation_tracking"] = False
    if "current_numeric_cols" not in st.session_state:
        st.session_state["current_numeric_cols"] = []
    if "current_categorical_cols" not in st.session_state:
        st.session_state["current_categorical_cols"] = []
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False

    st.subheader("⚙️ Operation Tracking Settings")
    st.session_state["enable_operation_tracking"] = st.checkbox(
        "Enable Operation Tracking for Undo (May Slow Down for Large Datasets)",
        value=False,
        help="When enabled, operations are tracked, allowing you to undo them. This may slow down the app for large datasets as all operations are reapplied each time."
    )
    if st.session_state["enable_operation_tracking"]:
        st.warning("Operation tracking is enabled. This may impact performance for large datasets as all operations will be reapplied after each change.")

    if st.button("Reset All"):
        st.session_state["operations"] = []
        st.session_state.pop("raw_df", None)
        st.session_state.pop("filtered_df", None)
        st.session_state.pop("current_numeric_cols", None)
        st.session_state.pop("current_categorical_cols", None)
        st.session_state.pop("last_uploaded_file", None)
        st.session_state.pop("is_imputed", None)
        st.session_state.pop("imputed_df", None)
        st.session_state.pop("enable_operation_tracking", None)
        st.session_state["data_loaded"] = False
        st.success("Session reset successfully. Please upload a new file.")
        return

    if st.session_state["enable_operation_tracking"] and st.session_state["operations"]:
        if st.button("Undo Last Operation"):
            try:
                undo_last_operation(st.session_state)
            except OperationError as exc:
                st.error(f"{exc} The data and operation history were left unchanged.")
            else:
                if st.session_state["operations"]:
                    st.success("Last operation undone successfully!")
                else:
                    st.info("No operations left; showing the original data.")

    if load_from_github and github_link:
        try:
            if not github_link.endswith('.csv'):
                st.error("The provided link does not point to a CSV file. Please ensure the URL ends with '.csv'.")
            else:
                if "github.com" in github_link and "raw.githubusercontent.com" not in github_link:
                    github_link = github_link.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                response = requests.get(github_link)
                response.raise_for_status()
                csv_content = response.content.decode('utf-8')
                raw_df = pd.read_csv(io.StringIO(csv_content), na_values=['', 'NA', 'N/A', 'na', 'n/a', 'NaN', 'nan', 'NULL', 'null', 'missing', '-', '--'], keep_default_na=True)
                st.session_state["operations"] = []
                st.session_state["last_uploaded_file"] = github_link
                st.session_state.pop("filtered_df", None)
                st.session_state.pop("current_numeric_cols", None)
                st.session_state.pop("current_categorical_cols", None)
                st.session_state.pop("is_imputed", None)
                st.session_state.pop("imputed_df", None)
                st.session_state["raw_df"] = raw_df.copy()
                set_working_df(st.session_state, raw_df.copy(), [])
                st.session_state["data_loaded"] = True
                st.success(f"CSV file successfully loaded from GitHub link: {github_link}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to load CSV from GitHub link: {e}")
        except pd.errors.EmptyDataError:
            st.error("The CSV file is empty or invalid.")
        except Exception as e:
            st.error(f"An error occurred while loading the CSV: {e}")

    if uploaded_file:
        na_values = ['', 'NA', 'N/A', 'na', 'n/a', 'NaN', 'nan', 'NULL', 'null', 'missing', '-', '--']
        if "last_uploaded_file" not in st.session_state or uploaded_file.name != st.session_state["last_uploaded_file"]:
            st.session_state["operations"] = []
            st.session_state["last_uploaded_file"] = uploaded_file.name
            st.session_state.pop("filtered_df", None)
            st.session_state.pop("current_numeric_cols", None)
            st.session_state.pop("current_categorical_cols", None)
            st.session_state.pop("is_imputed", None)
            st.session_state.pop("imputed_df", None)
        raw_df = pd.read_csv(uploaded_file, na_values=na_values, keep_default_na=True)
        st.success(f"File '{uploaded_file.name}' successfully uploaded!")
        st.session_state["raw_df"] = raw_df.copy()
        if "filtered_df" not in st.session_state:
            set_working_df(st.session_state, raw_df.copy(), [])
        st.session_state["data_loaded"] = True

    if st.session_state["data_loaded"] and "filtered_df" in st.session_state:
        # run_operation/undo keep filtered_df equal to replaying the history, so
        # no per-rerun replay is needed. Widgets only read this copy.
        df = st.session_state["filtered_df"].copy()

        st.subheader("🎯 Column Selection")
        selected_columns = st.multiselect(
            "Select columns to keep",
            options=df.columns.tolist(),
            default=df.columns.tolist(),
            key="select_columns"
        )
        if st.button("Apply Column Selection"):
            if selected_columns:
                result = _run(
                    {"type": "select_columns", "columns": selected_columns},
                    "Column selection applied successfully!",
                )
                if result is not None:
                    df = result.copy()
            else:
                st.warning("Please select at least one column.")

        st.subheader("🔧 Assign Variable Types")
        numeric_cols = st.multiselect(
            "Select Numeric Variables",
            options=df.columns.tolist(),
            default=st.session_state["current_numeric_cols"],
            key="numeric_cols"
        )
        categorical_cols = st.multiselect(
            "Select Categorical Variables",
            options=[col for col in df.columns if col not in numeric_cols],
            default=st.session_state["current_categorical_cols"],
            key="categorical_cols"
        )
        if st.button("Apply Variable Types"):
            result = _run(
                {
                    "type": "cast_types",
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols,
                },
                "Variable types applied successfully!",
            )
            if result is not None:
                df = result.copy()

        st.subheader("🧪 Random Subsampling")
        sample_pct = st.slider(
            "Select percentage of data to use",
            min_value=1,
            max_value=100,
            value=100,
            key="sample_pct"
        )
        if st.button("Apply Subsampling"):
            result = _run(
                {"type": "subsample", "sample_pct": sample_pct},
                lambda sampled: f"Dataset subsampled to {len(sampled)} rows.",
            )
            if result is not None:
                df = result.copy()

        st.subheader("🔥 Optional: Melt Data")
        id_vars = st.multiselect(
            "Select ID Variables for Melting",
            options=df.columns.tolist(),
            key="melt_keys"
        )
        if st.button("Apply Melting"):
            if id_vars:
                result = _run(
                    {"type": "melt", "id_vars": id_vars}, "Melting applied successfully!"
                )
                if result is not None:
                    df = result.copy()
            else:
                st.warning("Please select at least one ID variable before melting.")

        st.subheader("📅 Date Feature Engineering (Optional)")
        date_column = st.selectbox(
            "Select Date Column to Extract Features (optional)",
            options=["None"] + df.columns.tolist(),
            index=0,
            key="date_column"
        )
        dayfirst_option = st.checkbox("Use Day First when Parsing Dates (e.g., 31/12/2023)", value=True)
        if st.button("Parse Date and Create Features"):
            if date_column != "None":
                result = _run(
                    {
                        "type": "date_features",
                        "date_column": date_column,
                        "dayfirst": dayfirst_option,
                    },
                    "✅ Date features extracted successfully!",
                )
                if result is not None:
                    df = result.copy()
            else:
                st.warning("Please select a date column to extract features.")

        if st.session_state["enable_operation_tracking"] and st.session_state["operations"]:
            st.subheader("📜 Operation History")
            for i, op in enumerate(st.session_state["operations"], 1):
                st.write(f"{i}. {describe_operation(op)}")

        st.subheader("📊 Preview of the Processed Dataset")
        if "filtered_df" in st.session_state:
            st.dataframe(st.session_state["filtered_df"].head())
        else:
            st.info("No processed data available. Please apply some operations.")

        st.subheader("💾 Download Processed Data")
        if "filtered_df" in st.session_state:
            csv = st.session_state["filtered_df"].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Processed Data as CSV",
                data=csv,
                file_name="processed_dataset.csv",
                mime="text/csv"
            )