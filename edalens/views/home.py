import streamlit as st
import pandas as pd
import requests
import io
import re
from edalens.utils.data_processing import (
    BIN_METHODS,
    OperationError,
    parse_dates,
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

def _key(name):
    """Widget key tied to the current working frame (see ``set_working_df``)."""
    return f"{name}_{st.session_state.get('data_version', 0)}"


CATEGORY_SORTS = ("Current order", "Alphabetical (A → Z)", "Alphabetical (Z → A)", "Most frequent first",
                  "Least frequent first", "Calendar (months, weekdays)")
MAX_RECODE_VALUES = 200


def _text_values(series):
    """Distinct values as text, in the column's current order (category order if it has one)."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.dropna().astype(str))
        return [str(c) for c in series.cat.categories if str(c) in present]
    return list(pd.unique(series.dropna().astype(str)))


def sorted_labels(series, how):
    """The labels of ``series`` sorted by one of CATEGORY_SORTS."""
    from edalens.views.visualization import natural_sort

    labels = _text_values(series)
    counts = series.dropna().astype(str).value_counts()
    if how.startswith("Alphabetical"):
        return sorted(labels, key=_natural_key, reverse="Z → A" in how)
    if how == "Most frequent first":
        return sorted(labels, key=lambda v: -counts[v])
    if how == "Least frequent first":
        return sorted(labels, key=lambda v: counts[v])
    if how.startswith("Calendar"):
        return natural_sort(labels)
    return labels


def _natural_key(text):
    """"item 2" before "item 10": digits compare as numbers."""
    return [(0, int(part), "") if part.isdigit() else (1, 0, part.lower()) for part in re.split(r"(\d+)", text)]


RENAME, ORDER, DATES = "✏️ Rename a column", "🔤 Order category labels", "📆 Parse a column as dates"
FILTER, BINS, RECODE = "🔎 Filter rows", "📏 Bin a numeric column", "🔁 Recode values of a category"
# One tool shown at a time, picked from a row of options. Expanders were tried: opening
# one from the app after a step re-creates it, and the page jumps under the pointer.
TOOLS = (RENAME, ORDER, DATES, FILTER, BINS, RECODE)
FLASH_KEY = "_cleaning_messages"  # The last result of each tool, shown under it.
RENAMED_KEY = "_cleaning_renamed"  # {old: new} from the last rename, for the column pickers.
CHOICES_KEY = "_cleaning_choices"  # Last value of each picker and setting, by name.
PICK_PREFIX = "clean_"


def _remembered(name, options=None):
    """Put back a picker's or setting's last value when Streamlit dropped its widget state.

    Applying a step redraws the page before the tools below it are drawn, and
    Streamlit forgets the state of widgets a run did not draw.
    """
    key = PICK_PREFIX + name
    saved = st.session_state.setdefault(CHOICES_KEY, {})
    if key not in st.session_state and name in saved and (options is None or saved[name] in options):
        st.session_state[key] = saved[name]
    if options is not None and key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    return key


def _pick(label, options, name):
    """A column picker that stays on its column after a step is applied.

    Its key does not change with the data (unlike _key), it follows a renamed column,
    and it falls back to the first option only when its column is gone or no longer
    fits this tool.
    """
    value = st.selectbox(label, options, key=_remembered(name, options))
    st.session_state[CHOICES_KEY][name] = value
    return value


def _setting(widget, label, *args, name, **kwargs):
    """A tool setting (sort, bins, day first...) that keeps its value across steps."""
    key = _remembered(name)
    if key in st.session_state:
        kwargs.pop("value", None)  # Streamlit refuses a default together with a set state.
    value = widget(label, *args, key=key, **kwargs)
    st.session_state[CHOICES_KEY][name] = value
    return value


def _last_result(tool):
    """The tool's last result, under its controls (above them it would push the button down)."""
    message = st.session_state.get(FLASH_KEY, {}).get(tool)
    if message:
        st.success(message)


def _apply(operation, success_message, tool):
    """Run a cleaning operation, then redraw the page so every widget sees the new columns."""
    try:
        df = run_operation(st.session_state, operation)
    except OperationError as exc:
        st.error(f"{exc} The data and operation history were left unchanged.")
        return
    message = success_message(df) if callable(success_message) else success_message
    if operation["type"] == "rename_columns":
        st.session_state[RENAMED_KEY] = dict(operation["mapping"])
    st.session_state[FLASH_KEY] = {**st.session_state.get(FLASH_KEY, {}), tool: message}
    st.rerun()


def _cleaning(df):
    """Rename, parse dates, filter, bin, recode and order categories; each is an operation."""
    st.subheader("🧹 Data Cleaning")
    columns = df.columns.tolist()
    text_cols = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])
                 and not pd.api.types.is_datetime64_any_dtype(df[c])]
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    # Pickers that were on a column just renamed move to its new name (before any widget is drawn).
    renamed = st.session_state.pop(RENAMED_KEY, {})
    saved = st.session_state.setdefault(CHOICES_KEY, {})
    for name, value in list(saved.items()):
        if isinstance(value, str) and value in renamed:
            saved[name] = renamed[value]
            st.session_state.pop(PICK_PREFIX + name, None)  # Restored from saved below.
    tool = st.radio("Tool", TOOLS, horizontal=True, key=PICK_PREFIX + "tool", label_visibility="collapsed")

    if tool == RENAME:
        with st.container(border=True):
            column = _pick("Column", columns, "rename_column")
            new_name = st.text_input("New name", value=str(column), key=_key(f"rename_to_{column}"))
            if st.button("Rename", key=_key("rename_apply"), disabled=new_name.strip() in ("", str(column))):
                _apply({"type": "rename_columns", "mapping": {column: new_name.strip()}},
                              f"Renamed {column} to {new_name.strip()}.", RENAME)
            _last_result(RENAME)

    if tool == ORDER:
        with st.container(border=True):
            st.caption("Sets the order of a text column's labels for every page; on the Visualization page "
                       "each plot's Custom Order starts from it and can still be changed there.")
            if not text_cols:
                st.info("No text columns.")
            else:
                column = _pick("Column", text_cols, "order_column")
                labels = _text_values(df[column])
                if len(labels) > MAX_RECODE_VALUES:
                    st.warning(f"{column} has {len(labels)} labels; ordering works best with fewer.")
                how = _setting(st.selectbox, "Sort", CATEGORY_SORTS, name="order_how")
                order = st.multiselect("Order (remove and re-add a label to move it to the end)", labels,
                                       default=sorted_labels(df[column], how), key=_key(f"order_{column}_{how}"))
                if st.button("Apply order", key=_key("order_apply")):
                    _apply({"type": "category_order", "column": column, "order": order},
                                  f"{column} now follows this order." +
                                  (" Labels left out go last." if len(order) < len(labels) else ""), ORDER)
            _last_result(ORDER)

    if tool == DATES:
        with st.container(border=True):
            if not text_cols:
                st.info("No text columns.")
            else:
                column = _pick("Column", text_cols, "todate_column")
                dayfirst = _setting(st.checkbox, "Day first (31/12/2023)", value=True, name="todate_dayfirst")
                parsed = parse_dates(df[column], dayfirst)
                failed = int((parsed.isna() & df[column].notna()).sum())
                st.caption(f"{int(parsed.notna().sum())} values read as dates" +
                           (f"; {failed} cannot be read and would become missing." if failed else "."))
                if st.button("Parse dates", key=_key("todate_apply")):
                    _apply({"type": "to_datetime", "column": column, "dayfirst": dayfirst},
                                  f"{column} is now a date column.", DATES)
            _last_result(DATES)

    if tool == FILTER:
        with st.container(border=True):
            column = _pick("Column", columns, "filter_column")
            series = df[column]
            operation = None
            if pd.api.types.is_datetime64_any_dtype(series) and series.notna().any():
                low, high = series.min().date(), series.max().date()
                chosen = st.date_input("Keep dates from / to", (low, high), min_value=low, max_value=high,
                                       key=_key(f"filter_dates_{column}"))
                if isinstance(chosen, (tuple, list)) and len(chosen) == 2:
                    operation = {"type": "filter_rows", "column": column, "min": str(chosen[0]),
                                 "max": f"{chosen[1]} 23:59:59.999999"}
            elif column in numeric_cols and series.notna().any():
                low, high = float(series.min()), float(series.max())
                if low < high:
                    chosen = st.slider("Keep values between", low, high, (low, high), key=_key(f"filter_range_{column}"))
                    operation = {"type": "filter_rows", "column": column, "min": chosen[0], "max": chosen[1]}
                else:
                    st.info(f"{column} has a single value.")
            else:
                values = _text_values(series)
                chosen = st.multiselect("Keep rows with these values", values, default=values,
                                        key=_key(f"filter_values_{column}"))
                operation = {"type": "filter_rows", "column": column, "values": chosen}
            st.caption("Rows with a missing value in this column are removed too.")
            if operation and st.button("Apply filter", key=_key("filter_apply")):
                _apply(operation, lambda kept: f"Kept {len(kept)} of {len(df)} rows.", FILTER)
            _last_result(FILTER)

    if tool == BINS:
        with st.container(border=True):
            if not numeric_cols:
                st.info("No numeric columns.")
            else:
                column = _pick("Column", numeric_cols, "bin_column")
                left, right = st.columns(2)
                bins = _setting(left.number_input, "Bins", 2, 50, value=4, name="bin_count")
                method = _setting(right.radio, "Bins of", BIN_METHODS, name="bin_method",
                                     help="Equal-width: same range each. Quantile: about the same number of rows each.")
                name = st.text_input("New column", value=f"{column}_bin", key=_key(f"bin_name_{column}"))
                if st.button("Create bins", key=_key("bin_apply")):
                    _apply({"type": "bin_numeric", "column": column, "bins": int(bins), "method": method,
                                   "name": name.strip() or f"{column}_bin"},
                                  f"Added {name}, a categorical column in bin order.", BINS)
            _last_result(BINS)

    if tool == RECODE:
        with st.container(border=True):
            st.caption("Fix typos or merge labels: give several labels the same new value.")
            if not text_cols:
                st.info("No text columns.")
            else:
                column = _pick("Column", text_cols, "recode_column")
                labels = _text_values(df[column])
                if len(labels) > MAX_RECODE_VALUES:
                    st.warning(f"{column} has {len(labels)} distinct values; only the first {MAX_RECODE_VALUES} are listed.")
                    labels = labels[:MAX_RECODE_VALUES]
                table = pd.DataFrame({"Value": labels, "New value": labels})
                edited = st.data_editor(table, disabled=["Value"], hide_index=True, key=_key(f"recode_{column}"))
                mapping = {row["Value"]: str(row["New value"]).strip() for _, row in edited.iterrows()
                           if pd.notna(row["New value"]) and str(row["New value"]).strip() != row["Value"]}
                if st.button("Apply recoding", key=_key("recode_apply"), disabled=not mapping):
                    _apply({"type": "recode_values", "column": column, "mapping": mapping},
                                  f"Recoded {len(mapping)} value{'s' if len(mapping) != 1 else ''} of {column}.", RECODE)
            _last_result(RECODE)



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
            key=_key("select_columns")
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
            key=_key("numeric_cols")
        )
        categorical_cols = st.multiselect(
            "Select Categorical Variables",
            options=[col for col in df.columns if col not in numeric_cols],
            default=st.session_state["current_categorical_cols"],
            key=_key("categorical_cols")
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

        _cleaning(df)

        st.subheader("🧪 Random Subsampling")
        sample_pct = st.slider(
            "Select percentage of data to use",
            min_value=1,
            max_value=100,
            value=100,
            key=_key("sample_pct")
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
            key=_key("melt_keys")
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
            key=_key("date_column")
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