# Architecture

EDALens is a single-process Streamlit app. There is no backend, database or API:
all data lives in the user's Streamlit session (`st.session_state`) and is lost
when the browser session ends.

## Request flow

Streamlit re-runs the whole script on every widget interaction:

```
streamlit run edalens/app.py      (or the root my_viz_plotly.py wrapper)
        │
        ▼
edalens.app.main()
  ├─ st.set_page_config(layout="wide", page_title="EDALens App")
  ├─ sidebar option_menu: Home | Missing Data Imputation | Visualization | Time Series | About
  └─ lazy-import and call the selected view:
        edalens.views.home.home()
        edalens.views.missing_data.missing_data_imputation()
        edalens.views.visualization.visualization()
        edalens.views.time_series.time_series_analysis()
        edalens.views.about.about()
```

Views are imported lazily inside the `if/elif` so that heavy libraries (Prophet,
statsmodels) are only loaded when the Time Series page is opened.

## Local launcher

`edalens/launcher.py` (the `edalens` console script, `python -m edalens`, and
`edalens.show()`) does not import the app. It starts `sys.executable -m streamlit
run <installed edalens/app.py>` as a child process bound to `127.0.0.1`, with
headless mode and a random `server.baseUrlPath` (`edalens-<token>`) passed on the
command line, so config files cannot widen the bind address. The child runs with the app's
install directory as its working directory, so modules in the caller's directory
are not imported, and logs to a temporary file.

- **Port:** an explicit port must be free on `127.0.0.1` and `0.0.0.0`;
  otherwise the first free port from 8501 is used.
- **Readiness:** success is reported only after `<prefix>/_stcore/health`
  answers `ok`. The prefix means another server on the same port cannot satisfy
  the check. If the child exits or times out, `LaunchError` includes the log tail.
- **Shutdown:** `stop()` signals only the `Popen` it created (CTRL_BREAK on
  Windows in a separate process group, SIGTERM on POSIX), kills it after 10 s,
  and removes the log. An `atexit` hook stops servers left running. The CLI
  turns Ctrl+C, SIGTERM, SIGHUP, and SIGBREAK into the same shutdown.

## Modules

| File | Lines | Responsibility |
|---|---:|---|
| `edalens/app.py` | 40 | Entry point and router (the root `my_viz_plotly.py` only calls it) |
| `edalens/launcher.py` | 325 | Local launcher (CLI and `edalens.show()`); not imported by the app |
| `edalens/views/home.py` | 257 | Load CSV (upload or GitHub URL), preprocessing operations, undo, preview, download |
| `edalens/views/missing_data.py` | 78 | Missing-value table, drop rows, KNN/mode imputation |
| `edalens/views/visualization.py` | 1597 | All general plots in one `visualization()` function with nested helpers |
| `edalens/views/time_series.py` | 1219 | Time series loader and ten analysis sections in one function |
| `edalens/views/about.py` | 256 | Static manual / marketing text |
| `edalens/utils/data_processing.py` | 316 | Single implementation of every operation (`apply_operation`), replay (`apply_operations`), and atomic session updates (`run_operation`, `undo_last_operation`, `set_working_df`) |
| `edalens/utils/plot_utils.py` | 379 | Streamlit-free figure builders: `regression_figure` (polynomial/LOWESS fits, marginal rug/histogram/box/violin) and `numeric_heatmap_figure` (per-column scaling, sorting/clustering), unit-tested in `tests/test_plot_utils.py` |

## Session state

### Shared by Home / Missing Data / Visualization

| Key | Type | Meaning |
|---|---|---|
| `raw_df` | DataFrame | Data exactly as loaded |
| `filtered_df` | DataFrame | Current working dataset used by all other pages |
| `current_numeric_cols` | list[str] | Columns treated as numeric (recomputed from dtypes after each op) |
| `current_categorical_cols` | list[str] | Columns treated as categorical (`object`/`category` dtypes) |
| `operations` | list[dict] | Operation history; always recorded, so `filtered_df == apply_operations(raw_df, operations)` |
| `enable_operation_tracking` | bool | Replay mode toggle; also shows Undo and the history |
| `data_loaded` | bool | Whether a dataset is present |
| `last_uploaded_file` | str | Filename / URL, used to detect a new upload |
| `is_imputed` | bool | Derived from `operations` (an `impute` step is present) |
| `viz_state` | dict | Remembered plot type and X/Y/Color/Facet/Size mapping |

### Time Series only

| Key | Meaning |
|---|---|
| `df` | Time-indexed DataFrame (date column parsed, sorted, set as index) |

The Time Series page **does not** read `filtered_df`; it has its own loader.

## Operation log and undo

Every Home and Missing Data action calls
`utils.data_processing.run_operation(st.session_state, operation)`:

- **Direct mode (default):** the operation is applied to the current
  `filtered_df` with `apply_operation`.
- **Tracking mode:** the full history plus the new operation is replayed from
  `raw_df` with `apply_operations`, which calls `apply_operation` per step.

Both modes use the same per-operation code and always append to `operations`,
so turning tracking on later replays the same result. “Undo Last Operation”
calls `undo_last_operation`, which replays all but the last step; undoing the
only step restores `raw_df`, and the data stays loaded.

**Atomic failures.** Operations never mutate their input. Invalid requests and
pandas/scikit-learn errors raise `OperationError`; replay names the failing
step. The new frame, history, column-type lists, and `is_imputed` are written
only after success, so a failure leaves the last valid state unchanged and the
view shows `st.error` instead of a success message.

**Read-only consumers.** Visualization and the Missing Data overview work on
`filtered_df.copy()`; only `run_operation`, `undo_last_operation`, and the
loaders write the working frame.

Supported operation types:

| `type` | Payload | Effect |
|---|---|---|
| `select_columns` | `columns` | Keep these columns in the frame's existing order; unknown or empty selection fails |
| `cast_types` | `numeric_cols`, `categorical_cols` | Numeric: existing numeric dtypes (incl. nullable `Int64`) are kept, others use `pd.to_numeric(errors="coerce")`; datetime-like columns and columns with no numeric values fail. Categorical: values become text with missing values kept missing; `category` columns are kept |
| `subsample` | `sample_pct` | Same rows as `df.sample(frac, random_state=42)`, kept in original row order with index labels; 100 % is a no-op |
| `melt` | `id_vars` | Wide → long with `variable` / `value` columns and a new RangeIndex; needs at least one non-ID column |
| `date_features` | `date_column`, `dayfirst` | Adds Year, Month, Day, DayOfWeek, Hour (`int64`, or `float64` when any date is missing); the date column becomes ISO text with unparseable values missing. Fails if nothing parses or a feature column already exists |
| `remove_nan` | — | `dropna()` keeping index, order, and dtypes; fails if it would remove every row |
| `impute` | — | See below |

**Imputation rules.** Numeric columns other than booleans are KNN features
(`KNNImputer(n_neighbors=5)`); only columns with missing values are replaced.
Integer columns that receive imputed values become float (`Int*` → `Float64`,
NumPy ints → `float64`) so fractional means are kept; float dtypes are kept.
Text, `category`, and nullable boolean columns use their most frequent value
(first after pandas' sort on ties). Datetime-like and all-missing columns are
left unchanged, and the page warns that they still contain missing values.
Column order, index, and the dtypes of unchanged columns are preserved. Empty
frames pass through unchanged.

Add new operation types only to `_HANDLERS` in `edalens/utils/data_processing.py`, with
tests for direct execution, replay, and source preservation.

## Visualization page internals

- `PLOT_TYPES` lists the 15 plot types; it shrinks to 7 when there are no numeric columns.
- A row of seven selectors (plot type, X, Y, Color, Facet col, Facet row, Size) drives everything. Options depend on plot type; the sidebar “Risk It All” checkbox unlocks any column for any role.
- `render_plot()` dispatches on `plot_type`. Most branches use `render_side_by_side(config_fn, plot_fn)` to show plot-specific settings in a left column and the chart on the right.
- “missingno” plots (matrix, bars, heatmap, dendrogram) are re-implemented in Plotly/SciPy; the `missingno` package is not actually used.
- `check_cardinality()` warns when a color/facet column has more than 50 unique values.

## Time Series page internals

1. Sidebar menu picks one of ten sections. “Load Data” parses the chosen date
   column, drops unparseable rows, sorts and sets it as the index, then stores
   `st.session_state["df"]`.
2. For every other section a shared preamble reads sidebar options (variable(s),
   resample frequency, mean/sum, rolling window) and builds `df_summary`.
3. Each section renders from `df` or `df_summary`:

| Section | Technique |
|---|---|
| Time Series Visualization | Plotly lines, optional facets |
| Decomposition & ACF/PACF | `seasonal_decompose` or Prophet components; statsmodels ACF/PACF (matplotlib) |
| Anomaly Detection | `IsolationForest(contamination)` per variable |
| Temporal Profiles | Group by hour/weekday/month/quarter/year, box or line |
| Calendar Heatmap | `plotly_calheatmap` daily / hourly |
| Heatmap Profiles | Pivot heatmaps (month×year, hour×weekday, …) |
| Forecasting | Prophet with yearly/weekly/daily seasonality |
| Changepoint Analysis | Prophet changepoints, auto (percentile) or fixed N |
| Trend Analysis | Linear and degree-2 polynomial regression on the index |

## Deployment

`.github/workflows/validate.yml` runs the validation gates on every push and pull
request; there are no deploy workflows. `scripts/build_deployment.py` builds the
allowlisted `dist/release.zip` for hosting (e.g. Azure App Service), where the
Streamlit start command is configured on the host.
