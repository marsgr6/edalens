# Review: Bugs, Risks and Suggested Improvements

Findings from a read-only review of the code (October 2026). Nothing was run or
installed, so items marked **likely** come from reading the code and should be
confirmed by reproducing them.

Priority: 🔴 high · 🟠 medium · 🟢 low / nice-to-have

**Resolved since the review:** 1.1, 1.3, 1.6, 1.7, 5.3, 5.5, and the chained
`fillna(inplace=True)` part of 2.2 — transformations now live only in
`utils/data_processing.py`, failures are explicit and atomic, and regression
tests cover them (`tests/test_transformations.py`,
`tests/test_views_operations.py`). Line references below predate that change.

---

## 1. Correctness bugs

| # | Pri | Where | Issue | Suggested fix |
|---|---|---|---|---|
| 1.1 ✅ | 🔴 | `utils/data_processing.py:41-58`, `views/missing_data.py:122-140` | Imputation rebuilds the frame from only `float64/int64` and `object/category/bool` columns. Columns of any other dtype (datetime, `int32`, `float32`, nullable `Int64`, …) are **silently dropped**, and column order changes (numeric first). `df = df[df.columns]` is a no-op. | Impute in place: `df[num_cols] = imputer.fit_transform(df[num_cols])` using `select_dtypes("number")`, leave other columns alone. |
| 1.2 | 🔴 | `views/time_series.py:587, 635` | **Likely:** with frequency “None”, `resample_map.get("None", "D")` returns `None` (the key exists), so Prophet’s `make_future_dataframe(freq=None)` passes `freq=None` to `pd.date_range`, which raises. | `freq = resample_map.get(time_granularity) or "D"` (or infer with `pd.infer_freq`). |
| 1.3 ✅ | 🟠 | `utils/data_processing.py:59-60` | Any exception in an operation is swallowed and the partially processed frame is returned. The user sees “applied successfully” even when it wasn’t. | Re-raise, or return an error the view shows with `st.error`. |
| 1.4 | 🟠 | `views/time_series.py:1187-1197` | Single-variable Trend Analysis doesn’t drop NaN before `LinearRegression.fit`. With a rolling window or resampling gaps it will raise. (The multi-variable branch does `dropna()`.) | `y = df_summary[selected_col].dropna()`. |
| 1.5 | 🟠 | `views/time_series.py` Forecasting | Forecasting uses `df`, not `df_summary`, so the rolling-window setting is ignored there while other sections honour it. | Decide on one behaviour and document it. |
| 1.6 ✅ | 🟢 | `views/home.py:268-280` | Operation History doesn’t list `remove_nan` and `impute` steps (they show as gaps in numbering). | Add the two cases, or give each op a `label`. |
| 1.7 ✅ | 🟢 | `views/visualization.py:24, 38-39` | `data` is the session DataFrame itself (no `.copy()`), and category columns are cast to `str` in place, mutating `filtered_df` on every rerun. | `data = st.session_state["filtered_df"].copy()`. |
| 1.8 | 🟢 | `views/time_series.py` preamble | If the loaded data has no numeric columns, `selectbox` returns `None` and later code fails with an unclear error. | Check `numerical_cols` and show a friendly message. |

## 2. Compatibility / dependency risks

| # | Pri | Issue | Suggested fix |
|---|---|---|---|
| 2.1 | 🔴 | `requirements.txt` is unpinned except NumPy. A new Streamlit, pandas, Prophet or Plotly release can break the deployed app on the next push. | Pin versions (`pip freeze` from a working env) or use a lock file (`pip-tools`, `uv`). |
| 2.2 | 🟠 | Deprecated pandas APIs: `fillna(method="ffill")` (`time_series.py:256, 419, 563, 611`), resample aliases `"H"`, `"M"`, `"Q"`, `"Y"` (`time_series.py:148`), chained `categorical_df[col].fillna(..., inplace=True)` (no effect under copy-on-write). | Use `.ffill()`, `"h"`, `"ME"`, `"QE"`, `"YE"`, and assignment instead of `inplace`. |
| 2.3 | 🟠 | `use_container_width=True` is deprecated in recent Streamlit versions in favour of `width="stretch"`. | Update when Streamlit is upgraded/pinned. |
| 2.4 | 🟢 | `missingno` and `plotly-calplot` are installed but never imported. `requests` is imported but not listed (it arrives transitively via Streamlit). | Remove unused packages; add `requests` explicitly. |

## 3. Performance

| # | Pri | Issue | Suggested fix |
|---|---|---|---|
| 3.1 | 🔴 | Prophet is refit on **every** widget interaction (Decomposition, Forecasting, Changepoints). Moving the forecast-horizon slider refits the model. | Wrap fitting in `@st.cache_resource` / `@st.cache_data` keyed by data hash + parameters; fit once, predict for different horizons. |
| 3.2 | 🟠 | CSV parsing, KNN imputation, Isolation Forest and clustermap linkage are recomputed on rerun. | `@st.cache_data` around pure functions. |
| 3.3 | 🟢 | Tracking mode replays the entire operation log after each change. | Cache intermediate results per step, or snapshot after each op. |

## 4. Security & robustness

| # | Pri | Issue | Suggested fix |
|---|---|---|---|
| 4.1 | 🟠 | The “Load from GitHub” box fetches **any** URL ending in `.csv` from the server (`home.py:84`, `time_series.py:66`). On a public deployment this lets users make the server request arbitrary hosts (SSRF), including internal addresses. | Allow-list hosts (`raw.githubusercontent.com`, `github.com`), limit response size. |
| 4.2 | 🟠 | `home.py:84` calls `requests.get` with **no timeout** — a slow host hangs the session. | `timeout=10`, as `time_series.py` already does. |
| 4.3 | 🟢 | No upload size limit or row cap; large CSVs can exhaust the App Service memory. | Set `server.maxUploadSize` in `.streamlit/config.toml`; warn above N rows. |
| 4.4 | 🟢 | CSVs are decoded as UTF-8 only; Latin-1 files (common with Spanish data such as `FECHA` columns) fail. | Try `utf-8`, fall back to `latin-1`, or offer an encoding/separator selector. |

## 5. Code quality & maintainability

| # | Pri | Issue | Suggested fix |
|---|---|---|---|
| 5.1 | 🔴 | Root-level `visualization.py` (2630 lines) and `time_series.py` (identical to `views/time_series.py`) are dead code. It’s easy to edit the wrong file. | Delete them (git history keeps them) or move to `legacy/`. |
| 5.2 | 🟠 | `visualization()` is one ~1600-line function and `time_series_analysis()` one ~1200-line function with nested helpers and repeated local imports. | Split into one module per plot type / analysis section (e.g. `views/plots/boxes.py`) with a registry dict `{name: render_fn}`. |
| 5.3 ✅ | 🟠 | Every preprocessing step is implemented twice (inline in views and in `apply_operations`). The two copies can drift. | Always go through `apply_operations` (single op on current df in direct mode, full replay in tracking mode). |
| 5.4 | 🟠 | Forecasting single- vs multi-variable branches are copy-pasted (~50 lines each); same for anomaly/trend. | Loop over `selected_vars` once; single variable is just a list of one. |
| 5.5 ✅ | 🟢 | After every operation the same three lines recompute `filtered_df` / numeric / categorical columns. | Helper `set_working_df(df)`. |
| 5.6 | 🟢 | `utils/plot_utils.py` now holds the regression figure builder (first plot moved out of the view); `views/` and `utils/` have no `__init__.py`. | Remove or populate; add `__init__.py` for clarity. |
| 5.7 | 🟢 | Long `if/elif` chains for slider ranges and unit maps. | Small config dicts: `{"Daily": ("days", 7, 365, 30), …}`. |

## 6. User experience

- **Unify data between pages.** Time Series has its own loader, so cleaning done on Home isn’t available there. Let it reuse `filtered_df` (pick a date column) and keep its loader as an option.
- **Separator / decimal options** on upload (`;` and `,` decimals are common in Excel exports), plus Excel (`.xlsx`) and Parquet support.
- **Export charts** (PNG/HTML) and export the plot configuration so a chart can be reproduced.
- **Persist the session** via URL query params (`st.query_params`) so a view can be shared.
- **Imputation options:** let users pick the strategy per column (mean, median, KNN k, constant) and show before/after.
- **Language consistency:** the UI is English, while sample data uses Spanish column names. Consider i18n if the audience is Spanish-speaking.
- Use Streamlit’s native multipage (`st.navigation` / `pages/`) instead of a manual router; it gives deep links per page.

## 7. Project hygiene

| Item | Status | Recommendation |
|---|---|---|
| Tests | None | Start with `pytest` unit tests for `apply_operations` (each op type, imputation dtype preservation) and Streamlit `AppTest` smoke tests per page. |
| CI | Deploys without testing | Add a test + lint job (`ruff`) that must pass before the deploy job. |
| Duplicate deploy workflows | Two workflows deploy the same code to two Azure apps on every push | Keep one if both apps aren’t needed (halves build time and cost); otherwise one workflow with a matrix. |
| Start command | Only in the Azure portal | Commit a `startup.sh` or document it (see README) so redeploys are reproducible. |
| `.gitignore` | Missing | Add Python defaults (`__pycache__/`, `.venv/`, `.streamlit/secrets.toml`). |
| License | GPL-3.0-or-later declared in `pyproject.toml` with `LICENSE` | Confirm the copyright holder agrees to this license. |
| Commit messages | Many “Add files via upload” | Use descriptive messages; work via branches/PRs since `main` auto-deploys. |
| Python version | CI uses 3.10 (EOL Oct 2026) | Move to 3.12 once dependencies are pinned and tested. |

## Suggested order of work

1. Pin dependencies, add `.gitignore`, delete the legacy root files. *(low effort, removes risk)*
2. Fix 1.1 (imputation drops columns), 1.2 (forecast freq), 4.2 (timeout), 1.4 (trend NaN).
3. Cache Prophet fits (3.1) — the biggest speed win.
4. Add unit tests for `apply_operations` and a CI test job.
5. Refactor the two large view modules into per-plot / per-section files.
