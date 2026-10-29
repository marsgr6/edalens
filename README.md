# EDALens — Interactive Exploratory Data Analysis with Streamlit + Plotly

EDALens is a browser-based tool for **exploring a CSV dataset without writing code**.
You upload a file (or paste a GitHub link), clean and reshape it, inspect and fill
missing values, then build interactive Plotly charts — from simple bar charts to
pair plots, cluster maps, ridge plots and regressions with confidence bands.
A separate **Time Series** module adds decomposition, anomaly detection,
Prophet forecasting, changepoint detection, calendar heatmaps and trend analysis.

It is aimed at EDA, data-quality checks, teaching statistics/data science, and
quick visualization prototyping.

```bash
pip install edalens
```

```python
import edalens

edalens.walk(df)                    # the app inside your notebook, df preloaded
edalens.walk(df, mode="browser")    # same, with no server at all (Colab, JupyterHub)
edalens.show()                      # the app in its own browser tab
```

> **Status:** under active development. Some combinations (heavy faceting, very
> high-cardinality categories) may behave unexpectedly — reload the page and
> re-upload the data if the app gets into a bad state.

---

## Features at a glance

| Section (sidebar) | What you can do |
|---|---|
| **Home** | Upload a CSV or load one from GitHub · keep/drop columns · set numeric vs categorical types · random subsample · melt (wide → long) · extract date features (Year, Month, Day, DayOfWeek, Hour) · optional undo history · download the processed CSV |
| **Missing Data Imputation** | Per-column missing counts/percentages · drop rows with any NaN · impute (KNN for numeric, mode for categorical) |
| **Visualization** | 15 plot types: bars, boxes (box/violin), lineplot, histogram, density 1 (1-D KDE), density 2 (2-D hist/KDE), scatter (2-D or 3-D), catplot (strip/swarm), missingno-style plots, correlation, clustermap, pairplot, regression (poly order 1–3 + CI and/or LOWESS, marginal rug/histogram/box/violin), heatmap (numeric columns with scaling/sorting/clustering), ridges. Shared X / Y / Color / Facet-col / Facet-row / Size mapping, custom category order, high-cardinality warnings |
| **Time Series** | Own data loader (date column + day-first/month-first) · resampling (hourly → yearly, mean/sum) and rolling smoothing · multi-variable mode · decomposition (statsmodels or Prophet) + ACF/PACF · Isolation-Forest anomaly detection · temporal profiles · calendar heatmaps · profile heatmaps · Prophet forecasting · Prophet changepoint analysis · linear/polynomial trend |
| **About** | Built-in user manual with screenshots and example datasets |

See [docs/FEATURES.md](docs/FEATURES.md) for a full walkthrough.

---

## Quick start

Requirements: **Python 3.10 or newer** (tested on 3.10, 3.12 and 3.13; the
pinned lock files and CI use 3.10). Prophet and scikit-learn are fairly heavy;
a virtual environment is strongly recommended.

```bash
git clone https://github.com/marsgr6/edalens.git
cd edalens

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
streamlit run my_viz_plotly.py
```

Streamlit opens the app at <http://localhost:8501>.

### Install and launch as a package

EDALens installs as the `edalens` package. The launcher runs the installed app
on `127.0.0.1` only, so it works from any directory:

```bash
pip install edalens                                         # from PyPI
# or, from a checkout with the exact pinned dependencies:
pip install --require-hashes -r requirements-runtime.lock
pip install --no-deps .
```

**Terminal** — starts the server, prints the URL, opens a browser tab, and
blocks until you press Ctrl+C:

```bash
edalens                      # first free port from 8501
edalens --port 8600          # fail instead of switching if 8600 is taken
edalens --no-browser         # only print the URL
python -m edalens --help     # same launcher without the console script
```

**Python / Jupyter** — `show()` returns once the server answers its health
check, so the notebook stays usable while the app runs:

```python
import edalens

app = edalens.show()              # open_browser=False to only get the URL
app.url                          # e.g. http://127.0.0.1:8501/edalens-3f9c.../
app.stop()                       # stops only this server; safe to call twice

with edalens.show(open_browser=False) as app:   # stopped when the block ends
    print(app.url)
```

Use the printed URL as-is: each server gets a random path prefix so the
launcher can tell its own server from anything else on the port. In a plain
script, call `app.wait()` to keep the server running; servers still running
when Python exits are stopped automatically. Startup failures and timeouts
raise `edalens.LaunchError` with the end of the server log.

**Inside a Jupyter notebook** (like pygwalker) — `walk()` starts the app with
your DataFrame already loaded and shows it in the cell output:

```python
import edalens

app = edalens.walk(df)            # df (or a Series) is preloaded; walk() alone opens the upload page
app = edalens.walk(df, height=1100)
app.display()                    # show the same app again in another cell
app.stop()                       # stops the server and deletes the temporary data file
```

The app is embedded with an iframe pointing at `127.0.0.1`, so it works in
local Jupyter Notebook, JupyterLab and VS Code. In Google Colab, `walk()`
embeds the app through Colab's own port proxy instead (and installs the
requirements once, like `run_colab()`); there Streamlit's CORS and XSRF checks
are turned off, because the proxied iframe is cross-site, and the server still
listens on the VM's loopback only. If the frame stays blank or stuck on
*Connecting…*, use `run_colab()` below. It does not work on JupyterHub or other
remote kernels. The DataFrame is pickled to a private temporary file, and
every browser session of that app starts from it; "Reset All" or a new upload
replaces it for that session. Each `walk()` call starts its own server.

**Inside a notebook with no server at all** (exactly like pygwalker) —
`mode="browser"` runs the same app in the notebook's browser with
[stlite](https://github.com/whitphx/stlite) (Streamlit on WebAssembly). Nothing
listens on a port and nothing is tunnelled, so it works in Colab, on
JupyterHub and remote kernels as well as locally:

```python
app = edalens.walk(df, mode="browser")
app.save("edalens.html")          # optional: a standalone page that opens in any browser
```

The app's code and the DataFrame are embedded in the cell output; the browser
downloads Pyodide and the packages from jsDelivr on first use (20–60 s, then
cached). Differences from the server mode: Prophet cannot run in WebAssembly,
so forecasting is unavailable; the sidebar uses a plain radio menu (stlite
cannot load custom components); the "Download Processed Data" button does not
work; and a large DataFrame makes the output, and a saved notebook, large
(sample it first). Changes made in the app do not flow back to the kernel.

**Google Colab**: open
[`notebooks/colab_edalens.ipynb`](notebooks/colab_edalens.ipynb) in Colab, or run:

```python
!git clone -q --depth 1 https://github.com/marsgr6/edalens.git /content/edalens
%cd /content/edalens

import edalens
app = edalens.run_colab()   # installs requirements once, starts the app, opens a localtunnel link
app                        # shows the link and the tunnel password
app.stop()                 # closes the link and stops the app
```

Colab can't open `127.0.0.1` in your browser, so `run_colab()` exposes the app
through [localtunnel](https://github.com/localtunnel/localtunnel) (Node.js
is preinstalled in Colab). On the first visit localtunnel asks for a *tunnel
password*: it's the runtime's public IP, which `run_colab()` prints. Anyone
with the link and password can use the app while it runs.

### Try it with sample data

Example CSVs live in
[marsgr6/r-scripts › data/viz_data](https://github.com/marsgr6/r-scripts/tree/master/data/viz_data).
Paste a link into **Home → Load from GitHub**, for example:

- `Tips.csv` — bars, boxes, histograms, densities, scatter, catplot
- `Penguins.csv` — missing data, clustermap, pairplot, regression
- `ts_data_precipitation.csv` — time-series bars/lines, ridges, heatmaps, and the Time Series module

### CSV requirements

- Comma-separated, `.` as decimal separator, no thousands separators.
- These strings are read as missing: `'' NA N/A na n/a NaN nan NULL null missing - --`.
- Excel often rewrites CSVs (`;` separators, locale decimals) — re-export as plain CSV if columns look wrong.

---

## Typical workflow

1. **Home** — load data, choose columns, fix variable types, optionally subsample / melt / derive date features. Check the preview after each step and download the result if you want to keep it.
2. **Missing Data Imputation** — drop or impute missing values (optional).
3. **Visualization** — pick a plot type and map variables to X / Y / Color / Facets / Size.
4. **Time Series** — load a dataset with a date column (this module has its *own* loader) and pick an analysis section.

The app recommends: *load → apply one operation → check preview → download → repeat.*

---

## Project structure

```
edalens/                  The pip package
  app.py                  Streamlit entry point: page config + sidebar router
  launcher.py             Local launcher: `edalens` command and edalens.show()
  notebook.py             edalens.walk(df): the app inside a notebook cell
  browser.py              edalens.walk(df, mode="browser"): the app in the browser via stlite
  colab.py                Google Colab: edalens.run_colab() (installs requirements, opens a localtunnel link)
  views/
    home.py               Data upload & preprocessing
    missing_data.py       Missing-value overview, drop / impute
    visualization.py      All general-purpose plots (~1.6k lines)
    time_series.py        Time series module (~1.2k lines)
    about.py              In-app manual and examples
  utils/
    data_processing.py    Shared transformations, replay, and atomic undo/history updates
    plot_utils.py         Streamlit-free figure builders (regression, numeric heatmap)
my_viz_plotly.py          Thin wrapper for `streamlit run` and Azure; runs edalens.app
notebooks/                Colab and test notebooks
.github/workflows/        Validation and two Azure Web App deploy pipelines
requirements.txt
```

Architecture details (state flow, session keys, operation log): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Deployment

Every push to `main` triggers **two** GitHub Actions workflows, each building the
app and deploying it to a different Azure App Service:

| Workflow | Azure app |
|---|---|
| `main_effective-visualization-streamlit-plotly.yml` | `effective-visualization-streamlit-plotly` |
| `main_myviz-efective-vizualization-with-streamlit.yml` | `myviz-efective-vizualization-with-streamlit` |

Both use a publish-profile secret (`AZUREAPPSERVICE_PUBLISHPROFILE_*`). The
Streamlit start command is not in the repo; it must be set in the Azure portal,
typically:

```bash
python -m streamlit run my_viz_plotly.py --server.port 8000 --server.address 0.0.0.0
```

---

## Tech stack

Streamlit · streamlit-option-menu · pandas · NumPy (1.26.4) · Plotly · SciPy ·
scikit-learn (KNN imputation, Isolation Forest, trend fits) · statsmodels
(decomposition, ACF/PACF) · Prophet (forecasting, changepoints) ·
plotly-calheatmap · matplotlib (ACF/PACF and statsmodels decomposition).

---

## Known limitations & roadmap

The main points (details and suggested fixes in
[docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md)):

- Time Series uses a separate data loader; data prepared on Home is not reused there.
- Prophet models are refit on every interaction (no caching), so forecasting/changepoints are slow.
- Forecasting with frequency “None” likely fails (`freq=None` passed to Prophet).
- Unpinned dependencies and deprecated pandas APIs (`fillna(method=...)`, `"H"/"M"/"Q"/"Y"` aliases).
- Two duplicate legacy files remain at the root; they are excluded from the installed package.

---

## Contributing

Issues and PRs are welcome. With Python 3.10, install the validation dependencies
and run the checks before submitting:

```bash
python -m pip install --require-hashes -r requirements-validation.lock
python scripts/validate.py
```

See [docs/VALIDATION.md](docs/VALIDATION.md) for details. Also run the app locally
and exercise the pages you touched with one of the sample datasets above.
