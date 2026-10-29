# Feature Guide

A page-by-page guide to what EDAViz can do. The in-app **About** page has the
same manual with screenshots.

## 1. Home — load and prepare data

**Loading**
- *Upload CSV* from your computer, or
- *Load from GitHub*: paste a `.csv` link. `github.com/.../blob/...` links are
  converted to `raw.githubusercontent.com` automatically.

**Operation tracking (undo).** Every step is recorded. The checkbox *Enable
Operation Tracking for Undo* shows the history and an *Undo Last Operation*
button, and rebuilds the data by replaying every step from the raw data after
each change, so it is slower on large files. Undoing the only step returns to
the original data. *Reset All* clears the session.

**Errors.** If a step cannot be applied (for example a melt with no value
columns, a date column where nothing parses, or removing rows would leave no
data), the page shows an error and keeps the previous data and history.

**Preprocessing steps** (each has its own *Apply* button):

| Step | What it does |
|---|---|
| Column Selection | Keep only the chosen columns, in their current order |
| Assign Variable Types | Mark columns numeric (non-numbers become NaN) or categorical (cast to text; missing values stay missing) |
| Random Subsampling | Keep 1–100 % of rows (fixed seed 42, reproducible, original row order kept) |
| Melt Data | Reshape wide → long; chosen ID columns stay, the rest become `variable` / `value` |
| Date Feature Engineering | Parse a date column (day-first optional) and add `Year`, `Month`, `Day`, `DayOfWeek`, `Hour`; unparseable dates become missing |

Then *Preview* the result and *Download Processed Data as CSV*.

## 2. Missing Data Imputation

- Table of missing count and % per column.
- **Remove Rows with Any Missing Values.** Refused if every row has a missing value.
- **Impute:** numeric columns with `KNNImputer(n_neighbors=5)`, text, categorical
  and boolean columns with the most frequent value. All columns are kept in
  their order. Integer columns that receive imputed values become decimal
  columns. Date and completely empty columns are not imputed; the page lists
  any columns that still have missing values.

## 3. Visualization

Choose a plot type, then map columns to **X, Y, Color, Facet Col, Facet Row, Size**.
By default only sensible column types are offered; tick **Risk It All** in the
sidebar to allow any column in any role.

| Plot | Use it for | Notable options |
|---|---|---|
| bars | Counts or values per category; raw time series as bars | Histogram vs raw-value mode, bars or heatmap |
| boxes | Distribution per group | Box or violin, swarm overlay, CI bands (68/95/99 %) |
| lineplot | Trends across an ordered X | Mean/Sum aggregation, CI band, line style/width by column, “time is here” raw mode |
| histogram | One numeric distribution | layer/dodge/stack, count/probability/percent/density, bars/step |
| density 1 | Smoothed 1-D distribution (KDE) | layer/stack, cumulative, normalize |
| density 2 | Joint distribution of two numerics | 2-D histogram or KDE contours |
| scatter | Two numerics, or three in 3D | Symbol by category, size by numeric (min-max enhance). **3D Scatter** adds a Z variable (rotatable; colour, symbol, size and opacity still apply; facets are not available in 3D) with optional projections onto the walls |
| catplot | Numeric vs category, every point | strip or swarm |
| missingno | Missing-data structure | matrix, bars, heatmap (nullity correlation), dendrogram |
| correlation | Correlation matrix of numerics | — |
| clustermap | Correlation matrix reordered by hierarchical clustering | z-score / standard scale |
| pairplot | Scatter matrix with KDE diagonals | — |
| regression | Scatter + fitted line(s) | Fit methods: Polynomial (order 1–3, CI 68/90/95/99 %) and/or LOWESS (adjustable smoothing fraction, no CI; dashed when shown with the polynomial). Optional marginal plots: rug, histogram, box or violin, one per hue group (box/violin panels grow with the number of groups and show outliers as jittered points); the Y marginal defaults to the X choice and can be changed or turned off |
| heatmap | Numeric columns × observations (X = None), or category × numeric | Colormap choice. Numeric mode: pick columns (non-numeric ones are ignored), scale each column (None / Z-score / Min-max; hover shows original values), order observations (original, sort by a column, or hierarchical clustering up to 5,000 rows), cluster columns, variables on the Y or X axis |
| ridges | Stacked densities by category or column | Falls back to numeric ridges with > 100 categories |

Many plots support **custom category order** for X, color and facets — useful for
months, weekdays or ordinal scales. A warning appears when a color/facet column has
more than 50 categories.

## 4. Time Series

This page has its **own loader** (sidebar → *Load Data*): upload or GitHub link,
choose the date column and date format (DD/MM vs MM/DD). Rows with unparseable
dates are dropped; data is sorted and indexed by date.

**Sidebar options used by all sections**
- Variable to analyse, or *Use multiple variables* for multi-series views.
- Summarization frequency: None, Hourly, Daily, Weekly, Monthly, Quarterly, Yearly — with Mean or Sum.
- Rolling window (0 = no smoothing).

| Section | What you get |
|---|---|
| Time Series Visualization | Line chart(s); facets with optional shared Y axis |
| Decomposition & ACF/PACF | Trend/seasonal/residual (statsmodels, adjustable period) or Prophet components; ACF and PACF plots |
| Anomaly Detection | Isolation Forest with adjustable contamination; anomalies in red |
| Temporal Profiles | Box or line plot by hour, weekday, month, quarter or year |
| Calendar Heatmap | GitHub-style daily calendar, or hourly month grid |
| Heatmap Profiles | Month×year, hour, weekday, week-of-year heatmaps with mean/median/sum/min/max/std |
| Forecasting | Prophet forecast with uncertainty band; horizon slider adapts to the frequency |
| Changepoint Analysis | Prophet changepoints, selected by importance percentile (Auto) or a fixed count |
| Trend Analysis | Linear and quadratic trend lines over the observed series |

## 5. Example datasets

From [marsgr6/r-scripts › data/viz_data](https://github.com/marsgr6/r-scripts/tree/master/data/viz_data):

| Dataset | Good for |
|---|---|
| `Tips.csv` | bars, boxes, ridges, histograms, densities, scatter, catplot |
| `Penguins.csv` | missingno, clustermap, pairplot, regression |
| `ts_data_precipitation.csv` | time-series bars/lines (keep `FECHA`), numeric ridges and heatmaps, the Time Series page |
