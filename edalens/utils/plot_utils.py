"""Figure builders that are independent of Streamlit, so they can be unit-tested."""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from plotly.colors import hex_to_rgb
from plotly.subplots import make_subplots
from statsmodels.nonparametric.smoothers_lowess import lowess

from edalens.utils.data_processing import parse_dates

FIT_METHODS = ("Polynomial", "LOWESS")
MARGINAL_KINDS = ("rug", "histogram", "box", "violin")
DEFAULT_PALETTE = px.colors.qualitative.Set2
# Pixel budget for the regression figure; box/violin marginals get room per group
# so their shape stays readable instead of being squeezed into a thin strip.
MAIN_PLOT_PX = 480
THIN_MARGINAL_PX = 120
DISTRIBUTION_MARGINAL_PX_PER_GROUP = 130
DISTRIBUTION_MARGINAL_PX = (200, 450)
RIGHT_MARGINAL_MAX_PX = 260  # of the 800 px design width, so the main plot stays dominant
FIGURE_CHROME_PX = 140
# Violin thickness in category units (1 = the whole slot of one group). Plotly's
# automatic width draws the top (horizontal) violins as thin slivers here.
MARGINAL_VIOLIN_WIDTH = 0.7


def _rgba(color: str, alpha: float) -> str:
    """Return ``color`` (hex or ``rgb(...)``) as an ``rgba(...)`` string."""
    if color.startswith("#"):
        r, g, b = hex_to_rgb(color)
    else:
        r, g, b = (int(float(v)) for v in color[color.index("(") + 1 : color.index(")")].split(",")[:3])
    return f"rgba({r},{g},{b},{alpha})"


def polynomial_fit(
    x: np.ndarray, y: np.ndarray, order: int, ci_level: float, n_points: int = 100
) -> Optional[pd.DataFrame]:
    """OLS polynomial fit with a confidence band for the mean.

    Returns columns ``x``, ``fit``, ``lower`` and ``upper``, or ``None`` when
    there are not enough distinct x values to fit the requested order.
    """
    import statsmodels.api as sm

    if np.unique(x).size <= order:
        return None
    design = np.vander(x, N=order + 1, increasing=True)
    model = sm.OLS(y, design).fit()
    x_pred = np.linspace(x.min(), x.max(), n_points)
    summary = model.get_prediction(np.vander(x_pred, N=order + 1, increasing=True)).summary_frame(
        alpha=1 - ci_level / 100
    )
    return pd.DataFrame(
        {
            "x": x_pred,
            "fit": summary["mean"].to_numpy(),
            "lower": summary["mean_ci_lower"].to_numpy(),
            "upper": summary["mean_ci_upper"].to_numpy(),
        }
    )


def lowess_fit(x: np.ndarray, y: np.ndarray, frac: float) -> Optional[pd.DataFrame]:
    """LOWESS smooth sorted by x; ``None`` when there are too few points."""
    if not 0 < frac <= 1:
        raise ValueError("LOWESS fraction must be greater than 0 and at most 1.")
    if x.size < 3:
        return None
    smoothed = lowess(y, x, frac=frac, return_sorted=True)
    return pd.DataFrame({"x": smoothed[:, 0], "fit": smoothed[:, 1]})


# Outliers (beyond 1.5 IQR) drawn as jittered points over the box/violin centre.
_OUTLIER_POINTS = dict(boxpoints="outliers", jitter=0.4, pointpos=0, marker_size=5)


def _marginal_size(kind: Optional[str], n_groups: int) -> int:
    """Pixel size of a marginal panel (height on top, width on the right)."""
    if kind in ("box", "violin"):
        low, high = DISTRIBUTION_MARGINAL_PX
        return int(np.clip(DISTRIBUTION_MARGINAL_PX_PER_GROUP * n_groups, low, high))
    return THIN_MARGINAL_PX


def _marginal_traces(
    kind: str, values: pd.Series, axis: str, label: str, color: str, group: str
) -> list[BaseTraceType]:
    """Traces for one group's marginal distribution along ``axis`` ('x' or 'y')."""
    common = dict(name=label, legendgroup=group, showlegend=False, marker_color=color)
    other = "y" if axis == "x" else "x"
    if kind == "histogram":
        return [go.Histogram(**{axis: values, f"nbins{axis}": 30}, opacity=0.5, **common)]
    if kind == "box":
        return [
            go.Box(**{axis: values}, orientation="h" if axis == "x" else "v", **_OUTLIER_POINTS, **common)
        ]
    if kind == "violin":
        # "hard" span keeps the density inside the data range so axes are not stretched.
        return [
            go.Violin(
                **{axis: values}, orientation="h" if axis == "x" else "v", box_visible=True,
                width=MARGINAL_VIOLIN_WIDTH, spanmode="hard", points=_OUTLIER_POINTS["boxpoints"],
                **{k: v for k, v in _OUTLIER_POINTS.items() if k != "boxpoints"}, **common,
            )
        ]
    if kind == "rug":
        symbol = "line-ns-open" if axis == "x" else "line-ew-open"
        return [
            go.Scatter(
                **{axis: values, other: [label] * len(values)},
                mode="markers",
                marker=dict(symbol=symbol, size=10, color=color, line=dict(width=1, color=color)),
                hoverinfo=axis,
                **{k: v for k, v in common.items() if k != "marker_color"},
            )
        ]
    raise ValueError(f"Unknown marginal kind: {kind!r}. Use one of {', '.join(MARGINAL_KINDS)}.")


def regression_figure(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    methods: Sequence[str] = ("Polynomial",),
    order: int = 1,
    ci_level: float = 95,
    lowess_frac: float = 2 / 3,
    marginal_x: Optional[str] = None,
    marginal_y: Optional[str] = None,
    palette: Sequence[str] = DEFAULT_PALETTE,
    point_size: int = 5,
) -> go.Figure:
    """Scatter of ``y`` vs ``x`` with polynomial and/or LOWESS fits per hue group.

    ``methods`` may contain "Polynomial" (OLS, with a ``ci_level`` % band) and
    "LOWESS" (``lowess_frac`` of the points per local fit, no band). Optional
    ``marginal_x``/``marginal_y`` add a rug, histogram, box or violin per group.
    """
    for column in (x, y):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' is not in the data.")
        if not pd.api.types.is_numeric_dtype(data[column]) or pd.api.types.is_bool_dtype(data[column]):
            raise ValueError(f"Regression needs numeric X and Y; '{column}' is not numeric.")
    unknown = [m for m in methods if m not in FIT_METHODS]
    if unknown:
        raise ValueError(f"Unknown fit method(s): {', '.join(unknown)}.")
    for kind in (marginal_x, marginal_y):
        if kind is not None and kind not in MARGINAL_KINDS:
            raise ValueError(f"Unknown marginal kind: {kind!r}. Use one of {', '.join(MARGINAL_KINDS)}.")

    use_hue = hue is not None and hue in data.columns and hue not in (x, y)
    groups: list[Any] = list(pd.unique(data[hue].dropna())) if use_hue else [None]
    both = len(methods) > 1

    rows = 2 if marginal_x else 1
    cols = 2 if marginal_y else 1
    top_px = _marginal_size(marginal_x, len(groups)) if marginal_x else 0
    right_px = min(_marginal_size(marginal_y, len(groups)), RIGHT_MARGINAL_MAX_PX) if marginal_y else 0
    if rows == 1 and cols == 1:
        fig = go.Figure()
        cell = {}
    else:
        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=True,
            shared_yaxes=True,
            # Relative to the 800 px design width and the main plot height.
            column_widths=[800 - right_px, right_px] if cols == 2 else None,
            row_heights=[top_px, MAIN_PLOT_PX] if rows == 2 else None,
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )
        cell = dict(row=rows, col=1)

    for idx, group in enumerate(groups):
        subset = data[data[hue] == group] if group is not None else data
        pairs = subset[[x, y]].dropna()
        if pairs.empty:
            continue
        xs = pairs[x].to_numpy(dtype=float)
        ys = pairs[y].to_numpy(dtype=float)
        color = palette[idx % len(palette)]
        label = str(group) if group is not None else "Data"
        legend = dict(legendgroup=label)

        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=color, size=point_size), name=label, **legend),
            **cell,
        )

        if "Polynomial" in methods:
            fit = polynomial_fit(xs, ys, order, ci_level)
            if fit is not None:
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([fit["x"], fit["x"][::-1]]),
                        y=np.concatenate([fit["upper"], fit["lower"][::-1]]),
                        fill="toself",
                        fillcolor=_rgba(color, 0.2),
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                        **legend,
                    ),
                    **cell,
                )
                fig.add_trace(
                    go.Scatter(
                        x=fit["x"], y=fit["fit"], mode="lines", line=dict(color=color, width=2),
                        name=f"{'Poly' if both else 'Fit'} (order {order}): {label}", **legend,
                    ),
                    **cell,
                )

        if "LOWESS" in methods:
            fit = lowess_fit(xs, ys, lowess_frac)
            if fit is not None:
                fig.add_trace(
                    go.Scatter(
                        x=fit["x"], y=fit["fit"], mode="lines",
                        line=dict(color=color, width=2, dash="dash" if both else "solid"),
                        name=f"LOWESS (frac {lowess_frac:.2f}): {label}", **legend,
                    ),
                    **cell,
                )

        if marginal_x:
            for trace in _marginal_traces(marginal_x, pairs[x], "x", label, color, label):
                fig.add_trace(trace, row=1, col=1)
        if marginal_y:
            for trace in _marginal_traces(marginal_y, pairs[y], "y", label, color, label):
                fig.add_trace(trace, row=rows, col=2)

    title = "Regression Plot"
    if "Polynomial" in methods:
        title += f" with {ci_level:g}% Confidence Interval"
    fig.update_layout(title=title, width=800, height=MAIN_PLOT_PX + top_px + FIGURE_CHROME_PX, barmode="overlay")
    fig.update_xaxes(title_text=x, **cell)
    fig.update_yaxes(title_text=y, **cell)
    if marginal_x:
        fig.update_yaxes(showticklabels=marginal_x in ("box", "violin", "rug"), row=1, col=1)
    if marginal_y:
        fig.update_xaxes(showticklabels=marginal_y in ("box", "violin", "rug"), row=rows, col=2)
    return fig


HEATMAP_SCALINGS = ("None", "Z-score", "Min-max")
HEATMAP_ROW_ORDERS = ("Original", "Sort by column", "Cluster")
# Hierarchical clustering needs O(n^2) memory; beyond this, ask for a subsample.
MAX_CLUSTER_ROWS = 5000


def scale_columns(frame: pd.DataFrame, scaling: str) -> pd.DataFrame:
    """Scale each column independently so variables on different units are comparable.

    Constant columns become 0 (z-score) or 0.5 (min-max) instead of NaN; missing
    values stay missing.
    """
    frame = frame.astype(float)
    if scaling == "None":
        return frame
    if scaling == "Z-score":
        center, spread, constant_value = frame.mean(), frame.std(ddof=0), 0.0
    elif scaling == "Min-max":
        center, spread, constant_value = frame.min(), frame.max() - frame.min(), 0.5
    else:
        raise ValueError(f"Unknown scaling: {scaling!r}. Use one of {', '.join(HEATMAP_SCALINGS)}.")
    scaled = (frame - center) / spread.replace(0, np.nan)
    constant = spread == 0
    scaled.loc[:, constant] = constant_value
    return scaled.where(frame.notna())


def _cluster_order(matrix: pd.DataFrame) -> list[int]:
    """Leaf order of average-linkage clustering of the rows (NaN filled with column means)."""
    from scipy.cluster import hierarchy

    if len(matrix) < 3:
        return list(range(len(matrix)))
    filled = matrix.fillna(matrix.mean()).fillna(0).to_numpy()
    return list(hierarchy.leaves_list(hierarchy.linkage(filled, method="average", metric="euclidean")))


def numeric_heatmap_figure(
    data: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    scaling: str = "Z-score",
    row_order: str = "Original",
    sort_by: Optional[str] = None,
    descending: bool = False,
    cluster_columns: bool = False,
    variables_on: str = "y",
    colorscale: str = "Viridis",
) -> go.Figure:
    """Heatmap of observations x numeric variables.

    Non-numeric columns are ignored. ``scaling`` is applied per column; the hover
    always shows the original value. ``row_order`` keeps the data order, sorts
    by ``sort_by``, or clusters similar observations together; ``cluster_columns``
    also groups similar variables. ``variables_on`` puts variables on the "y"
    (default) or "x" axis. Observations are labelled with the data's index.
    """
    numeric = [
        c for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_bool_dtype(data[c])
    ]
    if columns is not None:
        not_numeric = [c for c in columns if c not in numeric]
        if not_numeric:
            raise ValueError(f"Heatmap columns must be numeric: {', '.join(map(str, not_numeric))}.")
        numeric = [c for c in numeric if c in set(columns)]
    if not numeric:
        raise ValueError("The heatmap needs at least one numeric column.")
    if row_order not in HEATMAP_ROW_ORDERS:
        raise ValueError(f"Unknown row order: {row_order!r}. Use one of {', '.join(HEATMAP_ROW_ORDERS)}.")
    if variables_on not in ("x", "y"):
        raise ValueError("variables_on must be 'x' or 'y'.")

    raw = data[numeric].astype(float)
    if row_order == "Sort by column":
        if sort_by not in data.columns:
            raise ValueError("Choose a column to sort the observations by.")
        raw = raw.loc[data[sort_by].sort_values(ascending=not descending, na_position="last", kind="stable").index]
    scaled = scale_columns(raw, scaling)
    if row_order == "Cluster":
        if len(scaled) > MAX_CLUSTER_ROWS:
            raise ValueError(
                f"Clustering is limited to {MAX_CLUSTER_ROWS} observations ({len(scaled)} given); "
                "subsample the data in Home or use another row order."
            )
        order = _cluster_order(scaled)
        raw, scaled = raw.iloc[order], scaled.iloc[order]
    if cluster_columns and len(numeric) > 2:
        order = _cluster_order(scaled.T)
        raw, scaled = raw.iloc[:, order], scaled.iloc[:, order]

    observations = [str(i) for i in raw.index]
    variables = [str(c) for c in raw.columns]
    value_label = "Value" if scaling == "None" else f"{scaling}"
    if variables_on == "y":
        z, custom, x_labels, y_labels = scaled.T.to_numpy(), raw.T.to_numpy(), observations, variables
        x_title, y_title = "Observation (index)", "Variable"
        hover = "Observation: %{x}<br>Variable: %{y}"
    else:
        z, custom, x_labels, y_labels = scaled.to_numpy(), raw.to_numpy(), variables, observations
        x_title, y_title = "Variable", "Observation (index)"
        hover = "Observation: %{y}<br>Variable: %{x}"
    hover += "<br>Value: %{customdata:.4g}"
    if scaling != "None":
        hover += f"<br>{scaling}: " + "%{z:.3f}"

    fig = go.Figure(
        go.Heatmap(
            z=z, x=x_labels, y=y_labels, customdata=custom, colorscale=colorscale,
            hovertemplate=hover + "<extra></extra>", hoverongaps=False,
            colorbar=dict(title=value_label),
            # Centre z-scores so diverging scales put the column mean at the neutral colour.
            zmid=0 if scaling == "Z-score" else None,
        )
    )
    n_rows = len(y_labels)
    title = f"Heatmap of {len(variables)} numeric variables x {len(observations)} observations"
    if scaling != "None":
        title += f" ({scaling} per variable)"
    fig.update_layout(
        title=title, width=800, height=int(np.clip(n_rows * 22 + 200, 450, 1400)),
        xaxis=dict(title=x_title, type="category"),
        yaxis=dict(title=y_title, type="category", autorange="reversed", tickfont=dict(size=10)),
        margin=dict(l=50, r=50, t=100, b=50),
    )
    # Too many observation labels are unreadable; let Plotly thin them out.
    obs_axis = "xaxis" if variables_on == "y" else "yaxis"
    if len(observations) > 60:
        fig.update_layout({obs_axis: dict(nticks=30)})
    return fig


CANDLE_SOURCES = ("One value column", "Open/High/Low/Close columns")
INCREASING_COLOR = "#2ca02c"
DECREASING_COLOR = "#d62728"
TOTAL_COLOR = "#1f77b4"


def ordered_steps(values: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Sort keys for the X steps of a candlestick or waterfall.

    Categories keep their (custom) order, numbers and dates sort naturally, and
    text that all parses as dates sorts by date; other text keeps its order of
    first appearance. Returns the X values to plot (dates parsed), uniquely, in order.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        present = set(values.dropna())
        return pd.Series([c for c in values.cat.categories if c in present], dtype=object)
    values = values.dropna()
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_datetime64_any_dtype(values):
        return pd.Series(np.sort(values.unique()))
    dates = parse_dates(values, dayfirst)
    if len(values) and dates.notna().all():
        return pd.Series(np.sort(dates.unique()))
    return pd.Series(values.unique(), dtype=object)


def _step_keys(frame: pd.DataFrame, x: str, dayfirst: bool) -> tuple[pd.Series, pd.Series]:
    """Per-row step key (dates parsed when the steps are dates) and the ordered steps."""
    steps = ordered_steps(frame[x], dayfirst)
    keys = frame[x]
    if pd.api.types.is_datetime64_any_dtype(steps) and not pd.api.types.is_datetime64_any_dtype(keys):
        keys = parse_dates(keys, dayfirst)
    return keys, steps


def ohlc_frame(
    frame: pd.DataFrame,
    x: str,
    value: Optional[str] = None,
    ohlc: Optional[Sequence[str]] = None,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Open/high/low/close per X step, in step order.

    With ``value``, each step's rows (in their original order) give
    open = first, high = max, low = min, close = last. With ``ohlc`` (four column
    names), the columns are used as they are; several rows per step are combined
    the same way (first open, highest high, lowest low, last close).
    """
    if (value is None) == (ohlc is None):
        raise ValueError("Pass either value or ohlc.")
    keys, steps = _step_keys(frame, x, dayfirst)
    if value is not None:
        columns = dict(open=value, high=value, low=value, close=value)
    else:
        columns = dict(zip(("open", "high", "low", "close"), ohlc))
    data = pd.DataFrame({name: frame[col] for name, col in columns.items()})
    data["step"] = keys.to_numpy()
    data = data.dropna()
    grouped = data.groupby("step", sort=False, observed=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    return grouped.reindex(steps).dropna().rename_axis(x)


def candlestick_figure(
    ohlc: pd.DataFrame, title: str, y_label: str, range_slider: bool = False, hide_weekends: bool = False
) -> go.Figure:
    """Candlesticks from ``ohlc_frame`` output (index: steps).

    ``hide_weekends`` removes Saturday and Sunday from a date axis, so trading
    days sit next to each other instead of around two-day gaps.
    """
    fig = go.Figure(
        go.Candlestick(
            x=ohlc.index, open=ohlc["open"], high=ohlc["high"], low=ohlc["low"], close=ohlc["close"],
            increasing_line_color=INCREASING_COLOR, decreasing_line_color=DECREASING_COLOR,
            name=y_label,
        )
    )
    fig.update_layout(
        title=title, xaxis_title=ohlc.index.name, yaxis_title=y_label, height=550,
        xaxis_rangeslider_visible=range_slider, showlegend=False,
    )
    if pd.api.types.is_datetime64_any_dtype(ohlc.index):
        if hide_weekends:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    elif not pd.api.types.is_numeric_dtype(ohlc.index):
        fig.update_xaxes(type="category")
    return fig


def has_weekend_dates(values: pd.Series, dayfirst: bool = True) -> Optional[bool]:
    """Whether date-like ``values`` fall on a Saturday or Sunday; None when they are not dates."""
    dates = date_values(values, dayfirst)
    if dates is None:
        return None
    return bool((dates.dropna().dt.dayofweek >= 5).any())


def waterfall_frame(
    frame: pd.DataFrame, x: str, y: str, aggregation: str = "Sum", dayfirst: bool = True
) -> pd.Series:
    """One change per X step (sum or mean of ``y``), in step order."""
    keys, steps = _step_keys(frame, x, dayfirst)
    data = pd.DataFrame({"step": keys.to_numpy(), "value": frame[y].to_numpy()}).dropna()
    grouped = data.groupby("step", sort=False, observed=True)["value"]
    changes = grouped.mean() if aggregation == "Mean" else grouped.sum()
    return changes.reindex(steps).dropna().rename_axis(x)


def waterfall_figure(
    changes: pd.Series, title: str, y_label: str, total: bool = True, total_label: str = "Total"
) -> go.Figure:
    """A waterfall of ``changes`` (index: steps), with an optional final total bar."""
    steps = [str(s.date()) if isinstance(s, pd.Timestamp) and s == s.normalize() else str(s) for s in changes.index]
    values = changes.to_list()
    measures = ["relative"] * len(values)
    if total:
        steps.append(total_label)
        values.append(0)  # Plotly computes the total bar's height.
        measures.append("total")
    fig = go.Figure(
        go.Waterfall(
            x=steps, y=values, measure=measures, name=y_label,
            text=[f"{v:+.4g}" for v in changes] + ([f"{changes.sum():.4g}"] if total else []),
            textposition="outside",
            increasing=dict(marker_color=INCREASING_COLOR), decreasing=dict(marker_color=DECREASING_COLOR),
            totals=dict(marker_color=TOTAL_COLOR), connector=dict(line=dict(color="gray", width=1, dash="dot")),
        )
    )
    fig.update_layout(
        title=title, xaxis_title=changes.index.name, yaxis_title=y_label, height=550,
        xaxis=dict(type="category"), showlegend=False,
    )
    return fig


BAR_AGGREGATIONS = ("Sum", "Mean", "Median", "Min", "Max", "Count")
# Aggregations whose values add up, so shares of a total make sense.
ADDITIVE_AGGREGATIONS = ("Sum", "Count")
BAR_SCALES = ("Values", "Percent of total", "Percent within X", "Percent within color")
BAR_ERRORS = ("None", "SD", "95% CI")
_BAR_FUNCS = {"Sum": "sum", "Mean": "mean", "Median": "median", "Min": "min", "Max": "max", "Count": "count"}


def bar_summary(
    frame: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    facets: Sequence[Optional[str]] = (),
    aggregation: str = "Sum",
    scale: str = "Values",
    error: str = "None",
) -> pd.DataFrame:
    """One row per bar: the grouping columns plus ``value`` (and ``error`` for means).

    Without ``y`` the value is the number of rows (a count plot); with ``y`` it is
    ``aggregation`` of ``y``. ``scale`` turns values into percentages of the total
    of each facet panel, of each X, or of each color. Rows missing a grouping
    value are left out.
    """
    panel = [f for f in facets if f]
    keys = list(dict.fromkeys(k for k in [x, hue, *panel] if k))
    data = frame.dropna(subset=keys)
    grouped = data.groupby(keys, observed=True, sort=False)
    if y is None:
        summary = grouped.size().rename("value").reset_index()
    else:
        summary = grouped[y].agg(_BAR_FUNCS[aggregation]).rename("value").reset_index()
        if aggregation == "Mean" and error != "None":
            spread = grouped[y].agg(["std", "count"]).reset_index(drop=True)
            if error == "SD":
                summary["error"] = spread["std"].to_numpy()
            else:
                from scipy.stats import t

                n = spread["count"].to_numpy()
                with np.errstate(invalid="ignore", divide="ignore"):
                    summary["error"] = t.ppf(0.975, np.maximum(n - 1, 1)) * spread["std"].to_numpy() / np.sqrt(n)
    if scale != "Values":
        within = {
            "Percent of total": panel,
            "Percent within X": panel + [x],
            "Percent within color": panel + ([hue] if hue else []),
        }[scale]
        within = list(dict.fromkeys(within))
        totals = summary.groupby(within, observed=True)["value"].transform("sum") if within else summary["value"].sum()
        summary["value"] = 100 * summary["value"] / totals
    return summary


# Bar labels: one size everywhere. Above grouped bars they stay horizontal while the
# busiest panel row has at most this many bars, vertical beyond (no overlaps).
BAR_LABEL_PX = 11
HORIZONTAL_LABEL_MAX_BARS = 24


def _bar_labels(fig, summary, x, hue, facet_col, facet_row, stacked, percent) -> None:
    template = "%{y:.1f}%" if percent else "%{y:.4~g}"
    if stacked:
        # Inside each segment, all the same size; segments too small for it stay blank.
        fig.update_traces(texttemplate=template, textposition="inside", insidetextanchor="middle", textangle=0)
        fig.update_layout(uniformtext=dict(minsize=BAR_LABEL_PX - 2, mode="hide"))
        return
    # Bars in the busiest panel times panels per row: the width labels compete for.
    panel = [c for c in (facet_col, facet_row) if c]
    bars = summary.groupby(panel, observed=True).size().max() if panel else len(summary)
    per_row = summary[facet_col].nunique() if facet_col else 1
    vertical = bars * per_row > HORIZONTAL_LABEL_MAX_BARS
    fig.update_traces(
        texttemplate=template,
        textposition="outside",
        textangle=-90 if vertical else 0,
        textfont_size=BAR_LABEL_PX,
        constraintext="none",  # Never shrunk to the bar's width: the same size everywhere.
        cliponaxis=False,
    )
    values = summary["value"] + (summary["error"].fillna(0) if "error" in summary else 0)
    top, bottom = float(values.max()), float(summary["value"].min())
    if bottom >= 0 and top > 0:  # Room above the tallest bar for its label.
        fig.update_yaxes(range=[0, top * (1.16 if vertical else 1.08)])


def bar_figure(
    summary: pd.DataFrame,
    x: str,
    value_label: str,
    hue: Optional[str] = None,
    facet_col: Optional[str] = None,
    facet_row: Optional[str] = None,
    category_orders: Optional[dict] = None,
    stacked: bool = False,
    show_labels: bool = True,
    percent: bool = False,
    title: str = "",
    palette: Sequence[str] = DEFAULT_PALETTE,
    date_group: Optional[str] = None,
) -> go.Figure:
    """Bars from ``bar_summary`` output; a datetime X gets a date axis."""
    fig = px.bar(
        summary,
        x=x,
        y="value",
        color=hue,
        facet_col=facet_col,
        facet_row=facet_row,
        error_y="error" if "error" in summary else None,
        barmode="relative" if stacked else "group",
        category_orders=category_orders or {},
        color_discrete_sequence=palette,
        labels={"value": value_label},
        title=title,
        width=800,
        height=600,
    )
    # px gives each color its own offsetgroup and aligns them across facet panels, so
    # every panel reserves a slot for every color, even colors it does not show: with
    # facets the bars get thin. Without those groups each panel splits its width among
    # its own bars; stacking uses one shared group (side by side otherwise, even in
    # "relative" mode).
    fig.update_traces(offsetgroup="stack" if stacked else "", alignmentgroup="")
    fig.update_layout(bargap=0.15, bargroupgap=0.05)
    if show_labels:
        _bar_labels(fig, summary, x, hue, facet_col, facet_row, stacked, percent)
    if pd.api.types.is_datetime64_any_dtype(summary[x]):
        if date_group in ("Week", "Month", "Year"):
            fig.update_xaxes(tickformat=DATE_GROUPS[date_group][1])
    elif not pd.api.types.is_numeric_dtype(summary[x]):
        fig.update_xaxes(type="category")
    return fig


# "Group dates by" choices for a date X: pandas period frequency and axis label format.
DATE_GROUPS = {
    "As is": (None, None),
    "Week": ("W", "%d %b %Y"),
    "Month": ("M", "%b %Y"),
    "Quarter": ("Q", None),
    "Year": ("Y", "%Y"),
}


def date_values(values: pd.Series, dayfirst: bool = True) -> Optional[pd.Series]:
    """``values`` as datetimes when they are dates (a datetime column, or text that
    all parses as dates); None otherwise."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return None
    present = values.dropna()
    if present.empty:
        return None
    parsed = parse_dates(values.astype(object), dayfirst)
    return parsed if parsed[present.index].notna().all() else None


def group_dates(dates: pd.Series, by: str) -> pd.Series:
    """Each date replaced by the start of its week, month, quarter or year."""
    freq = DATE_GROUPS[by][0]
    if freq is None:
        return dates
    return dates.dt.to_period(freq).dt.start_time


def date_labels(dates: pd.Series, by: str) -> pd.Series:
    """Readable labels for grouped dates (for heatmap rows)."""
    if by == "Quarter":
        return dates.dt.year.astype(str) + " Q" + dates.dt.quarter.astype(str)
    fmt = DATE_GROUPS[by][1]
    if fmt is None:
        has_time = (dates.dropna() != dates.dropna().dt.normalize()).any()
        fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
    return dates.dt.strftime(fmt)


RAW_VALUES = "None (raw values)"
LINE_AGGREGATIONS = ("Mean", "Median", "Sum", "Min", "Max", RAW_VALUES)
# Spread around a mean line: a t-based confidence interval, or ±1 standard deviation.
LINE_INTERVALS = {"None": None, "95% CI": 0.95, "68% CI": 0.68, "99% CI": 0.99, "SD": "sd"}


def line_summary(
    frame: pd.DataFrame,
    x: str,
    y: str,
    groups: Sequence[Optional[str]] = (),
    aggregation: str = "Mean",
    interval: str = "None",
) -> pd.DataFrame:
    """One row per X step of each line: columns ``groups``, ``x``, value, lower, upper.

    Lines are split by ``groups`` (color, style, facets). Each X step is aggregated,
    or, with RAW_VALUES, every row is kept. Rows come in line order, then X order
    (category order for categoricals). lower/upper hold the interval around a Mean
    (NaN where it needs two or more values, or no interval is asked for).
    """
    groups = [g for g in dict.fromkeys(groups) if g and g not in (x, y)]
    data = frame[list(dict.fromkeys([x, y, *groups]))].copy()
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna(subset=[x, y])
    keys = groups + [x]
    if aggregation == RAW_VALUES:
        out = data.rename(columns={y: "value"})
        if out.empty:
            return out.assign(lower=[], upper=[])
        out = out.sort_values(keys, kind="stable") if keys else out
        return out[keys + ["value"]].assign(lower=np.nan, upper=np.nan).reset_index(drop=True)

    stats = data.groupby(keys, observed=True, sort=True)[y].agg(
        value=_BAR_FUNCS[aggregation], mean="mean", std="std", count="count"
    ).reset_index()
    stats = stats[stats["count"] > 0]
    level = LINE_INTERVALS.get(interval)
    lower = upper = pd.Series(np.nan, index=stats.index)
    if aggregation == "Mean" and level == "sd":
        lower, upper = stats["mean"] - stats["std"], stats["mean"] + stats["std"]
    elif aggregation == "Mean" and level:
        from scipy.stats import t

        se = stats["std"] / np.sqrt(stats["count"])
        with np.errstate(invalid="ignore", divide="ignore"):
            half = t.ppf((1 + level) / 2, stats["count"] - 1) * se
        lower, upper = stats["mean"] - half, stats["mean"] + half
    return stats[keys + ["value"]].assign(lower=lower, upper=upper).reset_index(drop=True)


def line_widths(frame: pd.DataFrame, groups: Sequence[str], width: str,
                thinnest: float = 1.0, widest: float = 6.0) -> dict:
    """Line width per line (a tuple of ``groups`` values): the mean of ``width``
    for that line, scaled to ``thinnest``..``widest`` pixels."""
    values = pd.to_numeric(frame[width], errors="coerce")
    means = values.groupby([frame[g] for g in groups], observed=True).mean() if groups else \
        pd.Series([values.mean()], index=[()])
    low, high = means.min(), means.max()
    span = high - low
    scaled = thinnest + (means - low) / span * (widest - thinnest) if span > 0 else \
        means * 0 + (thinnest + widest) / 2
    return {(k if isinstance(k, tuple) else (k,)): float(v) for k, v in scaled.items() if pd.notna(v)}


HISTOGRAM_STATS = ("count", "probability", "percent", "density")
MAX_STEP_BINS = 60


def histogram_edges(values: pd.Series) -> Optional[np.ndarray]:
    """Equal-width bin edges shared by every group: numpy's "auto" rule, at most
    MAX_STEP_BINS bins. None when there are no numbers."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    edges = np.histogram_bin_edges(values, bins="auto")
    if len(edges) - 1 > MAX_STEP_BINS:
        edges = np.histogram_bin_edges(values, bins=MAX_STEP_BINS)
    return edges


def _panel_sizes(data: pd.DataFrame, values: pd.Series, panel: list):
    """A function giving the count of numbers in the panel that ``labels`` belongs to."""
    present = values.notna()
    if not panel:
        total = int(present.sum())
        return lambda labels: total
    sizes = present.groupby([data[f] for f in panel], observed=True).sum()
    return lambda labels: int(sizes.loc[tuple(labels[f] for f in panel) if len(panel) > 1 else labels[panel[0]]])


def histogram_weights(
    frame: pd.DataFrame,
    x: str,
    facets: Sequence[Optional[str]] = (),
    stat: str = "count",
    cumulative: bool = False,
) -> tuple[pd.Series, np.ndarray]:
    """Row weights and bin edges for a histogram normalized over each panel (all
    colors together, i.e. common norm): summing the weights in a bin gives the
    color's height in ``stat`` units, so colors add up to the panel's total."""
    edges = histogram_edges(frame[x])
    values = pd.to_numeric(frame[x], errors="coerce")
    panel = [f for f in facets if f]
    present = values.notna()
    sizes = present.groupby([frame[f] for f in panel], observed=True).transform("sum") if panel else \
        pd.Series(present.sum(), index=frame.index)
    weights = pd.Series(1.0, index=frame.index)
    if stat in ("probability", "percent", "density"):
        weights = weights / sizes.replace(0, np.nan)
    if stat == "density" and not cumulative and edges is not None:
        weights = weights / (edges[1] - edges[0])
    if stat == "percent":
        weights = weights * 100
    return weights.where(present), edges


def step_histogram_frame(
    frame: pd.DataFrame,
    x: str,
    hue: Optional[str] = None,
    facets: Sequence[Optional[str]] = (),
    stat: str = "count",
    cumulative: bool = False,
    stacked: bool = False,
    common_norm: bool = False,
) -> pd.DataFrame:
    """Outline points of a step histogram: one closed step line per group.

    All groups share the same bins (histogram_edges), so their outlines are
    comparable. Each group is normalized by its own size, like Plotly's histnorm,
    or, with ``common_norm``, by the size of its panel (all colors together), so
    each color's area is its share. ``stacked`` adds each color on top of the
    previous ones (in color order) within a panel.
    """
    edges = histogram_edges(frame[x])
    if edges is None:
        return pd.DataFrame(columns=[x, "value"])
    widths = np.diff(edges)
    panel = [f for f in facets if f]
    keys = list(dict.fromkeys(k for k in [*panel, hue] if k))
    data = frame.dropna(subset=keys)
    values = pd.to_numeric(data[x], errors="coerce")
    panel_sizes = _panel_sizes(data, values, panel)
    groups = data.groupby(keys, observed=True, sort=False) if keys else [((), data)]
    rows = []
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        counts, _ = np.histogram(values[group.index].dropna(), bins=edges)
        labels = dict(zip(keys, key))
        n = panel_sizes(labels) if common_norm else counts.sum()
        heights = counts.astype(float)
        if stat in ("probability", "percent", "density") and n:
            heights = heights / n
        if cumulative:
            heights = np.cumsum(heights)
        elif stat == "density":
            heights = heights / widths
        if stat == "percent":
            heights = heights * 100
        rows.append((labels, heights))
    if stacked and hue:
        running = {}
        for labels, heights in rows:
            panel_key = tuple(labels.get(f) for f in panel)
            base = running.get(panel_key, np.zeros_like(heights))
            heights += base
            running[panel_key] = heights.copy()
    out = []
    for labels, heights in rows:
        # A closed outline: up from zero at the first edge, across each bin, down at the last edge.
        xs = np.concatenate([[edges[0]], edges[:-1], [edges[-1], edges[-1]]])
        ys = np.concatenate([[0.0], heights, [heights[-1], 0.0]])
        part = pd.DataFrame({x: xs, "value": ys})
        for column, value in labels.items():
            part[column] = value
        out.append(part)
    return pd.concat(out, ignore_index=True)


def fill_steps(fig: go.Figure, alpha: float) -> go.Figure:
    """Fill each closed step outline with its own line color at ``alpha`` (0: outline only)."""
    if alpha > 0:
        for trace in fig.data:
            trace.update(fill="toself", fillcolor=_rgba(trace.line.color, alpha))
    return fig


# Histogram "Multiple" choices (seaborn's names) and Plotly's barmode for each.
HISTOGRAM_MULTIPLE = {"layer": "overlay", "dodge": "group", "stack": "stack"}


def histogram_bars(fig: go.Figure, multiple: str, opacity: float) -> go.Figure:
    """Lay out px.histogram bars as ``multiple`` says, at ``opacity``.

    px gives each color its own offsetgroup, which puts the colors side by side
    in every barmode: "layer" never overlapped and "stack" never stacked. Layer
    and stack share one group; dodge groups only the colors within each panel.
    """
    shared = multiple in ("layer", "stack")
    fig.update_traces(offsetgroup="hist" if shared else "", alignmentgroup="", opacity=opacity)
    fig.update_layout(barmode=HISTOGRAM_MULTIPLE[multiple])
    return fig
