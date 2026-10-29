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
                spanmode="hard", points=_OUTLIER_POINTS["boxpoints"],
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
