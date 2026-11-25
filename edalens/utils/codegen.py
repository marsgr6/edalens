"""Python code that recreates a Visualization chart with pandas and plotly express.

The code starts from the data as the app has it ("Download Processed Data as CSV"
on the Load data page) and repeats the plot's settings, so it runs on its own in a
notebook. EDALens adds some finishing touches (bar labels, step outlines, mean
markers) that the code leaves out or notes in a comment.
"""

from __future__ import annotations

from typing import Optional, Sequence

HEADER = '''import pandas as pd
import plotly.express as px

# The data as it is in EDALens now: Load data page → "Download Processed Data as CSV".
df = pd.read_csv("processed_dataset.csv")
'''
PALETTE = "px.colors.qualitative.Set2"


def _literal(value) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_literal(k)}: {_literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if hasattr(value, "item"):  # numpy scalars
        value = value.item()
    return repr(value)


def px_call(function: str, frame: str = "df", **kwargs) -> str:
    """``fig = px.<function>(frame, key=value, ...)``, leaving out None values."""
    lines = [f"fig = px.{function}(", f"    {frame},"]
    for key, value in kwargs.items():
        if value is None or value == {} or value == "None":
            continue
        text = PALETTE if key == "color_discrete_sequence" else _literal(value)
        lines.append(f"    {key}={text},")
    lines.append(")")
    return "\n".join(lines)


def script(*blocks: Optional[str]) -> str:
    body = "\n\n".join(block.strip("\n") for block in blocks if block)
    return f"{HEADER}\n{body}\n\nfig.show()\n"


def dates_block(x: str, dayfirst: bool, group: Optional[str]) -> str:
    """Parse a text date column and, optionally, group it by week/month/quarter/year."""
    from edalens.utils.plot_utils import DATE_GROUPS

    line = f"df[{x!r}] = pd.to_datetime(df[{x!r}], dayfirst={dayfirst})"
    freq = DATE_GROUPS.get(group or "As is", (None, None))[0]
    if freq:
        line += f"\ndf[{x!r}] = df[{x!r}].dt.to_period({freq!r}).dt.start_time  # Group by {group.lower()}."
    return line


def bars(x: str, y: Optional[str], hue: Optional[str], facets: Sequence[Optional[str]], aggregation: str,
         scale: str, error: str, stacked: bool, category_orders: dict, title: str, value_label: str,
         dates: Optional[str] = None) -> str:
    panel = [f for f in facets if f]
    keys = list(dict.fromkeys(k for k in [x, hue, *panel] if k))
    func = {"Sum": "sum", "Mean": "mean", "Median": "median", "Min": "min", "Max": "max", "Count": "count"}
    if y is None:
        summary = f"summary = df.dropna(subset={keys!r}).groupby({keys!r}).size().reset_index(name='value')"
    else:
        summary = (f"grouped = df.dropna(subset={keys!r}).groupby({keys!r})[{y!r}]\n"
                   f"summary = grouped.agg({func[aggregation]!r}).reset_index(name='value')")
        if aggregation == "Mean" and error == "SD":
            summary += "\nsummary['error'] = grouped.std().to_numpy()"
        elif aggregation == "Mean" and error != "None":
            summary += ("\nfrom scipy.stats import t\nn = grouped.count().to_numpy()\n"
                        "summary['error'] = t.ppf(0.975, n - 1) * grouped.std().to_numpy() / n ** 0.5  # 95% CI")
    if scale != "Values":
        within = {"Percent of total": panel, "Percent within X": panel + [x],
                  "Percent within color": panel + ([hue] if hue else [])}[scale]
        within = list(dict.fromkeys(within))
        total = f"summary.groupby({within!r})['value'].transform('sum')" if within else "summary['value'].sum()"
        summary += f"\nsummary['value'] = 100 * summary['value'] / {total}  # {scale}"
    figure = px_call(
        "bar", "summary", x=x, y="value", color=hue, facet_col=facets[0] if facets else None,
        facet_row=facets[1] if len(facets) > 1 else None,
        error_y="error" if aggregation == "Mean" and error != "None" and y else None,
        barmode="relative" if stacked else "group", category_orders=category_orders,
        color_discrete_sequence=True, labels={"value": value_label}, title=title,
    )
    if not dates:
        figure += "\nfig.update_xaxes(type='category')"
    return script(dates, summary, figure)


def lineplot(x: str, y: str, groups: Sequence[str], hue: Optional[str], style: Optional[str],
             facet_col: Optional[str], facet_row: Optional[str], aggregation: str, interval: str,
             markers: bool, category_orders: dict, title: str, y_title: str, dates: Optional[str] = None) -> str:
    keys = list(dict.fromkeys([*[g for g in groups if g and g != x], x]))
    if aggregation.startswith("None"):
        summary = f"summary = df.dropna(subset=[{x!r}, {y!r}]).sort_values({keys!r}).rename(columns={{{y!r}: 'value'}})"
        interval = "None"
    else:
        func = {"Mean": "mean", "Median": "median", "Sum": "sum", "Min": "min", "Max": "max"}[aggregation]
        summary = (f"grouped = df.dropna(subset=[{x!r}, {y!r}]).groupby({keys!r})[{y!r}]\n"
                   f"summary = grouped.agg({func!r}).reset_index(name='value')")
        if aggregation == "Mean" and interval == "SD":
            summary += "\nsummary['spread'] = grouped.std().to_numpy()  # ±1 SD"
        elif aggregation == "Mean" and interval != "None":
            level = float(interval.split("%")[0]) / 100
            summary += ("\nfrom scipy.stats import t\nn = grouped.count().to_numpy()\n"
                        f"summary['spread'] = t.ppf({(1 + level) / 2!r}, n - 1) * grouped.std().to_numpy() / n ** 0.5"
                        f"  # {interval}")
    spread = "spread" if aggregation == "Mean" and interval != "None" else None
    figure = px_call(
        "line", "summary", x=x, y="value", color=hue, line_dash=style if style != hue else None,
        facet_col=facet_col, facet_row=facet_row, error_y=spread, markers=markers or None,
        category_orders=category_orders, color_discrete_sequence=True, labels={"value": y_title}, title=title,
    )
    if spread:
        figure += "\n# EDALens can draw the spread as a shaded band instead of error bars."
    return script(dates, summary, figure)


def histogram(x: str, hue: Optional[str], facet_col: Optional[str], facet_row: Optional[str], stat: str,
              cumulative: bool, multiple: str, opacity: float, category_orders: dict, step: bool,
              common_norm: bool) -> str:
    barmode = {"layer": "overlay", "dodge": "group", "stack": "stack"}[multiple]
    figure = px_call(
        "histogram", x=x, color=hue, facet_col=facet_col, facet_row=facet_row,
        histnorm=None if stat == "count" else stat, cumulative=cumulative or None, barmode=barmode,
        opacity=opacity if opacity < 1 else None, category_orders=category_orders, color_discrete_sequence=True,
    )
    notes = []
    if step:
        notes.append("# EDALens draws this as step outlines on shared bins; plotly express draws bars.")
    if common_norm:
        notes.append("# Common norm: EDALens normalizes all colors of a panel together;\n"
                     "# histnorm above normalizes each color on its own.")
    return script(figure, "\n".join(notes) or None)


def simple(function: str, prep: Optional[str] = None, note: Optional[str] = None, **kwargs) -> str:
    """Any single plotly express call, e.g. scatter, box, violin or strip."""
    kwargs.setdefault("color_discrete_sequence", True)
    return script(prep, px_call(function, **kwargs), note)


def correlation(drop_constant: bool = True) -> str:
    prep = "numbers = df.select_dtypes('number')"
    if drop_constant:
        prep += "\nnumbers = numbers.loc[:, numbers.nunique() > 1]"
    figure = ("fig = px.imshow(numbers.corr(), text_auto=True, color_continuous_scale='Viridis',\n"
              "                title='Correlation Matrix', width=800, height=600)")
    return script(prep, figure)
