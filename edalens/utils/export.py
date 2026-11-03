"""Streamlit-free figure export: Plotly and matplotlib figures as PDF bytes."""

import io

import plotly.graph_objects as go

PDF_HELP = (
    "PDF export renders the chart with kaleido: `pip install kaleido` "
    "(or `pip install 'edalens[pdf]'`). kaleido 1.x also needs Google Chrome or "
    "Chromium; without one, run `import kaleido; kaleido.get_chrome_sync()` once."
)

# Like seaborn's plotting contexts: one scale for text, lines and markers.
CONTEXTS = {"paper": 0.8, "notebook": 1.0, "talk": 1.5, "poster": 2.0}
DEFAULT_CONTEXT = "notebook"
# Plotly.js defaults, used when a figure leaves them unset.
PLOTLY_FONT_PX = 12
PLOTLY_LINE_PX = 2
PLOTLY_MARKER_PX = 6
PLOTLY_MARGIN_PX = dict(l=80, r=80, t=100, b=80)
# Page size for figures that follow the container width on screen.
PDF_SIZE_PX = (1000, 600)
# Larger text needs wider gaps between subplots; panels keep at least this share.
MIN_PANEL_SHARE = 0.5


class ExportError(RuntimeError):
    """A figure could not be exported; the message is shown to users."""


def _scale_fonts(node, scale: float) -> None:
    """Multiply every explicit font size (font, tickfont, textfont, ...) in place."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("font") and isinstance(value, dict) and isinstance(value.get("size"), (int, float)):
                value["size"] = value["size"] * scale
            _scale_fonts(value, scale)
    elif isinstance(node, list):
        for item in node:
            _scale_fonts(item, scale)


def _widen_gaps(layout: dict, axis: str, scale: float) -> None:
    """Grow the gaps between side-by-side (or stacked) subplots by ``scale``.

    Panels shrink to make room; nothing changes unless the axes' domains tile
    the figure without overlapping (subplots, marginals).
    """
    names = [k for k in layout if k == axis or (k.startswith(axis) and k[len(axis):].isdigit())]
    domains = {tuple(layout[n]["domain"]) for n in names if isinstance(layout[n], dict) and "domain" in layout[n]}
    spans = sorted(domains)
    if len(spans) < 2 or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        return
    gaps = [b[0] - a[1] for a, b in zip(spans, spans[1:])]
    panels = sum(b - a for a, b in spans)
    total = spans[-1][1] - spans[0][0]
    new_gaps = [g * scale for g in gaps]
    room = total - MIN_PANEL_SHARE * panels
    if sum(new_gaps) > room:
        new_gaps = [g * room / sum(new_gaps) for g in new_gaps]
    shrink = (total - sum(new_gaps)) / panels
    start, moved = spans[0][0], {}
    for i, (a, b) in enumerate(spans):
        moved[(a, b)] = [start, start + (b - a) * shrink]
        start = moved[(a, b)][1] + (new_gaps[i] if i < len(new_gaps) else 0)
    for n in names:
        if isinstance(layout[n], dict) and "domain" in layout[n]:
            old = tuple(layout[n]["domain"])
            layout[n]["domain"] = moved[old]
    # Facet titles sit just above their panel's top edge: keep them there.
    tops = {round(b, 6): moved[(a, b)][1] for a, b in spans}
    ref = "yref" if axis == "yaxis" else "xref"
    coord = "y" if axis == "yaxis" else "x"
    for note in layout.get("annotations", []):
        if note.get(ref) == "paper" and round(note.get(coord, -1), 6) in tops:
            note[coord] = tops[round(note[coord], 6)]


def scaled_figure(fig: go.Figure, context: str = DEFAULT_CONTEXT) -> go.Figure:
    """A copy of ``fig`` with text, lines and markers scaled for ``context``."""
    scale = CONTEXTS[context]
    spec = fig.to_dict()
    layout = spec.setdefault("layout", {})
    _scale_fonts(spec, scale)
    # Unset sizes inherit layout.font (titles are 1.4x it), so this scales the rest.
    font = layout.setdefault("font", {})
    font.setdefault("size", PLOTLY_FONT_PX * scale)
    for trace in spec.get("data", []):
        line = trace.get("line")
        if trace.get("type") in ("scatter", "scattergl") or isinstance(line, dict):
            trace.setdefault("line", {})
            width = trace["line"].get("width", PLOTLY_LINE_PX)
            if isinstance(width, (int, float)):
                trace["line"]["width"] = width * scale
        marker = trace.get("marker")
        if trace.get("type") in ("scatter", "scattergl") and isinstance(marker, dict):
            size = marker.get("size", PLOTLY_MARKER_PX)
            if isinstance(size, (int, float)):
                marker["size"] = size * scale
    if scale > 1:
        margin = layout.setdefault("margin", {})
        for side, px in PLOTLY_MARGIN_PX.items():
            margin[side] = margin.get(side, px) * scale
        _widen_gaps(layout, "yaxis", scale)
        _widen_gaps(layout, "xaxis", scale)
    layout.setdefault("width", PDF_SIZE_PX[0])
    layout.setdefault("height", PDF_SIZE_PX[1])
    return go.Figure(spec)


def _matplotlib_pdf(fig, scale: float) -> bytes:
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    texts = [(t, t.get_fontsize()) for t in fig.findobj(Text)]
    lines = [(line, line.get_linewidth(), line.get_markersize()) for line in fig.findobj(Line2D)]
    try:
        for text, size in texts:
            text.set_fontsize(size * scale)
        for line, width, marker in lines:
            line.set_linewidth(width * scale)
            line.set_markersize(marker * scale)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="pdf", bbox_inches="tight")
        return buffer.getvalue()
    finally:  # The figure may be shown again; leave it as it was.
        for text, size in texts:
            text.set_fontsize(size)
        for line, width, marker in lines:
            line.set_linewidth(width)
            line.set_markersize(marker)


def figure_pdf(fig, context: str = DEFAULT_CONTEXT) -> bytes:
    """PDF bytes of a Plotly figure or a matplotlib figure, scaled for ``context``."""
    if context not in CONTEXTS:
        raise ValueError(f"Unknown context {context!r}; choose one of {', '.join(CONTEXTS)}.")
    if hasattr(fig, "savefig"):  # matplotlib needs nothing extra.
        return _matplotlib_pdf(fig, CONTEXTS[context])
    try:
        import kaleido
    except ImportError as exc:
        raise ExportError(PDF_HELP) from exc
    fig = scaled_figure(fig, context)
    try:
        if hasattr(kaleido, "calc_fig_sync"):
            # kaleido >= 1: works with plotly 5 and 6 (plotly 5's to_image does not).
            return kaleido.calc_fig_sync(fig, opts=dict(format="pdf"))
        return fig.to_image(format="pdf")  # kaleido 0.2 bundles its own Chromium.
    except Exception as exc:
        raise ExportError(f"{PDF_HELP}\n\nDetails: {exc}") from exc


def figures_pdf(figs, context: str = DEFAULT_CONTEXT) -> list:
    """PDF bytes for several Plotly figures, rendered with one browser when possible.

    Starting Chrome takes seconds; kaleido >= 1 renders a batch in one session.
    """
    if context not in CONTEXTS:
        raise ValueError(f"Unknown context {context!r}; choose one of {', '.join(CONTEXTS)}.")
    try:
        import kaleido
    except ImportError as exc:
        raise ExportError(PDF_HELP) from exc
    if not hasattr(kaleido, "write_fig_from_object_sync"):
        return [figure_pdf(fig, context) for fig in figs]
    import tempfile
    from pathlib import Path

    scaled = [scaled_figure(fig, context) for fig in figs]
    with tempfile.TemporaryDirectory(prefix="edalens-pdf-") as folder:
        paths = [Path(folder) / f"page-{i}.pdf" for i in range(len(scaled))]
        try:
            kaleido.write_fig_from_object_sync(
                [dict(fig=fig, path=path, opts=dict(format="pdf")) for fig, path in zip(scaled, paths)]
            )
            return [path.read_bytes() for path in paths]
        except Exception as exc:
            raise ExportError(f"{PDF_HELP}\n\nDetails: {exc}") from exc
