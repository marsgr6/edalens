"""Streamlit-free reports: charts saved while exploring, exported as HTML or PDF.

A ``Report`` holds Plotly figures (kept as their JSON spec, so they stay
interactive in HTML) and matplotlib figures (kept as SVG for HTML and PNG for
PDF). It serializes to JSON, so the app can hand reports back to Python::

    >>> from edalens import Report
    >>> report = Report("Tips")
    >>> report.add(fig, note="Dinner tips are higher.")
    >>> report.save_html("tips.html")
    >>> report.save_pdf("tips.pdf", context="talk")
"""

import base64
import html
import io
import json
import textwrap
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import plotly.graph_objects as go
import plotly.io as pio

from edalens.utils.export import DEFAULT_CONTEXT, PDF_HELP, ExportError, figures_pdf

PLOTLY = "plotly"
IMAGE = "image"
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
COLUMN_CHOICES = (1, 2)
NOTE_WRAP_CHARS = 110
# Set by the launcher: a folder where the app saves reports for the Python session.
REPORT_DIR_ENV = "EDALENS_REPORT_DIR"
IMAGE_DPI = 200


@dataclass
class ReportItem:
    """One saved chart: a Plotly spec, or a matplotlib figure as SVG and PNG."""

    kind: str
    title: str
    note: str = ""
    figure: Optional[str] = None  # Plotly figure JSON.
    svg: Optional[str] = None
    png: Optional[str] = None  # Base64.
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])  # Stable across reorders.

    def plotly_figure(self) -> go.Figure:
        return pio.from_json(self.figure)


def figure_title(fig, default: str = "Chart") -> str:
    if hasattr(fig, "layout"):
        text = fig.layout.title.text
    else:
        text = fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else ""
        text = text or next((ax.get_title() for ax in fig.axes if ax.get_title()), "")
    return text or default


@dataclass
class Report:
    """Charts in the order they were saved, with a title and optional notes."""

    title: str = "EDALens report"
    items: list = field(default_factory=list)
    columns: int = 1
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="minutes"))

    # -- Building ---------------------------------------------------------------
    def add(self, fig, title: Optional[str] = None, note: str = "") -> ReportItem:
        """Append a Plotly or matplotlib figure; returns the new item."""
        title = title or figure_title(fig, f"Chart {len(self.items) + 1}")
        if hasattr(fig, "savefig"):
            svg, png = io.StringIO(), io.BytesIO()
            fig.savefig(svg, format="svg", bbox_inches="tight")
            fig.savefig(png, format="png", dpi=IMAGE_DPI, bbox_inches="tight")
            item = ReportItem(
                IMAGE, title, note, svg=svg.getvalue(), png=base64.b64encode(png.getvalue()).decode()
            )
        else:
            item = ReportItem(PLOTLY, title, note, figure=go.Figure(fig).to_json())
        self.items.append(item)
        return item

    def move(self, index: int, step: int) -> None:
        """Move item ``index`` up (step -1) or down (step 1), if there is room."""
        target = index + step
        if 0 <= index < len(self.items) and 0 <= target < len(self.items):
            self.items[index], self.items[target] = self.items[target], self.items[index]

    def remove(self, index: int) -> None:
        del self.items[index]

    def index_of(self, uid: str) -> int:
        return next(i for i, item in enumerate(self.items) if item.uid == uid)

    # -- JSON ---------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Report":
        data = json.loads(text)
        data["items"] = [ReportItem(**item) for item in data.get("items", [])]
        return cls(**data)

    def save(self, path: Union[str, Path]) -> Path:
        """Write the report as JSON (reload it with ``Report.load``)."""
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Report":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # -- HTML ---------------------------------------------------------------------
    def to_html(self, embed_plotly: bool = True) -> str:
        """A single static page: a dashboard grid of the charts with their notes.

        ``embed_plotly=True`` inlines plotly.js (~4.5 MB) so the page works
        offline; ``False`` loads it from the Plotly CDN (small file, needs internet).
        """
        if embed_plotly:
            from plotly.offline import get_plotlyjs

            script = f"<script>{get_plotlyjs()}</script>"
        else:
            script = f'<script src="{PLOTLY_CDN}"></script>'
        cards = "\n".join(self._card(i, item) for i, item in enumerate(self.items))
        columns = self.columns if self.columns in COLUMN_CHOICES else 1
        created = html.escape(self.created.replace("T", " "))
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(self.title)}</title>
{script}
<style>
  :root {{ --bg: #f6f7f9; --card: #fff; --text: #1f2933; --muted: #616e7c; --line: #e4e7eb; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #111418; --card: #1b1f24; --text: #e4e7eb; --muted: #9aa5b1; --line: #2d333b; }}
  }}
  body {{ margin: 0; background: var(--bg); color: var(--text);
         font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  header {{ padding: 28px 24px 8px; max-width: 1400px; margin: 0 auto; }}
  h1 {{ margin: 0 0 4px; font-size: 1.7rem; }}
  .meta {{ color: var(--muted); font-size: .9rem; }}
  main {{ display: grid; grid-template-columns: repeat({columns}, minmax(0, 1fr)); gap: 16px;
         padding: 16px 24px 32px; max-width: 1400px; margin: 0 auto; }}
  @media (max-width: 900px) {{ main {{ grid-template-columns: minmax(0, 1fr); padding: 12px 16px; }} }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
          padding: 14px 16px; min-width: 0; }}
  .card h2 {{ margin: 0 0 6px; font-size: 1.05rem; }}
  .note {{ color: var(--muted); margin: 0 0 8px; white-space: pre-wrap; }}
  .figure img, .figure svg {{ max-width: 100%; height: auto; }}
  .plot {{ width: 100%; }}
  footer {{ text-align: center; color: var(--muted); font-size: .8rem; padding: 0 0 24px; }}
</style>
</head>
<body>
<header>
  <h1>{html.escape(self.title)}</h1>
  <div class="meta">{len(self.items)} chart{"s" if len(self.items) != 1 else ""} · {created}</div>
</header>
<main>
{cards}
</main>
<footer>Made with EDALens</footer>
</body>
</html>
"""

    def _card(self, index: int, item: ReportItem) -> str:
        note = f'<p class="note">{html.escape(item.note)}</p>' if item.note.strip() else ""
        if item.kind == PLOTLY:
            fig = item.plotly_figure()
            # The card sets the width and shows the title; keep the chart's own height.
            fig.update_layout(width=None, autosize=True, title=None, margin=dict(t=40))
            body = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=False,
                div_id=f"chart-{index}",
                default_width="100%",
                config={"responsive": True, "displaylogo": False},
            )
        else:
            body = item.svg or f'<img alt="" src="data:image/png;base64,{item.png}">'
        return (
            f'<section class="card"><h2>{html.escape(item.title)}</h2>{note}'
            f'<div class="figure">{body}</div></section>'
        )

    def save_html(self, path: Union[str, Path], embed_plotly: bool = True) -> Path:
        path = Path(path)
        path.write_text(self.to_html(embed_plotly), encoding="utf-8")
        return path

    def html_height(self) -> int:
        """Rough page height in px, for showing the page in a fixed-height frame."""
        heights = []
        for item in self.items:
            height = 450
            if item.kind == PLOTLY:
                height = item.plotly_figure().layout.height or 450
            heights.append(height + 90 + (40 if item.note else 0))
        columns = self.columns if self.columns in COLUMN_CHOICES else 1
        rows = [max(heights[i : i + columns]) for i in range(0, len(heights), columns)]
        return int(160 + sum(rows) + 16 * len(rows))

    # -- PDF ----------------------------------------------------------------------
    def _title_page(self) -> go.Figure:
        lines = [f"{i + 1}. {html.escape(item.title)}" for i, item in enumerate(self.items[:30])]
        if len(self.items) > 30:
            lines.append(f"… and {len(self.items) - 30} more")
        fig = go.Figure()
        fig.add_annotation(
            text=f"<b>{html.escape(self.title)}</b>", x=0.5, y=0.8, xref="paper", yref="paper",
            yanchor="bottom", showarrow=False, font=dict(size=30),
        )
        fig.add_annotation(
            text=f"{len(self.items)} charts · {html.escape(self.created.replace('T', ' '))}",
            x=0.5, y=0.78, xref="paper", yref="paper", yanchor="top", showarrow=False,
            font=dict(size=14, color="#616e7c"),
        )
        fig.add_annotation(
            text="<br>".join(lines), x=0.5, y=0.64, xref="paper", yref="paper", yanchor="top",
            showarrow=False, align="left", font=dict(size=13),
        )
        fig.update_layout(
            xaxis_visible=False, yaxis_visible=False, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

    def _page(self, item: ReportItem) -> go.Figure:
        """The item as a Plotly figure titled with its title and note."""
        note = "<br>".join(textwrap.wrap(html.escape(item.note), NOTE_WRAP_CHARS))
        if item.kind == PLOTLY:
            fig = item.plotly_figure()
        else:
            fig = go.Figure()
            fig.add_layout_image(
                source=f"data:image/png;base64,{item.png}", xref="paper", yref="paper",
                x=0.5, y=0.5, sizex=1, sizey=1, xanchor="center", yanchor="middle",
            )
            fig.update_layout(
                xaxis_visible=False, yaxis_visible=False, plot_bgcolor="white", margin=dict(l=30, r=30, b=30)
            )
        title = f"{html.escape(item.title)}" + (f"<br><sup>{note}</sup>" if note else "")
        top = max(fig.layout.margin.t or 100, 100 + 18 * note.count("<br>"))
        fig.update_layout(title=dict(text=title), margin=dict(t=top))
        return fig

    def to_pdf(self, context: str = DEFAULT_CONTEXT, title_page: bool = True) -> bytes:
        """One PDF: an optional contents page, then one page per chart.

        Needs kaleido (see ``edalens.utils.export``) and pypdf to join the pages.
        """
        if not self.items:
            raise ExportError("The report is empty: add charts first.")
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError as exc:
            raise ExportError(
                "Joining report pages needs pypdf: `pip install pypdf` "
                "(or `pip install 'edalens[pdf]'`).\n\n" + PDF_HELP
            ) from exc
        pages = [self._title_page()] if title_page else []
        pages += [self._page(item) for item in self.items]
        writer = PdfWriter()
        for pdf in figures_pdf(pages, context):
            for pdf_page in PdfReader(io.BytesIO(pdf)).pages:
                writer.add_page(pdf_page)
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()

    def save_pdf(self, path: Union[str, Path], context: str = DEFAULT_CONTEXT, title_page: bool = True) -> Path:
        path = Path(path)
        path.write_bytes(self.to_pdf(context, title_page))
        return path

    def _repr_html_(self) -> str:
        titles = "".join(f"<li>{html.escape(item.title)}</li>" for item in self.items)
        return f"<b>{html.escape(self.title)}</b> ({len(self.items)} charts)<ol>{titles}</ol>"
