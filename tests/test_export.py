"""PDF export of Plotly and matplotlib figures."""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from edalens.utils.export import ExportError, figure_pdf, scaled_figure


def test_matplotlib_figures_export_without_kaleido(monkeypatch):
    monkeypatch.setitem(sys.modules, "kaleido", None)
    fig, ax = plt.subplots()
    ax.plot([1, 3, 2])
    assert figure_pdf(fig).startswith(b"%PDF")
    plt.close(fig)


def test_missing_kaleido_explains_how_to_get_pdf_export(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "kaleido", None
    )  # import kaleido raises ImportError
    with pytest.raises(ExportError, match="pip install kaleido"):
        figure_pdf(go.Figure(go.Scatter(y=[1, 2])))


def test_plotly_figures_export_with_kaleido(monkeypatch):
    calls = []

    class FakeKaleido:
        @staticmethod
        def calc_fig_sync(fig, opts):
            calls.append(opts)
            return b"%PDF-fake"

    monkeypatch.setitem(sys.modules, "kaleido", FakeKaleido)
    assert figure_pdf(go.Figure()) == b"%PDF-fake"
    assert calls == [{"format": "pdf"}]


def test_render_failures_become_export_errors(monkeypatch):
    class BrokenKaleido:
        @staticmethod
        def calc_fig_sync(fig, opts):
            raise RuntimeError("Chrome not found")

    monkeypatch.setitem(sys.modules, "kaleido", BrokenKaleido)
    with pytest.raises(ExportError, match="Chrome not found"):
        figure_pdf(go.Figure())


def test_contexts_scale_text_lines_and_markers_like_seaborn():
    fig = go.Figure(go.Scatter(y=[1, 2], marker=dict(size=8)))
    fig.update_layout(xaxis=dict(tickfont=dict(size=10)))
    poster = scaled_figure(fig, "poster")
    assert poster.layout.font.size == 24  # Plotly's 12 px default, doubled.
    assert poster.layout.xaxis.tickfont.size == 20
    assert poster.data[0].line.width == 4 and poster.data[0].marker.size == 16
    assert scaled_figure(fig, "paper").layout.font.size == pytest.approx(9.6)
    assert fig.layout.font.size is None  # The original is untouched.


def test_larger_contexts_widen_the_gaps_between_subplots():
    fig = make_subplots(rows=2, cols=1, subplot_titles=["a", "b"], vertical_spacing=0.1)
    gap = lambda f: f.layout.yaxis.domain[0] - f.layout.yaxis2.domain[1]  # noqa: E731
    talk = scaled_figure(fig, "talk")
    assert gap(talk) == pytest.approx(1.5 * gap(fig))
    # Facet titles stay on top of their panels.
    tops = sorted(a.y for a in talk.layout.annotations)
    assert tops == pytest.approx(
        sorted([talk.layout.yaxis.domain[1], talk.layout.yaxis2.domain[1]])
    )
    assert gap(scaled_figure(fig, "notebook")) == pytest.approx(gap(fig))


def test_unknown_context_is_rejected():
    with pytest.raises(ValueError, match="poster"):
        figure_pdf(go.Figure(), "huge")


def test_matplotlib_context_is_restored_after_export():
    fig, ax = plt.subplots()
    (line,) = ax.plot([1, 2])
    title = ax.set_title("t")
    before = (title.get_fontsize(), line.get_linewidth())
    assert figure_pdf(fig, "poster").startswith(b"%PDF")
    assert (title.get_fontsize(), line.get_linewidth()) == before
    plt.close(fig)
