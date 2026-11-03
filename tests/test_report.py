"""Reports: charts saved from the app, exported as HTML/PDF and handed back to Python."""

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from streamlit.testing.v1 import AppTest

from edalens.utils import report as report_module
from edalens.utils.export import ExportError
from edalens.utils.report import IMAGE, PLOTLY, REPORT_DIR_ENV, Report

APP = str(Path(__file__).resolve().parents[1] / "edalens" / "app.py")


def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"day": rng.choice(["Thu", "Fri"], 40), "tip": rng.normal(3, 1, 40)}
    )


def sample_report() -> Report:
    report = Report("Tips <review>", columns=2)
    report.add(
        px.box(frame(), x="day", y="tip", title="Tips by day"), note="Fri < Thu?"
    )
    fig, ax = plt.subplots()
    ax.plot([1, 3, 2])
    ax.set_title("A matplotlib chart")
    report.add(fig)
    plt.close(fig)
    return report


def test_items_keep_titles_kinds_and_order():
    report = sample_report()
    assert [(i.kind, i.title) for i in report.items] == [
        (PLOTLY, "Tips by day"),
        (IMAGE, "A matplotlib chart"),
    ]
    first = report.items[0].uid
    report.move(0, 1)
    assert report.items[1].uid == first
    report.move(1, 1)  # Already last: no change.
    assert report.items[1].uid == first
    report.remove(report.index_of(first))
    assert [i.title for i in report.items] == ["A matplotlib chart"]


def test_json_round_trip(tmp_path):
    report = sample_report()
    assert Report.load(report.save(tmp_path / "r.json")) == report


def test_html_is_a_dashboard_with_escaped_text():
    report = sample_report()
    page = report.to_html(embed_plotly=False)
    assert "<title>Tips &lt;review&gt;</title>" in page
    assert "repeat(2, minmax(0, 1fr))" in page
    assert page.count('class="card"') == 2
    assert "Fri &lt; Thu?" in page and "<svg" in page
    cdn_tag = f'<script src="{report_module.PLOTLY_CDN}">'
    assert cdn_tag in page
    # Embedded: plotly.js inline instead (the bundle itself mentions the CDN host).
    # Booleans only, so a failure does not make pytest diff megabytes of text.
    embedded = report.to_html(embed_plotly=True)
    assert (cdn_tag in embedded, len(embedded) > 1_000_000) == (False, True)


def test_pdf_joins_a_contents_page_and_one_page_per_chart(monkeypatch):
    pypdf = pytest.importorskip("pypdf")

    def blank_pdfs(figs, context):
        pages = []
        for _ in figs:
            fig = plt.figure()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="pdf")
            plt.close(fig)
            pages.append(buffer.getvalue())
        return pages

    monkeypatch.setattr(report_module, "figures_pdf", blank_pdfs)
    pdf = sample_report().to_pdf("talk")
    assert len(pypdf.PdfReader(io.BytesIO(pdf)).pages) == 3


def test_pdf_explains_missing_pypdf_and_empty_reports(monkeypatch):
    with pytest.raises(ExportError, match="empty"):
        Report().to_pdf()
    monkeypatch.setitem(sys.modules, "pypdf", None)
    with pytest.raises(ExportError, match="pip install pypdf"):
        sample_report().to_pdf()


def test_pages_carry_title_and_note():
    report = sample_report()
    page = report._page(report.items[0])
    assert page.layout.title.text == "Tips by day<br><sup>Fri &lt; Thu?</sup>"
    image_page = report._page(report.items[1])
    assert image_page.layout.images[0].source.startswith("data:image/png;base64,")


# -- In the app --------------------------------------------------------------------


def app_with_data(monkeypatch) -> AppTest:
    monkeypatch.setattr(
        sys, "platform", "emscripten"
    )  # Radio menu, drivable in AppTest.
    app = AppTest.from_file(APP, default_timeout=30)
    data = frame()
    app.session_state["filtered_df"] = data
    app.session_state["operations"] = []
    app.session_state["current_numeric_cols"] = ["tip"]
    app.session_state["current_categorical_cols"] = ["day"]
    app.session_state["viz_state"] = {
        "plot_type": "boxes",
        "x": "day",
        "y": "tip",
        "color": "None",
        "facet_col": "None",
        "facet_row": "None",
        "size": "None",
    }
    app.session_state["_plot_type_selector"] = "boxes"
    return app


def go_to(app: AppTest, page: str) -> None:
    app.sidebar.radio[0].set_value(page)
    app.run()


def button(app: AppTest, label: str):
    return next(b for b in app.button if b.label == label)


def test_add_to_report_then_export_html_and_publish(monkeypatch):
    app = app_with_data(monkeypatch)
    app.run()
    go_to(app, "Report")
    assert any("report is empty" in i.value for i in app.info)
    go_to(app, "Visualization")
    button(app, "➕ Add to report").click()
    app.run()
    assert not app.exception
    assert len(app.session_state["report"].items) == 1

    go_to(app, "Report")
    assert not app.exception
    assert next(t for t in app.text_input if t.label == "Chart 1").value
    next(t for t in app.text_area if t.label == "Note (optional)").set_value(
        "Thursday wins."
    )
    app.run()
    assert app.session_state["report"].items[0].note == "Thursday wins."

    # Browser mode: Build HTML gives a data: link; publishing needs a server.
    assert button(app, "🔗 Publish link").disabled
    button(app, "🌐 Build HTML").click()
    app.run()
    assert any("data:text/html;base64," in m.value for m in app.markdown)


def test_published_link_shows_the_report(monkeypatch):
    from edalens.views import report as report_view

    report_view._published.clear()
    app = app_with_data(monkeypatch)
    monkeypatch.setattr(report_view, "in_browser", lambda: False)
    app.session_state["report"] = sample_report()
    app.run()
    go_to(app, "Report")
    button(app, "🔗 Publish link").click()
    app.run()
    assert not app.exception
    (report_id,) = report_view._published()
    viewer = AppTest.from_file(APP, default_timeout=30)
    viewer.query_params["report"] = report_id
    viewer.run()
    assert not viewer.exception
    assert viewer.get("download_button")


def test_send_to_python_saves_a_report_for_the_launcher(monkeypatch, tmp_path):
    monkeypatch.setenv(REPORT_DIR_ENV, str(tmp_path))
    app = app_with_data(monkeypatch)
    app.session_state["report"] = sample_report()
    app.run()
    go_to(app, "Report")
    button(app, "💾 Send to Python").click()
    app.run()
    assert not app.exception

    from edalens.launcher import EDALensServer

    server = EDALensServer(8599)
    server.report_dir = tmp_path
    assert server.report().title == "Tips <review>"
    assert [r.title for r in server.reports()] == ["Tips <review>"]


def test_report_before_anything_was_sent_explains_how():
    from edalens.launcher import EDALensServer

    with pytest.raises(LookupError, match="Send to Python"):
        EDALensServer(8599).report()
