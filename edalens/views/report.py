"""Report page: the charts saved with "Add to report", edited and exported."""

import base64
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from edalens.utils.export import DEFAULT_CONTEXT, ExportError
from edalens.utils.report import COLUMN_CHOICES, PLOTLY, REPORT_DIR_ENV, Report
from edalens.utils.ui import PDF_CONTEXT_KEY, in_browser, report

PUBLISHED_PARAM = "report"


@st.cache_resource
def _published() -> dict:
    """Reports published on this server: id -> (title, html, height). Shared by all sessions."""
    return {}


def _slug(text: str) -> str:
    return re.sub(r"[^\w-]+", "_", text).strip("_")[:60] or "edalens_report"


def show_published(report_id: str) -> None:
    """The page behind a published link (``?report=<id>``)."""
    entry = _published().get(report_id)
    if entry is None:
        st.error("This report link is not available. Published reports last until the app's server restarts.")
        return
    title, page, height = entry
    st.download_button("⬇️ Download HTML", page, file_name=f"{_slug(title)}.html", mime="text/html", on_click="ignore")
    components.html(page, height=height, scrolling=True)


def _move(uid: str, step: int) -> None:
    current = report()
    current.move(current.index_of(uid), step)


def _remove(uid: str) -> None:
    current = report()
    current.remove(current.index_of(uid))


def _clear() -> None:
    report().items.clear()


def _item_editor(index: int, item, count: int) -> None:
    with st.container(border=True):
        top = st.columns([0.76, 0.08, 0.08, 0.08], vertical_alignment="bottom")
        title_key, note_key = f"report_title_{item.uid}", f"report_note_{item.uid}"
        st.session_state.setdefault(title_key, item.title)
        st.session_state.setdefault(note_key, item.note)
        item.title = top[0].text_input(f"Chart {index + 1}", key=title_key)
        top[1].button("⬆️", key=f"report_up_{item.uid}", help="Move up", disabled=index == 0,
                      on_click=_move, args=(item.uid, -1))
        top[2].button("⬇️", key=f"report_down_{item.uid}", help="Move down", disabled=index == count - 1,
                      on_click=_move, args=(item.uid, 1))
        top[3].button("🗑️", key=f"report_remove_{item.uid}", help="Remove from the report",
                      on_click=_remove, args=(item.uid,))
        item.note = st.text_area("Note (optional)", key=note_key, height=68,
                                 placeholder="What does this chart show?")
        with st.expander("Preview", expanded=False):
            if item.kind == PLOTLY:
                st.plotly_chart(item.plotly_figure(), key=f"report_preview_{item.uid}", use_container_width=True)
            else:
                st.image(base64.b64decode(item.png))


def _browser_download(page: str, file_name: str) -> None:
    """stlite cannot serve st.download_button files; a data: link downloads in most browsers."""
    data = base64.b64encode(page.encode()).decode()
    st.markdown(
        f'<a download="{file_name}" href="data:text/html;base64,{data}">⬇️ Download {file_name}</a>',
        unsafe_allow_html=True,
    )
    st.caption("If the download is blocked (some notebook frames block it), use the preview below.")


def report_page() -> None:
    current = report()
    st.subheader("🧾 Report")
    if not current.items:
        st.info("The report is empty. Click **➕ Add to report** under any chart on the "
                "Visualization or Time Series pages; the charts collect here.")
        return

    settings = st.columns([0.5, 0.25, 0.25], vertical_alignment="bottom")
    st.session_state.setdefault("report_name", current.title)
    current.title = settings[0].text_input("Report title", key="report_name")
    st.session_state.setdefault("report_columns", current.columns)
    current.columns = settings[1].radio("Dashboard columns", COLUMN_CHOICES, horizontal=True, key="report_columns")
    settings[2].button("Clear report", on_click=_clear, type="tertiary", icon="🧹")

    count = len(current.items)
    for index, item in enumerate(list(current.items)):
        _item_editor(index, item, count)

    st.markdown("#### Export")
    browser = in_browser()
    embed = st.checkbox(
        "Embed plotly.js in the HTML (works offline; adds ~4.5 MB)", value=not browser,
        help="Off: the page loads plotly.js from the Plotly CDN, so it needs internet to show the charts.",
    )
    current.created = datetime.now().isoformat(timespec="minutes")
    file_stem = _slug(current.title)
    actions = st.columns(4)
    build_html = actions[0].button("🌐 Build HTML", use_container_width=True)
    build_pdf = actions[1].button("📄 Build PDF", use_container_width=True, disabled=browser,
                                  help="Not available in browser mode (kaleido cannot run there)." if browser else None)
    publish = actions[2].button("🔗 Publish link", use_container_width=True, disabled=browser,
                                help="Needs the app's server; not available in browser mode." if browser else
                                "A page on this app's server, until it restarts.")
    report_dir = os.environ.get(REPORT_DIR_ENV)
    send = actions[3].button("💾 Send to Python", use_container_width=True, disabled=not report_dir,
                             help="Saves the report for app.report() in the notebook or script that "
                             "started the app." if report_dir else
                             "Only when the app was started from Python (edalens.walk or edalens.show).")
    if not browser:
        context = st.session_state.get(PDF_CONTEXT_KEY, DEFAULT_CONTEXT)
        st.caption(f"PDF size: **{context}** (change it in the sidebar under 📷 Chart export).")

    if build_html:
        page = current.to_html(embed_plotly=embed)
        if browser:
            _browser_download(page, f"{file_stem}.html")
        else:
            st.download_button("⬇️ Download HTML", page, file_name=f"{file_stem}.html", mime="text/html",
                               on_click="ignore", type="primary")
    if build_pdf:
        try:
            with st.spinner(f"Rendering {count + 1} PDF pages..."):
                pdf = current.to_pdf(context)
        except ExportError as exc:
            st.warning(str(exc))
        else:
            st.download_button("⬇️ Download PDF", pdf, file_name=f"{file_stem}.pdf", mime="application/pdf",
                               on_click="ignore", type="primary")
    if publish:
        report_id = secrets.token_urlsafe(9)
        _published()[report_id] = (current.title, current.to_html(embed_plotly=embed), current.html_height())
        st.success(f"Published: [open the report](?{PUBLISHED_PARAM}={report_id}) · this link works while the "
                   "app's server runs. Download the HTML to keep it or host it elsewhere.")
        st.code(f"?{PUBLISHED_PARAM}={report_id}", language=None)
    if send:
        path = Report.from_json(current.to_json()).save(
            Path(report_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{file_stem}.json"
        )
        st.success(f"Saved. In Python: `app.report()` returns it (file: {path.name}).")

    if st.toggle("👁️ Preview the dashboard", value=False):
        components.html(current.to_html(embed_plotly=True), height=current.html_height(), scrolling=True)
