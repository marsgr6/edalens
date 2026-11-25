"""Report page: the charts saved with "Add to report", edited and exported."""

import base64
import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from edalens.utils.export import DEFAULT_CONTEXT, ExportError
from edalens.utils.report import COLUMN_CHOICES, PLOTLY, REPORT_DIR_ENV, TABLE, TEXT, Report
from edalens.utils.ui import PDF_CONTEXT_KEY, in_browser, report

PUBLISHED_PARAM = "report"
PUBLISH_DIR_ENV = "EDALENS_PUBLISH_DIR"
MAX_PUBLISHED = 100  # Oldest published reports beyond this are removed from the folder.
_REPORT_ID = re.compile(r"[A-Za-z0-9_-]{8,32}")


def publish_dir() -> Path:
    """Where published reports are kept: $EDALENS_PUBLISH_DIR, else ~/.edalens/published."""
    return Path(os.environ.get(PUBLISH_DIR_ENV) or Path.home() / ".edalens" / "published")


@st.cache_resource
def _published() -> dict:
    """Reports kept in memory when the publish folder cannot be written; lost on restart."""
    return {}


def publish(title: str, page: str, height: int) -> tuple[str, bool]:
    """Keep a report page under a new random id; returns (id, kept on disk)."""
    report_id = secrets.token_urlsafe(9)
    folder = publish_dir()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{report_id}.html").write_text(page, encoding="utf-8")
        (folder / f"{report_id}.json").write_text(json.dumps({"title": title, "height": height}), encoding="utf-8")
    except OSError:
        _published()[report_id] = (title, page, height)
        return report_id, False
    _prune(folder)
    return report_id, True


def _prune(folder: Path) -> None:
    pages = sorted(folder.glob("*.html"), key=lambda path: path.stat().st_mtime)
    for old in pages[:-MAX_PUBLISHED]:
        for path in (old, old.with_suffix(".json")):
            path.unlink(missing_ok=True)


def published(report_id: str):
    """(title, html, height) of a published report, or None."""
    if not _REPORT_ID.fullmatch(report_id or ""):
        return None  # Only ids publish() makes; nothing else can name a file.
    if report_id in _published():
        return _published()[report_id]
    folder = publish_dir()
    try:
        page = (folder / f"{report_id}.html").read_text(encoding="utf-8")
        meta = json.loads((folder / f"{report_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta.get("title", "EDALens report"), page, int(meta.get("height", 800))


def _slug(text: str) -> str:
    return re.sub(r"[^\w-]+", "_", text).strip("_")[:60] or "edalens_report"


def show_published(report_id: str) -> None:
    """The page behind a published link (``?report=<id>``)."""
    entry = published(report_id)
    if entry is None:
        st.error("This report link is not available: it may have been removed, or it was published on another server.")
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


def _add_text() -> None:
    report().add_text("New section", "")


def _item_editor(index: int, item, count: int) -> None:
    with st.container(border=True):
        top = st.columns([0.76, 0.08, 0.08, 0.08], vertical_alignment="bottom")
        title_key, note_key = f"report_title_{item.uid}", f"report_note_{item.uid}"
        st.session_state.setdefault(title_key, item.title)
        st.session_state.setdefault(note_key, item.note)
        label = {TEXT: "Heading", TABLE: "Table"}.get(item.kind, "Chart")
        item.title = top[0].text_input(f"{label} {index + 1}", key=title_key)
        top[1].button("⬆️", key=f"report_up_{item.uid}", help="Move up", disabled=index == 0,
                      on_click=_move, args=(item.uid, -1))
        top[2].button("⬇️", key=f"report_down_{item.uid}", help="Move down", disabled=index == count - 1,
                      on_click=_move, args=(item.uid, 1))
        top[3].button("🗑️", key=f"report_remove_{item.uid}", help="Remove from the report",
                      on_click=_remove, args=(item.uid,))
        item.note = st.text_area("Text" if item.kind == TEXT else "Note (optional)", key=note_key,
                                 height=100 if item.kind == TEXT else 68,
                                 placeholder="A paragraph under the heading." if item.kind == TEXT else
                                 "What does this chart show?")
        if item.kind == TEXT:
            return
        with st.expander("Preview", expanded=False):
            if item.kind == TABLE:
                st.dataframe(item.frame(), hide_index=True)
            elif item.kind == PLOTLY:
                st.plotly_chart(item.plotly_figure(), key=f"report_preview_{item.uid}")
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
                "Visualization or Time Series pages, or under a table on the Statistics page; they collect here.")
        st.button("Add a text block", on_click=_add_text, icon="📝")
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
    st.button("Add a text block", on_click=_add_text, icon="📝",
              help="A heading and a paragraph across the page; move it with ⬆️ ⬇️.")

    st.markdown("#### Export")
    browser = in_browser()
    embed = st.checkbox(
        "Embed plotly.js in the HTML (works offline; adds ~4.5 MB)", value=not browser,
        help="Off: the page loads plotly.js from the Plotly CDN, so it needs internet to show the charts.",
    )
    current.created = datetime.now().isoformat(timespec="minutes")
    file_stem = _slug(current.title)
    actions = st.columns(4)
    build_html = actions[0].button("🌐 Build HTML", width="stretch")
    build_pdf = actions[1].button("📄 Build PDF", width="stretch", disabled=browser,
                                  help="Not available in browser mode (kaleido cannot run there)." if browser else None)
    publish_link = actions[2].button("🔗 Publish link", width="stretch", disabled=browser,
                                help="Needs the app's server; not available in browser mode." if browser else
                                f"A page on this app's server, saved in {publish_dir()} (newest {MAX_PUBLISHED} kept).")
    report_dir = os.environ.get(REPORT_DIR_ENV)
    send = actions[3].button("💾 Send to Python", width="stretch", disabled=not report_dir,
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
    if publish_link:
        report_id, on_disk = publish(current.title, current.to_html(embed_plotly=embed), current.html_height())
        kept = ("it is saved on this server and keeps working after restarts" if on_disk else
                "the publish folder could not be written, so it works only until the app's server restarts")
        st.success(f"Published: [open the report](?{PUBLISHED_PARAM}={report_id}) · {kept}. "
                   "Download the HTML to host it elsewhere.")
        st.code(f"?{PUBLISHED_PARAM}={report_id}", language=None)
    if send:
        path = Report.from_json(current.to_json()).save(
            Path(report_dir) / f"{datetime.now():%Y%m%d-%H%M%S}-{file_stem}.json"
        )
        st.success(f"Saved. In Python: `app.report()` returns it (file: {path.name}).")

    if st.toggle("👁️ Preview the dashboard", value=False):
        components.html(current.to_html(embed_plotly=True), height=current.html_height(), scrolling=True)
