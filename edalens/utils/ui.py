"""Small UI helpers shared by the app's pages."""

import sys
from typing import Optional, Sequence

import re
import streamlit as st

from edalens.utils.export import CONTEXTS, DEFAULT_CONTEXT, ExportError, figure_pdf

# Formats the chart toolbar's camera button can save (Plotly.js, in the browser).
IMAGE_FORMATS = ("png", "svg", "jpeg", "webp")
IMAGE_FORMAT_KEY = "chart_image_format"
PDF_CONTEXT_KEY = "chart_pdf_context"
_CHART_COUNT_KEY = "_chart_count"


def in_browser() -> bool:
    """True under Pyodide (stlite, ``edalens.walk(mode="browser")``).

    stlite cannot serve a custom component's frontend files, so option_menu
    never shows there.
    """
    return sys.platform == "emscripten"


def sidebar_menu(
    title: str,
    options: Sequence[str],
    icons: Sequence[str],
    menu_icon: Optional[str] = None,
    key: Optional[str] = None,
) -> str:
    """A navigation menu: option_menu on a server, a plain radio in the browser.

    Call it inside ``with st.sidebar:``. Returns the selected option.
    """
    if in_browser():
        st.markdown(f"### {title}")
        return st.radio(title, list(options), label_visibility="collapsed", key=key)
    from streamlit_option_menu import option_menu

    return option_menu(
        menu_title=title,
        options=list(options),
        icons=list(icons),
        menu_icon=menu_icon,
        default_index=0,
        key=key,
    )


def export_settings() -> None:
    """Sidebar controls: the camera button's format and the PDF context."""
    with st.expander("📷 Chart export", expanded=False):
        st.selectbox(
            "Camera button saves as",
            IMAGE_FORMATS,
            key=IMAGE_FORMAT_KEY,
            help="The camera icon in each chart's toolbar. SVG is vector, like PDF.",
        )
        if not in_browser():
            st.selectbox(
                "PDF text and line size",
                list(CONTEXTS),
                index=list(CONTEXTS).index(DEFAULT_CONTEXT),
                key=PDF_CONTEXT_KEY,
                help="Like seaborn's contexts: paper 0.8x, notebook 1x, talk 1.5x, poster 2x "
                "for axis text, titles, legends, lines and markers.",
            )


def start_charts() -> None:
    """Call at the top of a page: numbers its charts so their buttons keep keys across reruns."""
    st.session_state[_CHART_COUNT_KEY] = 0


def _chart_key() -> str:
    count = st.session_state.get(_CHART_COUNT_KEY, 0) + 1
    st.session_state[_CHART_COUNT_KEY] = count
    return f"chart_{count}"


def _file_name(fig, default: str) -> str:
    title = ""
    if hasattr(fig, "layout"):
        title = fig.layout.title.text or ""
    elif fig.axes:
        title = fig.axes[0].get_title()
    return (re.sub(r"[^\w-]+", "_", title).strip("_")[:60] or default) + ".pdf"


def _pdf_download(fig, key: str) -> None:
    """A "Prepare PDF" button; once clicked, the download of that chart.

    The PDF is rendered only on request (kaleido takes a few seconds). Not shown in
    the browser (stlite), where kaleido cannot run and downloads do not work.
    """
    if in_browser():
        return
    if not st.button("📄 Prepare PDF", key=f"{key}_pdf", type="tertiary"):
        return
    try:
        with st.spinner("Rendering PDF..."):
            data = figure_pdf(fig, st.session_state.get(PDF_CONTEXT_KEY, DEFAULT_CONTEXT))
    except ExportError as exc:
        st.warning(str(exc))
        return
    st.download_button(
        "⬇️ Download PDF",
        data,
        file_name=_file_name(fig, key),
        mime="application/pdf",
        key=f"{key}_pdf_download",
        on_click="ignore",
    )


def plotly_chart(fig, **kwargs) -> None:
    """``st.plotly_chart`` with the chosen camera format and a PDF download."""
    image_format = st.session_state.get(IMAGE_FORMAT_KEY, "png")
    config = {
        "toImageButtonOptions": {
            "format": image_format,
            "filename": "edalens_chart",
            "scale": 1 if image_format == "svg" else 2,
        }
    }
    key = _chart_key()
    st.plotly_chart(fig, config=config, key=key, **kwargs)
    _pdf_download(fig, key)


def pyplot(fig, **kwargs) -> None:
    """``st.pyplot`` with a PDF download."""
    key = _chart_key()
    st.pyplot(fig, **kwargs)
    _pdf_download(fig, key)
