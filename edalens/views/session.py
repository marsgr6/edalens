"""Sidebar panel: save the session to a file, or open a saved one."""

import base64
from datetime import datetime

import streamlit as st

from edalens.utils.session import SessionError, export_session, restore_session
from edalens.utils.ui import in_browser

OPENED_KEY = "_session_opened"  # The uploaded file already restored, so reruns do not repeat it.


def _save_link(text: str) -> None:
    name = f"edalens-session-{datetime.now():%Y%m%d-%H%M}.json"
    if in_browser():  # stlite cannot serve st.download_button files.
        link = base64.b64encode(text.encode()).decode()
        st.markdown(f'<a download="{name}" href="data:application/json;base64,{link}">⬇️ Save session</a>',
                    unsafe_allow_html=True)
    else:
        st.download_button("⬇️ Save session", text, file_name=name, mime="application/json",
                           on_click="ignore", width="stretch")


def session_panel() -> None:
    with st.expander("💾 Session", expanded=False):
        st.caption("Save the data, every cleaning step, the plot settings, category orders and the "
                   "report in one file, and open it later to continue where you left off.")
        if "raw_df" in st.session_state:
            include = st.checkbox("Include the data", value=True, key="session_include_data",
                                  help="Without it the file is small, but you must load the same data before opening it.")
            # Built on request: serializing big data on every rerun would slow every page down.
            if st.button("Prepare session file", key="session_prepare", width="stretch"):
                _save_link(export_session(st.session_state, include_data=include))
        upload = st.file_uploader("Open a session", type=["json"], key="session_upload")
        if upload is None:
            st.session_state.pop(OPENED_KEY, None)
            return
        if st.session_state.get(OPENED_KEY) == upload.file_id:
            return
        try:
            working = restore_session(st.session_state, upload.getvalue())
        except SessionError as exc:
            st.error(str(exc))
            return
        st.session_state[OPENED_KEY] = upload.file_id
        steps = len(st.session_state.get("operations", []))
        st.success(f"Session opened: {len(working)} rows, {steps} step{'s' if steps != 1 else ''} replayed.")
        st.rerun()
