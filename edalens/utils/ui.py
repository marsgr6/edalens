"""Small UI helpers shared by the app's pages."""

import sys
from typing import Optional, Sequence

import streamlit as st


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
