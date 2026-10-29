"""Show EDAViz inside a Jupyter notebook cell, pygwalker style::

    import edaviz
    app = edaviz.walk(df)   # the app appears in the cell output with df loaded
    app.stop()              # stops the app and deletes the temporary data file

The app runs as a local Streamlit server (see ``edaviz.launcher``) and is shown
in an iframe, so the notebook's browser must reach ``127.0.0.1`` on the kernel
machine: local Jupyter, JupyterLab and VS Code work. In Colab use
``edaviz.run_colab()`` instead.

The DataFrame is pickled to a private temporary file whose path is passed to
the app in ``EDAVIZ_DATA``; every browser session of that app starts with it.
"""

from __future__ import annotations

import atexit
import html
import os
from pathlib import Path
import tempfile
import weakref
from typing import Optional

import pandas as pd

from edaviz.launcher import STARTUP_TIMEOUT, EDAVizServer, LaunchError, choose_port
DEFAULT_HEIGHT = 900
# Same name as utils.data_processing.PRESET_DATA_ENV; not imported so that
# ``import edaviz`` does not load the app's analysis dependencies.
PRESET_DATA_ENV = "EDAVIZ_DATA"

_running: "weakref.WeakSet[NotebookApp]" = weakref.WeakSet()


class NotebookApp:
    """An EDAViz server embedded in notebook output, plus its preloaded data file."""

    def __init__(self, server: EDAVizServer, data_path: Optional[Path], height: int) -> None:
        self.server = server
        self.data_path = data_path
        self.height = height
        _running.add(self)

    @property
    def url(self) -> str:
        return self.server.url

    @property
    def running(self) -> bool:
        return self.server.running

    def iframe_html(self, height: Optional[int] = None) -> str:
        url = html.escape(self.url, quote=True)
        return (
            f'<iframe src="{url}" width="100%" height="{height or self.height}" '
            'style="border: 1px solid #ddd; border-radius: 4px;" '
            'allow="clipboard-write"></iframe>'
            f'<div style="font-size: 0.85em;">EDAViz is running at '
            f'<a href="{url}" target="_blank">{url}</a> (opens in a new tab). '
            "Call <code>.stop()</code> on the returned object to shut it down.</div>"
        )

    def display(self, height: Optional[int] = None) -> None:
        """Show the app (again) in the current cell's output."""
        from IPython.display import HTML, display

        display(HTML(self.iframe_html(height)))

    def stop(self) -> None:
        """Stop the app and delete the data file; safe to call repeatedly."""
        self.server.stop()
        if self.data_path is not None:
            try:
                self.data_path.unlink()
            except OSError:
                pass
            self.data_path = None
        _running.discard(self)

    def __enter__(self) -> "NotebookApp":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"<EDAViz notebook app {self.url} ({state})>"


def _write_data(df: pd.DataFrame) -> Path:
    # mkstemp creates the file readable by the current user only.
    fd, name = tempfile.mkstemp(prefix="edaviz-data-", suffix=".pkl")
    os.close(fd)
    path = Path(name)
    try:
        df.to_pickle(path)
    except BaseException:
        path.unlink()
        raise
    return path


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and getattr(shell, "kernel", None) is not None


def walk(
    df: Optional[pd.DataFrame | pd.Series] = None,
    *,
    height: int = DEFAULT_HEIGHT,
    port: Optional[int] = None,
    startup_timeout: float = STARTUP_TIMEOUT,
    display: bool = True,
) -> NotebookApp:
    """Start EDAViz with ``df`` preloaded and show it in the notebook output.

    ``df=None`` opens the app on its data-loading page. ``display=False`` only
    starts the server; call ``.display()`` on the result to show it later.
    Each call starts a separate server; call ``.stop()`` to free it.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if df is not None and not isinstance(df, pd.DataFrame):
        raise TypeError(f"walk() expects a pandas DataFrame or Series, not {type(df).__name__}.")
    data_path = _write_data(df) if df is not None else None
    env = {PRESET_DATA_ENV: str(data_path)} if data_path is not None else {}
    try:
        server = EDAVizServer(choose_port(port), env=env).start(startup_timeout)
    except BaseException:
        if data_path is not None:
            data_path.unlink()
        raise
    app = NotebookApp(server, data_path, height)
    if display and _in_notebook():
        app.display()
    elif display:
        print(f"EDAViz is running at {app.url} (not in a notebook, so nothing is embedded).")
    return app


@atexit.register
def _stop_all() -> None:
    for app in list(_running):
        app.stop()


__all__ = ["LaunchError", "NotebookApp", "walk"]
