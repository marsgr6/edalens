"""Run EDALens entirely in the notebook's browser, with no server or tunnel::

    import edalens
    app = edalens.walk(df, mode="browser")

Like pygwalker, everything happens in the cell output. The app's own Python
files and the DataFrame are embedded in the output HTML, and stlite
(Streamlit compiled to WebAssembly with Pyodide) runs them inside the browser.
Nothing listens on a port, so it works wherever notebook output can run
JavaScript and load stlite from jsDelivr: Colab, JupyterHub, remote kernels,
local Jupyter and VS Code.

Trade-offs against the server mode: the first start downloads Pyodide and the
packages (tens of MB, cached by the browser afterwards) and takes 20-60 s;
Prophet's Stan engine cannot run in WebAssembly, so Prophet sections use a numpy
version of its model (``edalens.utils.prophet_lite``); and the
DataFrame travels inside the notebook, so very large frames make the output
(and a saved notebook) large.
"""

from __future__ import annotations

import html
import json
import secrets
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from edalens.launcher import app_script

# Pinned: stlite's mount API changes between releases.
STLITE_VERSION = "1.11.0"
STLITE_URL = f"https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build"
# Runtime dependencies that Pyodide can install. prophet has no WebAssembly build
# (edalens.utils.prophet_lite stands in for it);
# plotly-calplot pins numpy<2 and the app does not import it.
BROWSER_REQUIREMENTS = [
    "matplotlib",
    "missingno",
    "plotly",
    "plotly-calheatmap",
    "requests",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "streamlit-option-menu",  # Imported, though the app uses a radio menu here.
]
DATA_FILE = "edalens_data.json"
ENTRYPOINT = "edalens_browser_app.py"
# Same name as edalens.utils.data_processing.PRESET_DATA_ENV (see edalens.notebook).
DATA_ENTRY_SOURCE = f"""import os

os.environ["EDALENS_DATA"] = {DATA_FILE!r}

from edalens.app import main

main()
"""
ENTRY_SOURCE = """from edalens.app import main

main()
"""
DEFAULT_HEIGHT = 900
LARGE_OUTPUT_BYTES = 50_000_000


def app_files() -> Dict[str, str]:
    """Source of the routed app, keyed by its path in the browser's file system."""
    package = app_script().parent
    paths = [package / "app.py"]
    for subpackage in ("views", "utils"):
        paths += sorted((package / subpackage).glob("*.py"))
    files = {
        path.relative_to(package.parent).as_posix(): path.read_text(encoding="utf-8")
        for path in paths
    }
    # The real edalens/__init__.py imports the launcher and notebook helpers,
    # which the browser neither needs nor can run.
    files["edalens/__init__.py"] = ""
    return files


def frame_json(df: pd.DataFrame) -> str:
    """Serialize ``df`` with its dtypes in a format Pyodide's pandas can read."""
    # orient="table" keeps dtypes (including categories and datetimes) and needs
    # unique string column names.
    frame = df.reset_index(drop=True)
    frame.columns = [str(column) for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("walk(mode='browser') needs unique column names.")
    return frame.to_json(orient="table", index=False, date_format="iso")


def _script_json(value: object) -> str:
    # "</" would end the inline <script> early when the page is parsed.
    return json.dumps(value).replace("</", "<\\/")


def page_html(df: Optional[pd.DataFrame]) -> str:
    """A standalone page that mounts EDALens with stlite, ``df`` preloaded."""
    files = dict(app_files())
    if df is not None:
        files[DATA_FILE] = frame_json(df)
        files[ENTRYPOINT] = DATA_ENTRY_SOURCE
    else:
        files[ENTRYPOINT] = ENTRY_SOURCE
    options = {
        "requirements": BROWSER_REQUIREMENTS,
        "entrypoint": ENTRYPOINT,
        "files": files,
        "streamlitConfig": {"client.toolbarMode": "viewer"},
    }
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="{STLITE_URL}/stlite.css">
<style>html, body {{ margin: 0; height: 100%; }}</style>
</head>
<body>
<div id="root"></div>
<script type="module">
import {{ mount }} from "{STLITE_URL}/stlite.js";
mount({_script_json(options)}, document.getElementById("root"));
</script>
</body>
</html>"""


class BrowserApp:
    """EDALens running in the notebook's browser; there is no server to stop."""

    def __init__(
        self, df: Optional[pd.DataFrame], height: int = DEFAULT_HEIGHT
    ) -> None:
        self.page = page_html(df)
        self.height = height
        self._id = f"edalens-{secrets.token_hex(4)}"

    def iframe_html(self, height: Optional[int] = None) -> str:
        # srcdoc keeps stlite's global styles out of the notebook page.
        return (
            f'<iframe id="{self._id}" srcdoc="{html.escape(self.page, quote=True)}" '
            f'width="100%" height="{height or self.height}" '
            'style="border: 1px solid #ddd; border-radius: 4px;" '
            'allow="clipboard-write"></iframe>'
            '<div style="font-size: 0.85em;">EDALens runs in your browser (stlite); '
            "the first start downloads Python packages and takes 20-60 s. "
            "Prophet sections use a numpy version of Prophet's model.</div>"
        )

    def display(self, height: Optional[int] = None) -> None:
        """Show the app (again) in the current cell's output."""
        from IPython.display import HTML, display

        display(HTML(self.iframe_html(height)))

    def report(self):
        """Not available: the browser-mode app has no way to reach this Python session."""
        raise RuntimeError(
            "In browser mode the app runs in your browser and cannot send reports to Python. "
            "On its Report page, use 🌐 Build HTML to download the dashboard, or start the app "
            "with edalens.walk(df) (server mode) to use app.report()."
        )

    reports = report

    def save(self, path: str | Path) -> Path:
        """Write the app as a standalone HTML file that opens in any browser."""
        path = Path(path)
        path.write_text(self.page, encoding="utf-8")
        return path

    def stop(self) -> None:
        """Nothing runs outside the browser; clear the cell output to close it."""

    def __repr__(self) -> str:
        size = len(self.page) / 1e6
        return f"<EDALens in-browser app ({size:.1f} MB page)>"


def walk_browser(
    df: Optional[pd.DataFrame], *, height: int = DEFAULT_HEIGHT, display: bool = True
) -> BrowserApp:
    app = BrowserApp(df, height)
    if len(app.page) > LARGE_OUTPUT_BYTES:
        print(
            f"Warning: the embedded data makes this output {len(app.page) / 1e6:.0f} MB; "
            "consider a sample (df.sample(...)) or mode='server'."
        )
    if display and _in_notebook():
        app.display()
    elif display:
        print("Not in a notebook, so nothing is shown: use .save('page.html'), or edalens.show(df, mode='browser').")
    return app


def _in_notebook() -> bool:
    from edalens.notebook import _in_notebook as in_notebook

    return in_notebook()


PAGE_NAME = "index.html"


class BrowserPageServer:
    """A browser-mode page served on 127.0.0.1, like ``python -m http.server``.

    The page only needs a web address: Python runs in the browser tab (stlite),
    so this server just hands out one HTML file. ``edalens.show(df, mode="browser")``
    and ``edalens --browser`` start it.
    """

    def __init__(self, df: Optional[pd.DataFrame], port: int) -> None:
        import tempfile

        self.port = port
        self._folder = tempfile.TemporaryDirectory(prefix="edalens-page-")
        self.path = Path(self._folder.name) / PAGE_NAME
        BrowserApp(df).save(self.path)
        self.url = f"http://127.0.0.1:{port}/{PAGE_NAME}"
        self._server = None
        self._thread = None

    def start(self) -> "BrowserPageServer":
        import functools
        import threading
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        class Quiet(SimpleHTTPRequestHandler):
            def log_message(self, *args) -> None:  # No request log in the terminal.
                pass

        handler = functools.partial(Quiet, directory=self._folder.name)
        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        except OSError as exc:
            from edalens.launcher import LaunchError

            self._folder.cleanup()
            raise LaunchError(f"Port {self.port} is not available: {exc}") from exc
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self) -> int:
        """Block until stopped (Ctrl+C in the terminal)."""
        import time

        while self.running:
            time.sleep(0.5)
        return 0

    def stop(self) -> None:
        """Stop serving and delete the page; safe to call repeatedly."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._folder.cleanup()

    def __enter__(self) -> "BrowserPageServer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"<EDALens browser-mode page {self.url} ({state})>"

    def _repr_html_(self) -> str:
        state = "running" if self.running else "stopped"
        return f'EDALens browser-mode page <a href="{self.url}" target="_blank">{self.url}</a> ({state})'


def serve_browser(df: Optional[pd.DataFrame], port: int) -> BrowserPageServer:
    return BrowserPageServer(df, port).start()
