"""Run EDAViz entirely in the notebook's browser, with no server or tunnel::

    import edaviz
    app = edaviz.walk(df, mode="browser")

Like pygwalker, everything happens in the cell output. The app's own Python
files and the DataFrame are embedded in the output HTML, and stlite
(Streamlit compiled to WebAssembly with Pyodide) runs them inside the browser.
Nothing listens on a port, so it works wherever notebook output can run
JavaScript and load stlite from jsDelivr: Colab, JupyterHub, remote kernels,
local Jupyter and VS Code.

Trade-offs against the server mode: the first start downloads Pyodide and the
packages (tens of MB, cached by the browser afterwards) and takes 20-60 s;
Prophet cannot run in WebAssembly, so forecasting is unavailable; and the
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

from edaviz.launcher import app_script

# Pinned: stlite's mount API changes between releases.
STLITE_VERSION = "1.11.0"
STLITE_URL = f"https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build"
# Runtime dependencies that Pyodide can install. prophet has no WebAssembly build;
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
DATA_FILE = "edaviz_data.json"
ENTRYPOINT = "edaviz_browser_app.py"
# Same name as utils.data_processing.PRESET_DATA_ENV (see edaviz.notebook).
ENTRY_SOURCE = f"""import os

os.environ["EDAVIZ_DATA"] = {DATA_FILE!r}

import my_viz_plotly

my_viz_plotly.main()
"""
DEFAULT_HEIGHT = 900
LARGE_OUTPUT_BYTES = 50_000_000


def app_files() -> Dict[str, str]:
    """Source of the routed app, keyed by its path relative to the app root."""
    root = app_script().parent
    paths = [root / "my_viz_plotly.py"]
    for package in ("views", "utils"):
        paths += sorted((root / package).glob("*.py"))
    return {path.relative_to(root).as_posix(): path.read_text(encoding="utf-8") for path in paths}


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
    """A standalone page that mounts EDAViz with stlite, ``df`` preloaded."""
    files = dict(app_files())
    if df is not None:
        files[DATA_FILE] = frame_json(df)
        files[ENTRYPOINT] = ENTRY_SOURCE
        entrypoint = ENTRYPOINT
    else:
        entrypoint = "my_viz_plotly.py"
    options = {
        "requirements": BROWSER_REQUIREMENTS,
        "entrypoint": entrypoint,
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
    """EDAViz running in the notebook's browser; there is no server to stop."""

    def __init__(self, df: Optional[pd.DataFrame], height: int = DEFAULT_HEIGHT) -> None:
        self.page = page_html(df)
        self.height = height
        self._id = f"edaviz-{secrets.token_hex(4)}"

    def iframe_html(self, height: Optional[int] = None) -> str:
        # srcdoc keeps stlite's global styles out of the notebook page.
        return (
            f'<iframe id="{self._id}" srcdoc="{html.escape(self.page, quote=True)}" '
            f'width="100%" height="{height or self.height}" '
            'style="border: 1px solid #ddd; border-radius: 4px;" '
            'allow="clipboard-write"></iframe>'
            '<div style="font-size: 0.85em;">EDAViz runs in your browser (stlite); '
            "the first start downloads Python packages and takes 20-60 s. "
            "Forecasting with Prophet is not available in this mode.</div>"
        )

    def display(self, height: Optional[int] = None) -> None:
        """Show the app (again) in the current cell's output."""
        from IPython.display import HTML, display

        display(HTML(self.iframe_html(height)))

    def save(self, path: str | Path) -> Path:
        """Write the app as a standalone HTML file that opens in any browser."""
        path = Path(path)
        path.write_text(self.page, encoding="utf-8")
        return path

    def stop(self) -> None:
        """Nothing runs outside the browser; clear the cell output to close it."""

    def __repr__(self) -> str:
        size = len(self.page) / 1e6
        return f"<EDAViz in-browser app ({size:.1f} MB page)>"


def walk_browser(
    df: Optional[pd.DataFrame], *, height: int = DEFAULT_HEIGHT, display: bool = True
) -> BrowserApp:
    app = BrowserApp(df, height)
    if len(app.page) > LARGE_OUTPUT_BYTES:
        print(
            f"Warning: the embedded data makes this output {len(app.page) / 1e6:.0f} MB; "
            "consider a sample (df.sample(...)) or mode='server'."
        )
    if display:
        app.display()
    return app
