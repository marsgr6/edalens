"""Local launcher for the EDAViz Streamlit app.

>>> import edaviz
>>> app = edaviz.show()  # returns once the server is ready; does not block
>>> app.url
>>> app.stop()

In Jupyter, ``edaviz.walk(df)`` shows the app inside the notebook with ``df`` loaded;
``edaviz.walk(df, mode="browser")`` runs it in the browser with no server (pygwalker style).
In Google Colab, ``edaviz.run_colab()`` also opens a public localtunnel link.
"""

from importlib.metadata import PackageNotFoundError, version

from edaviz.browser import BrowserApp
from edaviz.colab import ColabApp, run_colab
from edaviz.launcher import EDAVizServer, LaunchError, show
from edaviz.notebook import NotebookApp, walk

try:
    __version__ = version("edaviz")
except PackageNotFoundError:  # Imported from a checkout without installation.
    __version__ = "0+unknown"

__all__ = [
    "BrowserApp",
    "ColabApp",
    "EDAVizServer",
    "LaunchError",
    "NotebookApp",
    "run_colab",
    "show",
    "walk",
    "__version__",
]
