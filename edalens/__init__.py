"""Local launcher for the EDALens Streamlit app.

>>> import edalens
>>> app = edalens.show()  # returns once the server is ready; does not block
>>> app.url
>>> app.stop()

In Jupyter, ``edalens.walk(df)`` shows the app inside the notebook with ``df`` loaded;
``edalens.walk(df, mode="browser")`` runs it in the browser with no server (pygwalker style).
In Google Colab, ``edalens.run_colab()`` also opens a public localtunnel link.
"""

from importlib.metadata import PackageNotFoundError, version

from edalens.browser import BrowserApp
from edalens.colab import ColabApp, run_colab
from edalens.launcher import EDALensServer, LaunchError, show
from edalens.notebook import NotebookApp, walk

try:
    __version__ = version("edalens")
except PackageNotFoundError:  # Imported from a checkout without installation.
    __version__ = "0+unknown"

__all__ = [
    "BrowserApp",
    "ColabApp",
    "EDALensServer",
    "LaunchError",
    "NotebookApp",
    "run_colab",
    "show",
    "walk",
    "__version__",
]
