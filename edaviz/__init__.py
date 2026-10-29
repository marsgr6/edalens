"""Local launcher for the EDAViz Streamlit app.

>>> import edaviz
>>> app = edaviz.show()  # returns once the server is ready; does not block
>>> app.url
>>> app.stop()

In Jupyter, ``edaviz.walk(df)`` shows the app inside the notebook with ``df`` loaded.
"""

from importlib.metadata import PackageNotFoundError, version

from edaviz.launcher import EDAVizServer, LaunchError, show
from edaviz.notebook import NotebookApp, walk

try:
    __version__ = version("edaviz")
except PackageNotFoundError:  # Imported from a checkout without installation.
    __version__ = "0+unknown"

__all__ = ["EDAVizServer", "LaunchError", "NotebookApp", "show", "walk", "__version__"]
