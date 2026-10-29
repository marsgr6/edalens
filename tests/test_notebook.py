"""Notebook embedding: data handoff to the app, iframe output, and cleanup."""

from pathlib import Path
import sys
import types
import urllib.request

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import edalens
from edalens import launcher, notebook
from edalens.utils import data_processing

ROOT = Path(__file__).resolve().parents[1]

# Stand-in for Streamlit that serves the preload path it received at /<base>/env.
FAKE_SERVER = """
import http.server, os, sys
port, base = int(sys.argv[1]), sys.argv[2]
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = {
            f"/{base}/_stcore/health": b"ok",
            f"/{base}/env": os.environ.get("EDALENS_DATA", "").encode(),
        }.get(self.path)
        self.send_response(404 if body is None else 200)
        self.end_headers()
        self.wfile.write(body or b"")
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


@pytest.fixture
def fake_streamlit(monkeypatch):
    monkeypatch.setattr(
        launcher,
        "streamlit_command",
        lambda script, port, base: [sys.executable, "-c", FAKE_SERVER, str(port), base],
    )


def fetch(url: str) -> str:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        return response.read().decode()


def test_env_name_matches_the_app():
    assert notebook.PRESET_DATA_ENV == data_processing.PRESET_DATA_ENV


def test_walk_hands_the_dataframe_to_the_app_and_stop_cleans_up(fake_streamlit):
    df = pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})
    app = edalens.walk(df, display=False)
    try:
        assert app.running
        path = Path(fetch(app.url + "env"))
        assert path == app.data_path and path.is_file()
        pd.testing.assert_frame_equal(pd.read_pickle(path), df)
        html = app.iframe_html()
        assert f'<iframe src="{app.url}"' in html and 'height="900"' in html
    finally:
        app.stop()
    assert not app.running
    assert not path.exists()
    app.stop()  # idempotent


def test_walk_without_data_passes_no_preload(fake_streamlit):
    with edalens.walk(display=False) as app:
        assert app.data_path is None
        assert fetch(app.url + "env") == ""


def test_walk_accepts_a_series_and_rejects_other_types(fake_streamlit):
    with edalens.walk(pd.Series([1.0, 2.0], name="y"), display=False) as app:
        assert list(pd.read_pickle(app.data_path).columns) == ["y"]
    with pytest.raises(TypeError):
        edalens.walk([1, 2, 3], display=False)


def test_failed_start_removes_the_data_file(monkeypatch, tmp_path):
    written = []
    real_write = notebook._write_data
    monkeypatch.setattr(
        notebook,
        "_write_data",
        lambda df: written.append(real_write(df)) or written[-1],
    )

    def fail(self, timeout):
        raise launcher.LaunchError("boom")

    monkeypatch.setattr(launcher.EDALensServer, "start", fail)
    with pytest.raises(launcher.LaunchError):
        edalens.walk(pd.DataFrame({"x": [1]}), display=False)
    assert written and not written[0].exists()


def test_load_preset_data_runs_once_per_session(tmp_path):
    path = tmp_path / "df.pkl"
    pd.DataFrame({"x": [1.5, 2.5], "c": ["a", "b"]}).to_pickle(path)
    state = {}
    env = {data_processing.PRESET_DATA_ENV: str(path)}
    assert data_processing.load_preset_data(state, env)
    assert state["data_loaded"] and state["operations"] == []
    assert state["current_numeric_cols"] == ["x"]
    assert state["current_categorical_cols"] == ["c"]
    state["data_loaded"] = False  # e.g. after "Reset All"
    assert not data_processing.load_preset_data(state, env)
    assert state["data_loaded"] is False
    assert not data_processing.load_preset_data({}, {})


def test_app_starts_with_the_preloaded_dataframe(tmp_path, monkeypatch):
    path = tmp_path / "df.pkl"
    pd.DataFrame({"x": [1, 2, 3], "c": ["a", "b", "c"]}).to_pickle(path)
    monkeypatch.setenv(data_processing.PRESET_DATA_ENV, str(path))
    app = AppTest.from_file(str(ROOT / "edalens" / "app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["data_loaded"] is True
    assert list(app.session_state["filtered_df"].columns) == ["x", "c"]
    assert len(app.dataframe) == 1  # The processed-data preview is shown.


def fake_colab(monkeypatch):
    """Install a stand-in google.colab.output that records iframe requests."""
    calls = []
    output = types.ModuleType("google.colab.output")
    output.serve_kernel_port_as_iframe = lambda port, **kwargs: calls.append(
        (port, kwargs)
    )
    colab_module = types.ModuleType("google.colab")
    colab_module.output = output
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)
    monkeypatch.setitem(sys.modules, "google.colab.output", output)
    return calls


def test_walk_in_colab_embeds_through_colabs_port_proxy(fake_streamlit, monkeypatch):
    calls = fake_colab(monkeypatch)
    installs = []
    monkeypatch.setattr(
        "edalens.colab.install_requirements", lambda: installs.append(1)
    )
    monkeypatch.setattr(notebook, "_in_notebook", lambda: True)
    with edalens.walk(pd.DataFrame({"x": [1]}), height=700) as app:
        assert app.colab
        assert app.server.env["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] == "false"
        assert app.server.env["STREAMLIT_SERVER_ENABLE_CORS"] == "false"
        assert Path(fetch(app.url + "env")) == app.data_path
        # Colab's helper builds the iframe in the output frame, under the random prefix.
        assert calls == [(app.server.port, {"path": app.path, "height": "700"})]
        assert app.url.endswith(app.path) and app.path.startswith("/edalens-")
        app.display(height=500)
        assert calls[-1] == (app.server.port, {"path": app.path, "height": "500"})
    assert installs == [1]


def test_local_walk_keeps_streamlit_protections(fake_streamlit):
    with edalens.walk(display=False) as app:
        assert not app.colab
        assert not any(key.startswith("STREAMLIT_") for key in app.server.env)
