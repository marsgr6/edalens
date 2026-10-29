"""In-browser (stlite) mode: page contents, data round trip, and the app entry."""

import io
import json
import re
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import edalens
from edalens import browser, launcher
from edalens.utils import data_processing


def mount_options(page: str) -> dict:
    match = re.search(r"mount\((\{.*\}), document", page, re.S)
    return json.loads(match.group(1))


@pytest.fixture
def sample():
    return pd.DataFrame(
        {
            "x": [1.5, 2.5, None],
            "n": [1, 2, 3],
            "nullable": pd.array([1, None, 3], dtype="Int64"),
            "cat": pd.Categorical(["a", "b", "a"]),
            "when": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "text": ["</script><b>", "y", "z"],
            "flag": [True, False, True],
        },
        index=[10, 20, 30],
    )


def test_env_name_and_entry_match_the_app():
    assert (
        f'os.environ["{data_processing.PRESET_DATA_ENV}"]' in browser.DATA_ENTRY_SOURCE
    )


def test_frame_json_round_trips_dtypes(tmp_path, sample):
    path = tmp_path / "data.json"
    path.write_text(browser.frame_json(sample))
    state = {}
    assert data_processing.load_preset_data(
        state, {data_processing.PRESET_DATA_ENV: str(path)}
    )
    expected = sample.reset_index(drop=True)
    # JSON brings datetimes back in ns; pandas 3 creates them in us by default.
    expected["when"] = expected["when"].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(state["raw_df"], expected, check_index_type=False)


def test_frame_json_rejects_duplicate_columns():
    with pytest.raises(ValueError, match="unique"):
        browser.frame_json(pd.DataFrame([[1, 2]], columns=["a", "a"]))


def test_page_embeds_the_app_and_data_safely(sample):
    page = browser.page_html(sample)
    assert page.count("</script>") == 1  # The data cannot close the inline script.
    assert f"@stlite/browser@{browser.STLITE_VERSION}/" in page
    options = mount_options(page)
    assert options["entrypoint"] == browser.ENTRYPOINT
    assert options["requirements"] == browser.BROWSER_REQUIREMENTS
    assert "prophet" not in options["requirements"]
    files = options["files"]
    assert {
        "edalens/app.py",
        "edalens/views/home.py",
        "edalens/utils/data_processing.py",
    } <= set(files)
    assert (
        files["edalens/__init__.py"] == ""
    )  # The launcher is not shipped to the browser.
    assert (
        pd.read_json(io.StringIO(files[browser.DATA_FILE]), orient="table")["text"][0]
        == "</script><b>"
    )


def test_page_without_data_preloads_nothing():
    options = mount_options(browser.page_html(None))
    assert options["files"][options["entrypoint"]] == browser.ENTRY_SOURCE
    assert browser.DATA_FILE not in options["files"]


def test_walk_browser_mode_starts_no_server(monkeypatch, sample):
    def no_server(*args, **kwargs):
        raise AssertionError("browser mode must not start a server")

    monkeypatch.setattr(launcher.EDALensServer, "start", no_server)
    app = edalens.walk(sample, mode="browser", display=False, height=600)
    assert isinstance(app, edalens.BrowserApp)
    frame = app.iframe_html()
    assert 'srcdoc="' in frame and 'height="600"' in frame
    assert "<script" not in frame.split('srcdoc="', 1)[1].split('"', 1)[0]  # Escaped.
    app.stop()  # Nothing to stop; must not fail.


def test_walk_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        edalens.walk(pd.DataFrame({"x": [1]}), mode="tunnel", display=False)


def test_save_writes_a_standalone_page(tmp_path, sample):
    path = edalens.walk(sample, mode="browser", display=False).save(
        tmp_path / "edalens.html"
    )
    assert "mount(" in path.read_text()


def test_browser_entry_preloads_data_and_uses_a_radio_menu(
    tmp_path, monkeypatch, sample
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / browser.DATA_FILE).write_text(browser.frame_json(sample))
    monkeypatch.setattr(sys, "platform", "emscripten")  # As under Pyodide.
    # The entry sets EDALENS_DATA in os.environ; monkeypatch restores it afterwards.
    monkeypatch.setenv(data_processing.PRESET_DATA_ENV, "")
    app = AppTest.from_string(browser.DATA_ENTRY_SOURCE)
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["data_loaded"] is True
    assert list(app.session_state["filtered_df"].columns) == list(sample.columns)
    assert app.sidebar.radio[0].value == "Load data"
