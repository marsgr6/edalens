"""Exercise the real Streamlit entry point with a fresh session."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_home_renders_without_an_exception():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "edalens" / "app.py")
    )
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state["data_loaded"] is False
    assert app.session_state["operations"] == []
