"""Start, verify, and stop local EDALens servers without touching other processes."""

from pathlib import Path
import socket
import subprocess
import sys
import urllib.request

import pytest

import edalens
from edalens import launcher

ROOT = Path(__file__).resolve().parents[1]

# Minimal stand-in for Streamlit: answers the health check only under its own prefix.
FAKE_SERVER = """
import http.server, sys
port, base = int(sys.argv[1]), sys.argv[2]
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ok = self.path in (f"/{base}/_stcore/health", f"/{base}/")
        self.send_response(200 if ok else 404)
        self.end_headers()
        self.wfile.write(b"ok" if ok else b"")
    def log_message(self, *args):
        pass
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def fetch(url: str) -> int:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5) as response:
        return response.status


@pytest.fixture
def fake_streamlit(monkeypatch):
    monkeypatch.setattr(
        launcher,
        "streamlit_command",
        lambda script, port, base: [
            sys.executable,
            "-c",
            FAKE_SERVER,
            str(port),
            base,
        ],
    )


@pytest.fixture
def started(monkeypatch):
    """Record every child process the launcher starts."""
    processes = []
    original = subprocess.Popen

    def record(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(launcher.subprocess, "Popen", record)
    yield processes
    for process in processes:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.fixture
def busy_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        yield listener.getsockname()[1]


def test_command_binds_localhost_and_runs_installed_script():
    script = launcher.app_script()
    assert script == ROOT / "edalens" / "app.py"
    command = launcher.streamlit_command(script, 8765, "edalens-token")
    assert command[:5] == [sys.executable, "-m", "streamlit", "run", str(script)]
    options = dict(zip(command[5::2], command[6::2]))
    assert options["--server.address"] == "127.0.0.1"
    assert options["--server.port"] == "8765"
    assert options["--server.headless"] == "true"
    assert options["--server.baseUrlPath"] == "edalens-token"


@pytest.mark.parametrize("address", ["127.0.0.1", "0.0.0.0"])
def test_explicit_occupied_port_is_rejected_without_starting(
    address, fake_streamlit, started
):
    with socket.socket() as listener:
        listener.bind((address, 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(edalens.LaunchError, match=f"Port {port}.*in use"):
            edalens.show(port=port, open_browser=False)
    assert started == []


def test_default_port_skips_an_occupied_port(monkeypatch, busy_port):
    monkeypatch.setattr(launcher, "DEFAULT_PORT", busy_port)
    port = launcher.choose_port(None)
    assert port != busy_port
    assert launcher.port_is_free(port)


@pytest.mark.parametrize("port", [0, 65536])
def test_invalid_port_is_rejected(port):
    with pytest.raises(edalens.LaunchError, match="between 1 and 65535"):
        launcher.choose_port(port)


def test_startup_failure_reports_exit_code_and_output(monkeypatch, started):
    monkeypatch.setattr(
        launcher,
        "streamlit_command",
        lambda *args: [sys.executable, "-c", "print('boom'); raise SystemExit(3)"],
    )
    with pytest.raises(edalens.LaunchError, match=r"(?s)exited with code 3.*boom"):
        edalens.show(open_browser=False)
    assert [process.returncode for process in started] == [3]


def test_startup_timeout_stops_the_child(monkeypatch, started):
    monkeypatch.setattr(
        launcher,
        "streamlit_command",
        lambda *args: [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    with pytest.raises(edalens.LaunchError, match="not ready within 1"):
        edalens.show(open_browser=False, startup_timeout=1)
    assert len(started) == 1 and started[0].poll() is not None


def test_another_server_on_the_port_is_not_mistaken_for_ours(
    monkeypatch, busy_port, started
):
    # The child never serves our random prefix; the foreign listener must not count.
    monkeypatch.setattr(launcher, "port_is_free", lambda port: True)
    monkeypatch.setattr(
        launcher,
        "streamlit_command",
        lambda *args: [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    with pytest.raises(edalens.LaunchError):
        edalens.show(port=busy_port, open_browser=False, startup_timeout=1)


def test_stop_affects_only_the_server_it_started(fake_streamlit):
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        first = edalens.show(open_browser=False)
        second = edalens.show(open_browser=False)
        assert first.port != second.port
        assert fetch(first.url) == fetch(second.url) == 200

        assert first.stop() is not None
        assert not first.running
        assert second.running and fetch(second.url) == 200
        assert bystander.poll() is None

        second.stop()
        assert not second.running
        assert launcher.port_is_free(first.port)
        assert launcher.port_is_free(second.port)
    finally:
        bystander.kill()
        bystander.wait()


def test_stop_is_idempotent_and_context_manager_stops(fake_streamlit):
    with edalens.show(open_browser=False) as server:
        assert server.running
        assert server.url.startswith(f"http://127.0.0.1:{server.port}/edalens-")
    assert not server.running
    server.stop()
    assert repr(server) == f"<EDALens server {server.url} (stopped)>"


@pytest.mark.parametrize("open_browser", [False, True])
def test_browser_opening_is_optional(monkeypatch, fake_streamlit, open_browser):
    opened = []
    monkeypatch.setattr(launcher.webbrowser, "open", opened.append)
    with edalens.show(open_browser=open_browser) as server:
        assert opened == ([server.url] if open_browser else [])


def test_cli_prints_url_and_stops_on_interrupt(
    monkeypatch, capsys, fake_streamlit, started
):
    def interrupt(self):
        assert fetch(self.url) == 200
        raise KeyboardInterrupt

    monkeypatch.setattr(launcher.EDALensServer, "wait", interrupt)
    assert launcher.main(["--no-browser"]) == 0
    output = capsys.readouterr().out
    assert "http://127.0.0.1:" in output and "EDALens stopped." in output
    assert started[0].poll() is not None


def test_cli_reports_launch_failure(capsys, busy_port):
    assert launcher.main(["--port", str(busy_port), "--no-browser"]) == 1
    assert f"Port {busy_port}" in capsys.readouterr().err


def test_real_streamlit_server_starts_serves_and_stops():
    server = edalens.show(open_browser=False)
    try:
        assert fetch(server.url) == 200
    finally:
        assert server.stop() is not None
    assert not server.running
    assert launcher.port_is_free(server.port)
