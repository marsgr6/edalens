"""Colab helper: tunnel startup/shutdown, URL composition, and one-time install."""

import subprocess
import sys
import urllib.request

import pytest

import edaviz
from edaviz import colab
from edaviz.launcher import LaunchError


def fake_tunnel(script: str):
    return lambda port: [sys.executable, "-c", script]


ANNOUNCE = (
    "import sys, time\n"
    "print('npm warn see https://docs.npmjs.com/cli', flush=True)\n"
    "print('your url is: https://fake-name.loca.lt', flush=True)\n"
    "time.sleep(600)\n"
)


def local_ok(url: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url + "_stcore/health", timeout=5) as response:
        return response.read().strip() == b"ok"


def test_run_colab_exposes_the_prefixed_app_and_stop_closes_everything(monkeypatch):
    monkeypatch.setattr(colab, "tunnel_command", fake_tunnel(ANNOUNCE))
    monkeypatch.setattr(colab, "tunnel_password", lambda: "203.0.113.7")

    app = edaviz.run_colab(install=False)
    try:
        prefix = app.server.url.split("/", 3)[3]
        assert prefix.startswith("edaviz-")
        assert app.url == f"https://fake-name.loca.lt/{prefix}"
        assert app.running and local_ok(app.server.url)
        html = app._repr_html_()
        assert app.url in html and "203.0.113.7" in html
    finally:
        app.stop()
    assert not app.running
    assert app.tunnel.poll() is not None
    assert not app.server.running
    app.stop()  # idempotent


def test_tunnel_without_a_url_fails_and_stops_the_app(monkeypatch):
    monkeypatch.setattr(colab, "tunnel_command", fake_tunnel("print('connection refused')"))
    started = []
    real_start = colab.EDAVizServer.start

    def tracking_start(self, *args, **kwargs):
        started.append(self)
        return real_start(self, *args, **kwargs)

    monkeypatch.setattr(colab.EDAVizServer, "start", tracking_start)
    with pytest.raises(LaunchError, match="connection refused"):
        colab.run_colab(install=False, tunnel_timeout=10)
    assert started and not started[0].running


def test_tunnel_command_pins_loopback_and_requires_npx(monkeypatch):
    monkeypatch.setattr(colab.shutil, "which", lambda name: "/usr/bin/npx")
    command = colab.tunnel_command(8600)
    assert command[:3] == ["/usr/bin/npx", "--yes", "localtunnel"]
    assert command[command.index("--port") + 1] == "8600"
    assert command[command.index("--local-host") + 1] == "127.0.0.1"
    monkeypatch.setattr(colab.shutil, "which", lambda name: None)
    with pytest.raises(LaunchError, match="Node.js"):
        colab.tunnel_command(8600)


def test_requirements_are_installed_once_per_runtime(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(colab.subprocess, "run", fake_run)
    marker = tmp_path / "installed"
    assert colab.requirements_file().name == "requirements.txt"
    assert colab.install_requirements(marker=marker) is True
    assert colab.install_requirements(marker=marker) is False
    assert colab.install_requirements(force=True, marker=marker) is True
    assert len(calls) == 2
    assert calls[0][1:5] == ["-m", "pip", "install", "-q"]


def test_failed_install_raises_with_pip_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        colab.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "No matching distribution"),
    )
    with pytest.raises(LaunchError, match="No matching distribution"):
        colab.install_requirements(marker=tmp_path / "installed")
    assert not (tmp_path / "installed").exists()
