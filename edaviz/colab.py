"""Run EDAViz in Google Colab (or any machine with Node.js) behind a public tunnel.

Colab cannot open ``127.0.0.1`` in your browser, so the app is exposed with
localtunnel (``npx localtunnel``)::

    import edaviz
    app = edaviz.run_colab()   # installs requirements once, prints link + password
    app.stop()                 # closes the tunnel and stops the app

localtunnel shows a reminder page on first visit that asks for a "tunnel
password": it is the public IP of the machine running the tunnel, which
``run_colab`` looks up and prints. The app keeps its random URL prefix, so the
tunnel's root address alone does not reach it.
"""

from __future__ import annotations

import atexit
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import weakref
from pathlib import Path
from typing import List, Optional

from edaviz.launcher import (
    STARTUP_TIMEOUT,
    STOP_TIMEOUT,
    HOST,
    EDAVizServer,
    LaunchError,
    choose_port,
)

TUNNEL_TIMEOUT = 90.0  # First use downloads localtunnel through npx.
PASSWORD_URL = "https://loca.lt/mytunnelpassword"
_URL_PATTERN = re.compile(r"your url is: (https://\S+)", re.IGNORECASE)
_INSTALL_MARKER = Path("/tmp/edaviz-requirements-installed")

_running: "weakref.WeakSet[ColabApp]" = weakref.WeakSet()


def requirements_file() -> Optional[Path]:
    """Return the checkout's requirements.txt, or None for an installed wheel."""
    path = Path(__file__).resolve().parent.parent / "requirements.txt"
    return path if path.is_file() else None


def install_requirements(force: bool = False, marker: Path = _INSTALL_MARKER) -> bool:
    """``pip install -r requirements.txt`` once per runtime; return whether it ran.

    The app runs in a separate process, so packages installed here are picked
    up without restarting the notebook kernel.
    """
    requirements = requirements_file()
    if requirements is None or (marker.exists() and not force):
        return False
    print("Installing EDAViz requirements (first run only, 1-3 minutes)...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise LaunchError(f"pip install failed:\n{result.stderr[-3000:]}")
    marker.touch()
    return True


def tunnel_command(port: int) -> List[str]:
    npx = shutil.which("npx")
    if npx is None:
        raise LaunchError(
            "Node.js (npx) was not found; it is preinstalled on Colab. "
            "Elsewhere install Node.js to use the tunnel."
        )
    # Pin the loopback address: "localhost" may resolve to ::1, where the app does not listen.
    return [npx, "--yes", "localtunnel", "--port", str(port), "--local-host", HOST]


def tunnel_password(url: str = PASSWORD_URL, timeout: float = 10.0) -> Optional[str]:
    """Public IP that localtunnel asks for on its reminder page; None if unknown."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8").strip() or None
    except OSError:
        return None


class ColabApp:
    """An EDAViz server plus the localtunnel process that exposes it."""

    def __init__(self, server: EDAVizServer, tunnel: subprocess.Popen, public_root: str, password: Optional[str]):
        self.server = server
        self.tunnel = tunnel
        self.password = password
        # The server answers only under its random prefix, so the public URL keeps it.
        self.url = public_root.rstrip("/") + "/" + server.url.split("/", 3)[3]
        _running.add(self)

    @property
    def running(self) -> bool:
        return self.server.running and self.tunnel.poll() is None

    def stop(self) -> None:
        """Close the tunnel and stop the app; safe to call repeatedly."""
        if self.tunnel.poll() is None:
            self.tunnel.terminate()
            try:
                self.tunnel.wait(STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                self.tunnel.kill()
        self.server.stop()
        _running.discard(self)

    def __enter__(self) -> "ColabApp":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _status(self) -> str:
        return "running" if self.running else "stopped"

    def __repr__(self) -> str:
        return f"<EDAViz {self.url} password={self.password} ({self._status()})>"

    def _repr_html_(self) -> str:
        password = self.password or "the public IP of this runtime"
        return (
            f'<p><b>EDAViz</b> ({self._status()}): <a href="{self.url}" target="_blank">{self.url}</a></p>'
            f"<p>If a localtunnel page asks for a <b>Tunnel Password</b>, enter "
            f"<code>{password}</code> and click Submit.</p>"
        )


def _open_tunnel(port: int, timeout: float) -> tuple[subprocess.Popen, str]:
    """Start localtunnel and return the process and its public URL."""
    process = subprocess.Popen(
        tunnel_command(port),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def pump() -> None:
        # Keep draining output so the tunnel never blocks on a full pipe.
        for line in process.stdout:  # type: ignore[union-attr]
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    output: List[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        if line is None:
            break
        output.append(line)
        match = _URL_PATTERN.search(line)
        if match:
            return process, match.group(1)
    if process.poll() is None:
        process.terminate()
    tail = "".join(output[-20:]).strip()
    raise LaunchError(f"localtunnel did not report a public URL within {timeout:g} s.\n{tail}".rstrip())


def run_colab(
    *,
    port: Optional[int] = None,
    install: bool = True,
    startup_timeout: float = STARTUP_TIMEOUT,
    tunnel_timeout: float = TUNNEL_TIMEOUT,
) -> ColabApp:
    """Install requirements (once), start EDAViz, and expose it through localtunnel.

    Returns once the app answers locally and the tunnel has a public URL. The
    returned object shows the link and tunnel password when displayed in a
    notebook; call ``.stop()`` to shut both down.
    """
    if install:
        install_requirements()
    print("Starting EDAViz...", flush=True)
    server = EDAVizServer(choose_port(port)).start(startup_timeout)
    print("Opening the public tunnel...", flush=True)
    try:
        tunnel, public_root = _open_tunnel(server.port, tunnel_timeout)
    except BaseException:
        server.stop()
        raise
    return ColabApp(server, tunnel, public_root, tunnel_password())


@atexit.register
def _stop_all() -> None:
    # The tunnel is a separate process; do not leave it running after the kernel exits.
    for app in list(_running):
        app.stop()
