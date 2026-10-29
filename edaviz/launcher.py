"""Start and stop a local EDAViz Streamlit server from Python or the terminal.

The server binds to 127.0.0.1 only and runs the ``my_viz_plotly.py`` installed
next to this package. Each server gets a random URL prefix, so readiness checks
cannot be answered by another process that already uses the port.
"""

from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional
import urllib.request
import weakref
import webbrowser

HOST = "127.0.0.1"
DEFAULT_PORT = 8501
PORT_SEARCH_LIMIT = 50  # Ports tried, starting at DEFAULT_PORT, when none is given.
STARTUP_TIMEOUT = 60.0
STOP_TIMEOUT = 10.0
LOG_TAIL_LINES = 20

_running: "weakref.WeakSet[EDAVizServer]" = weakref.WeakSet()


class LaunchError(RuntimeError):
    """The EDAViz server could not be started or did not become ready."""


def app_script() -> Path:
    """Return the routed app script installed alongside the ``edaviz`` package."""
    script = Path(__file__).resolve().parent.parent / "my_viz_plotly.py"
    if not script.is_file():
        raise LaunchError(f"EDAViz app not found at {script}; reinstall edaviz.")
    return script


def streamlit_command(script: Path, port: int, base: str) -> List[str]:
    # Command-line options take precedence over Streamlit config files and
    # environment variables, so user configuration cannot widen the bind address.
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.address",
        HOST,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.baseUrlPath",
        base,
    ]


def port_is_free(port: int) -> bool:
    """Return whether ``port`` is unused on loopback and on all IPv4 addresses.

    Windows lets a loopback bind shadow an existing wildcard listener, so the
    wildcard address is probed too; the probe sockets never listen.
    """
    for address in (HOST, "0.0.0.0"):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if os.name == "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                probe.bind((address, port))
            except OSError:
                return False
    return True


def choose_port(port: Optional[int]) -> int:
    """Validate an explicit port, or find a free one starting at DEFAULT_PORT."""
    if port is not None:
        if not 1 <= port <= 65535:
            raise LaunchError(f"Port must be between 1 and 65535, not {port}.")
        if not port_is_free(port):
            raise LaunchError(
                f"Port {port} on {HOST} is already in use; "
                "choose another port or omit it to pick a free one."
            )
        return port
    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + PORT_SEARCH_LIMIT):
        if port_is_free(candidate):
            return candidate
    raise LaunchError(
        f"No free port in {DEFAULT_PORT}-{DEFAULT_PORT + PORT_SEARCH_LIMIT - 1} "
        f"on {HOST}; pass an explicit port."
    )


def _is_healthy(url: str) -> bool:
    # Ignore proxy settings: the check must reach the loopback server directly.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + "_stcore/health", timeout=2) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except OSError:  # URLError, HTTPError, refused or reset connections.
        return False


class EDAVizServer:
    """Handle for one local EDAViz server process started by this launcher."""

    def __init__(self, port: int, env: Optional[Dict[str, str]] = None) -> None:
        self.port = port
        self.env = dict(env or {})  # Extra environment variables for the app process.
        self.url = f"http://{HOST}:{port}/edaviz-{secrets.token_hex(8)}/"
        self.log_path: Optional[Path] = None
        self._process: Optional[subprocess.Popen] = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, startup_timeout: float = STARTUP_TIMEOUT) -> "EDAVizServer":
        """Start the server and return once it answers its health check."""
        if self._process is not None:
            raise LaunchError("This server was already started; call show() again.")
        base = self.url.rstrip("/").rsplit("/", 1)[1]
        script = app_script()
        log_fd, log_name = tempfile.mkstemp(prefix="edaviz-", suffix=".log")
        self.log_path = Path(log_name)
        if os.name == "nt":
            # Own process group: Ctrl+C in the caller's console does not reach it,
            # and stop() can send CTRL_BREAK_EVENT to this process alone.
            options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            options = {"start_new_session": True}
        with os.fdopen(log_fd, "wb") as log:
            # The app's directory as cwd keeps stray modules in the caller's
            # directory off sys.path; the app reads no cwd-relative files.
            self._process = subprocess.Popen(
                streamlit_command(script, self.port, base),
                cwd=script.parent,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", **self.env},
                **options,
            )
        _running.add(self)
        deadline = time.monotonic() + startup_timeout
        while not _is_healthy(self.url):
            code = self._process.poll()
            if code is not None:
                self._fail(f"EDAViz exited with code {code} during startup.")
            if time.monotonic() >= deadline:
                self._terminate()
                self._fail(f"EDAViz was not ready within {startup_timeout:g} s.")
            time.sleep(0.2)
        return self

    def wait(self) -> Optional[int]:
        """Block until the server exits; return its exit code."""
        if self._process is None:
            return None
        # Polling keeps Ctrl+C responsive; Windows cannot interrupt Popen.wait().
        while self._process.poll() is None:
            time.sleep(0.5)
        return self._process.returncode

    def stop(self, timeout: float = STOP_TIMEOUT) -> Optional[int]:
        """Stop the process started by this handle; safe to call repeatedly."""
        process = self._process
        if process is None:
            return None
        exited_by_itself = process.poll() is not None
        if not exited_by_itself:
            self._terminate(timeout)
        _running.discard(self)
        if not exited_by_itself and process.returncode is not None:
            self._remove_log()
        return process.returncode

    def _terminate(self, timeout: float = STOP_TIMEOUT) -> None:
        process = self._process
        assert process is not None
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
        except OSError:  # No console to deliver CTRL_BREAK, or already exited.
            process.kill()
        try:
            process.wait(timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout)

    def _fail(self, message: str) -> None:
        _running.discard(self)
        lines = []
        if self.log_path is not None:
            lines = self.log_path.read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-LOG_TAIL_LINES:])
        raise LaunchError(f"{message}\nLog: {self.log_path}\n{tail}".rstrip())

    def _remove_log(self) -> None:
        if self.log_path is not None:
            try:
                self.log_path.unlink()
            except OSError:
                pass

    def __enter__(self) -> "EDAVizServer":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"<EDAViz server {self.url} ({state})>"

    def _repr_html_(self) -> str:
        state = "running" if self.running else "stopped"
        return f'EDAViz server <a href="{self.url}" target="_blank">{self.url}</a> ({state})'


def show(
    *,
    port: Optional[int] = None,
    open_browser: bool = True,
    startup_timeout: float = STARTUP_TIMEOUT,
) -> EDAVizServer:
    """Start EDAViz on localhost without blocking and return its server handle.

    ``port=None`` uses the first free port from 8501. Call ``.stop()`` on the
    returned handle, or use it as a context manager, to shut the server down.
    """
    server = EDAVizServer(choose_port(port)).start(startup_timeout)
    if open_browser:
        webbrowser.open(server.url)
    return server


@atexit.register
def _stop_all() -> None:
    for server in list(_running):
        server.stop()


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: Optional[List[str]] = None) -> int:
    from edaviz import __version__

    parser = argparse.ArgumentParser(
        prog="edaviz", description=f"Run EDAViz locally on {HOST}."
    )
    parser.add_argument(
        "--port",
        type=int,
        help=f"port to use (default: first free port from {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="do not open a browser tab",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=STARTUP_TIMEOUT,
        help="seconds to wait for startup (default: %(default)s)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)

    # Closing the terminal or a service manager's stop also shuts the server down.
    signals = [
        getattr(signal, name, None) for name in ("SIGTERM", "SIGHUP", "SIGBREAK")
    ]
    previous = {
        number: signal.signal(number, _interrupt)
        for number in signals
        if isinstance(number, signal.Signals)
    }
    try:
        return _run(args)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _run(args: argparse.Namespace) -> int:
    print("Starting EDAViz...", flush=True)
    try:
        server = show(
            port=args.port,
            open_browser=args.open_browser,
            startup_timeout=args.timeout,
        )
    except LaunchError as error:
        print(f"edaviz: {error}", file=sys.stderr)
        return 1
    print(f"EDAViz is running at {server.url}\nPress Ctrl+C to stop.", flush=True)
    try:
        code = server.wait()
    except KeyboardInterrupt:
        server.stop()
        print("EDAViz stopped.")
        return 0
    print(
        f"edaviz: server exited with code {code}; log: {server.log_path}",
        file=sys.stderr,
    )
    return 1
