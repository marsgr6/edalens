"""Build the wheel and sdist outside the checkout and gate their quality.

Never uploads anything. Uses the locked build backend (``--no-isolation``) so the
gate does not fetch unhashed build dependencies.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# Exact top-level entries the wheel may contain (replaces check W009 with W201/W202).
TOPLEVEL = ("edalens",)
MIN_PYROMA_RATING = "8"  # pyroma's default threshold, stated explicitly.

SMOKE_TEST = """
import importlib.metadata, pathlib, sys, urllib.request
target, checkout = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(target))
import edalens, edalens.app, edalens.views.home, edalens.utils.data_processing
for module in (edalens, edalens.app, edalens.views.home, edalens.utils.data_processing):
    path = pathlib.Path(module.__file__).resolve()
    assert target in path.parents and checkout not in path.parents, path
assert importlib.metadata.version("edalens")
from streamlit.testing.v1 import AppTest
app = AppTest.from_file(edalens.app.__file__)
app.run(timeout=60)
assert not app.exception, app.exception
(script,) = importlib.metadata.entry_points(group="console_scripts", name="edalens")
assert script.value == "edalens.launcher:main", script
assert edalens.launcher.app_script() == pathlib.Path(edalens.app.__file__).resolve()
server = edalens.show(open_browser=False)
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    assert opener.open(server.url, timeout=10).status == 200
finally:
    server.stop()
assert not server.running and edalens.launcher.port_is_free(server.port)
print("Installed wheel imports, renders Home, and launches/stops outside the checkout.")
"""


def run(*args: str, cwd: Path) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True, timeout=300)


def check(project: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        outside = Path(temporary)
        dist = outside / "dist"
        run(sys.executable, "-m", "validate_pyproject", "pyproject.toml", cwd=project)
        # Without --sdist/--wheel, build creates the sdist and then builds the
        # wheel from that sdist, so a file missing from the sdist fails here.
        run(
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(dist),
            ".",
            cwd=project,
        )
        (wheel,) = dist.glob("*.whl")
        (sdist,) = dist.glob("*.tar.gz")
        run(
            sys.executable,
            "-m",
            "check_wheel_contents",
            "--toplevel",
            ",".join(TOPLEVEL),
            *(f"--package={entry}" for entry in TOPLEVEL),
            str(wheel),
            cwd=project,
        )
        run(
            sys.executable,
            "-m",
            "twine",
            "check",
            "--strict",
            str(wheel),
            str(sdist),
            cwd=outside,
        )
        run(
            sys.executable,
            "-m",
            "pyroma",
            "--min",
            MIN_PYROMA_RATING,
            "--file",
            str(sdist),
            cwd=outside,
        )
        target = outside / "site"
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-index",
            "--target",
            str(target),
            str(wheel),
            cwd=outside,
        )
        run(
            sys.executable,
            "-c",
            SMOKE_TEST,
            str(target),
            str(project.resolve()),
            cwd=outside,
        )
    print("Package build, metadata, contents, and installed-wheel checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    check(parser.parse_args().project)


if __name__ == "__main__":
    main()
