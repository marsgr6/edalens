"""Fail-fast validation entry point; never installs or updates dependencies."""

from pathlib import Path
import ast
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True, timeout=300)


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Validation requires Python 3.10, matching deployment.")
    sources = [ROOT / "my_viz_plotly.py"]
    for directory in ("edalens", "edalens/views", "edalens/utils", "scripts", "tests"):
        sources.extend(sorted((ROOT / directory).glob("*.py")))
    for source in sources:
        ast.parse(
            source.read_text(encoding="utf-8-sig"),
            filename=str(source),
            feature_version=(3, 10),
        )
    run(sys.executable, "-m", "pip", "check")
    run(sys.executable, "scripts/check_environment.py")
    run(sys.executable, "scripts/lock_dependencies.py", "--check")
    run(sys.executable, "scripts/check_lint.py")
    # The app modules keep their original formatting; only the helpers are checked.
    run(
        sys.executable,
        "-m",
        "ruff",
        "format",
        "--check",
        "edalens",
        "scripts",
        "tests",
        "--exclude",
        "edalens/app.py",
        "--exclude",
        "edalens/views",
        "--exclude",
        "edalens/utils",
    )
    run(sys.executable, "-m", "pytest")
    run(sys.executable, "scripts/check_package.py")
    print("All configured validation gates passed.")


if __name__ == "__main__":
    main()
