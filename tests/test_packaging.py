"""Keep package metadata aligned with runtime inputs and the package gate fail-closed."""

import shutil
import subprocess
from pathlib import Path

import pytest
import tomli
from packaging.requirements import Requirement

from scripts.check_package import TOPLEVEL, check

ROOT = Path(__file__).resolve().parents[1]


def read_requirements(path: Path) -> set[str]:
    requirements = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            requirements |= read_requirements(path.parent / line[3:].strip())
        elif line:
            requirements.add(str(Requirement(line)))
    return requirements


def test_package_dependencies_are_unpinned_and_locked_for_deployment():
    # The library declares names only, so pip can fit it next to other packages;
    # every one of them is pinned with hashes for CI/Azure via the runtime inputs.
    project = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = [Requirement(d) for d in project["project"]["dependencies"]]
    assert all(not r.specifier and not r.marker for r in declared), declared
    runtime = {
        Requirement(r).name for r in read_requirements(ROOT / "requirements-runtime.in")
    }
    assert {r.name for r in declared} <= runtime


def test_invalid_metadata_stops_the_package_gate(tmp_path):
    for name in ("pyproject.toml", "README.md", "LICENSE", *TOPLEVEL):
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(
                source, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__")
            )
        else:
            shutil.copy2(source, tmp_path / name)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            '"Development Status :: 3 - Alpha"', '"Development Status :: Unknown"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(subprocess.CalledProcessError):
        check(tmp_path)
