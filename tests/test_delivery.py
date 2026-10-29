"""Ensure only reviewed runtime files are included in the Azure archive."""

import zipfile
from pathlib import Path

import yaml

from scripts.build_deployment import build

ROOT = Path(__file__).resolve().parents[1]


def test_every_azure_deployment_requires_the_shared_validation_job():
    workflows = list((ROOT / ".github/workflows").glob("main_*.yml"))
    assert len(workflows) == 2
    for path in workflows:
        workflow = yaml.safe_load(path.read_text())
        jobs = workflow["jobs"]
        assert jobs["validation"]["uses"] == "./.github/workflows/validate.yml"
        assert jobs["deploy"]["needs"] == "validation"
        assert "if" not in jobs["deploy"]  # Do not allow always() to bypass failure.
        downloads = [
            step
            for step in jobs["deploy"]["steps"]
            if step.get("uses", "").startswith("actions/download-artifact@")
        ]
        assert len(downloads) == 1
        assert downloads[0]["with"]["name"] == "validated-app"
        deployments = [
            step
            for step in jobs["deploy"]["steps"]
            if step.get("uses", "").startswith("azure/webapps-deploy@")
        ]
        assert len(deployments) == 1
        assert deployments[0]["with"]["package"] == "release.zip"


def test_deployment_archive_is_allowlisted_and_repeatable(tmp_path):
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    build(first)
    build(second)
    assert first.read_bytes() == second.read_bytes()
    expected = {"my_viz_plotly.py", "requirements.txt"}
    expected.update(
        p.relative_to(ROOT).as_posix() for p in (ROOT / "edalens").rglob("*.py")
    )
    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == expected
        assert archive.read("requirements.txt").decode() == (
            "--require-hashes\n" + (ROOT / "requirements-runtime.lock").read_text()
        )
        for name in expected - {"requirements.txt"}:
            assert archive.read(name) == (ROOT / name).read_bytes()
