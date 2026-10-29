"""Build an allowlisted Azure archive from already validated source files."""

from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(destination: Path) -> None:
    # my_viz_plotly.py imports edalens.app, so the whole package ships.
    files = [ROOT / "my_viz_plotly.py", *sorted((ROOT / "edalens").rglob("*.py"))]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            info = zipfile.ZipInfo(
                source.relative_to(ROOT).as_posix(), (1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())
        # Azure receives the locked runtime requirements, never development tools.
        info = zipfile.ZipInfo("requirements.txt", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(
            info,
            "--require-hashes\n" + (ROOT / "requirements-runtime.lock").read_text(),
        )


if __name__ == "__main__":
    build(ROOT / "dist" / "release.zip")
