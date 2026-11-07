"""Build a source-and-data ZIP without private history or generated outputs."""

from __future__ import annotations

import argparse
import hashlib
import tomllib
import zipfile
from pathlib import Path

from stgrape.archive import verify_archive
from stgrape.paths import ensure_output_path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"src", "tests", "experiments", "docs", "scripts", "data", "notebooks", ".github"}
IGNORED = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".git",
    ".codex",
    ".vscode",
    ".venv",
}
ROOT_FILES = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CHANGELOG.md",
    ".gitignore",
    ".gitattributes",
    "environment.yml",
    "pyproject.toml",
    "requirements-reproduction.txt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    verify_archive(ROOT / "data")
    output = ensure_output_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    prefix = f"ST-GRAPE-public-v{version}"
    target = output / f"{prefix}.zip"
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink() or any(p in IGNORED for p in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name in {".DS_Store", "Thumbs.db"}:
            continue
        if path.name.startswith(".env"):
            continue
        if (len(relative.parts) == 1 and path.name in ROOT_FILES) or relative.parts[
            0
        ] in DIRECTORIES:
            files.append(path)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(Path(prefix) / path.relative_to(ROOT)))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed.")
    artifacts = sorted(
        p for p in output.iterdir() if p.suffix in {".zip", ".whl"} or p.name.endswith(".tar.gz")
    )
    checksums = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in artifacts
    )
    (output / "SHA256SUMS").write_text(checksums, encoding="ascii")
    print(f"Built {target.name}: {len(files)} files, {target.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
