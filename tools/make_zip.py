"""Build the submission ZIP (excludes venvs, download-caches, pycaches).

Keeps: all src/, app, tests, tools, data/audio + data/gita (bundled synthetic
samples), data/video/movie_30min.mp4 (the processed 30-min input), cache/p2
models, cache/p1_alpha.json, cache/p3 artefacts (so the demo runs offline).
Excludes: .venv, data/video/raw_* (~170 MB re-downloadable via
src/video/p3_prepare.py), __pycache__, cache/walkthrough temp video, logs.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ZIP = OUT / "Atul Tejaswi_Project.zip"
EXCLUDE_DIRS = {"__pycache__", ".venv", "venv", "logs", ".pytest_cache"}
EXCLUDE_PREFIX = ("data/video/raw_",)
EXCLUDE_SUFFIX_GLOBS = ("cache/walkthrough/*", "cache/app/*")


def wanted(rel: Path) -> bool:
    parts = rel.parts
    if rel.name == "Atul Tejaswi_Project.zip":
        return False
    if any(p in EXCLUDE_DIRS for p in parts):
        return False
    s = rel.as_posix()
    if s.startswith(EXCLUDE_PREFIX):
        return False
    for g in EXCLUDE_SUFFIX_GLOBS:
        sfx = g.rstrip("*")
        if s.startswith(sfx):
            return False
    if s.endswith(".pyc") or "__pycache__" in s:
        return False
    return True


def main():
    files = [p for p in OUT.rglob("*") if p.is_file()
             and wanted(p.relative_to(OUT))]
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            z.write(p, Path("project") / p.relative_to(OUT))
    mb = ZIP.stat().st_size / 2 ** 20
    print(f"[zip] {ZIP}  {mb:.1f} MB  ({len(files)} files)")


if __name__ == "__main__":
    main()
