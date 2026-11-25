"""Assemble build/space/: only what the Hugging Face Space needs (no data, tests, results).

    python scripts/build_space.py
    hf upload <user>/<space> build/space . --repo-type space

The Space README (with the Spaces config block at the top) comes from docs/space_README.md.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "space"
FILES = ["Dockerfile", "requirements.txt", "LICENSE", ".dockerignore"]
DIRS = ["api", "app", "neurovoice", "models"]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f in FILES:
        shutil.copy2(ROOT / f, OUT / f)
    for d in DIRS:
        shutil.copytree(ROOT / d, OUT / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "docs" / "space_README.md", OUT / "README.md")
    files = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file())
    size = sum((OUT / f).stat().st_size for f in files) / 1e6
    print(f"{len(files)} files, {size:.1f} MB in build/space")
    for f in files:
        print(" ", f)


if __name__ == "__main__":
    main()
