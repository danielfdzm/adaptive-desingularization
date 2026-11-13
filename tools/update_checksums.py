#!/usr/bin/env python3
"""Regenerate the distribution manifest after an intentional repository edit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce import digest, inventory, load_catalog


def main():
    load_catalog()
    files = inventory(ROOT, release=True)
    lines = [f"{digest(path)}  {name}\n" for name, path in sorted(files.items())]
    (ROOT / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    print(f"Wrote SHA256SUMS for {len(files):,} files.")


if __name__ == "__main__":
    main()
