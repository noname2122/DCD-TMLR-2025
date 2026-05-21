"""Tiny helper: dump the code cells of a Jupyter notebook to a ``.py`` file.

Usage:
    python -m dcd.utils.extract_code notebooks/DCD.ipynb [more.ipynb ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_ipynb_to_py(ipynb_path: Path) -> Path:
    """Write a sibling ``.py`` file containing the code cells of ``ipynb_path``."""
    with ipynb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    out_path = ipynb_path.with_suffix(".py")
    with out_path.open("w", encoding="utf-8") as f:
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                f.write("".join(cell.get("source", [])))
                f.write("\n\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract code cells from Jupyter notebooks into .py files."
    )
    parser.add_argument(
        "notebooks",
        nargs="+",
        type=Path,
        help="One or more .ipynb files to convert.",
    )
    args = parser.parse_args()

    for nb_path in args.notebooks:
        out = convert_ipynb_to_py(nb_path)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
