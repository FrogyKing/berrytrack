"""Download checksum-verified public data. Run with uv run scripts/prepare_public_data.py."""

from argparse import ArgumentParser
from pathlib import Path

from berrytrack.data.public import prepare_blueberrydcm, prepare_butom

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["butom", "blueberrydcm"])
    parser.add_argument("--root", type=Path, default=Path("data"))
    arguments = parser.parse_args()
    if arguments.dataset == "butom":
        print(prepare_butom(arguments.root))
    else:
        print(prepare_blueberrydcm(arguments.root))
