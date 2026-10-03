"""Create the audited VIA-derived BlueberryDCM detection development split."""

import argparse
from pathlib import Path

from berrytrack.detection.training import prepare_detection_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/processed/blueberrydcm"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/blueberrydcm-via-yolo"))
    parser.add_argument("--max-train", type=int, default=32)
    args = parser.parse_args()
    print(prepare_detection_split(args.root, args.output, args.max_train))
