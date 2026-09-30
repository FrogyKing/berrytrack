"""uv run scripts/modal_job.py smoke --seconds 120"""

import argparse
import json
from pathlib import Path

from berrytrack.infrastructure.launch import launch

parser = argparse.ArgumentParser(description="Bounded ephemeral Modal GPU jobs")
parser.add_argument(
    "kind", choices=["smoke", "train-encoder", "infer-encoder", "train-detector", "infer-detector"]
)
parser.add_argument("--gpu", choices=["T4", "L4", "A10G"], default="T4")
parser.add_argument("--seconds", type=int, default=300)
parser.add_argument("--options", default="{}", help="JSON job arguments; paths relative to volumes")
parser.add_argument("--options-file", type=Path, help="JSON options file, convenient on Windows")
if __name__ == "__main__":
    args = parser.parse_args()
    options = json.loads(args.options_file.read_text() if args.options_file else args.options)
    launch(args.kind, options, args.gpu, args.seconds)
