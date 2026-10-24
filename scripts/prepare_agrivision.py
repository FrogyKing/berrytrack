"""Download at most40 contiguous AgriVision DB-2 frames with HTTP ranges."""

from argparse import ArgumentParser
from pathlib import Path

from berrytrack.data.agrivision import prepare_agrivision, prepare_visual_reference

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/agrivision-camera1-video1")
    )
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--video", type=int, default=1)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--playback-fps", type=float, default=10)
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument(
        "--reference-output",
        type=Path,
        help="Also reproduce the frozen visual reference 8-frame crop; only camera1/video1/start1",
    )
    args = parser.parse_args()
    print(
        prepare_agrivision(
            args.output,
            args.camera,
            args.video,
            args.start,
            args.frames,
            args.playback_fps,
            masks=not args.no_masks,
        )
    )
    if args.reference_output:
        print(prepare_visual_reference(args.output, args.reference_output))
