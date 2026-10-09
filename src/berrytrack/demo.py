"""Local Gradio interface; opening the UI never allocates cloud compute."""

import json
import tempfile
from pathlib import Path

import gradio as gr

from berrytrack.config import TrackerConfig
from berrytrack.pipeline import process_video


def run_demo(video, algorithm, compare, weights, confidence, cache, gt, frame_limit):
    if not video:
        raise gr.Error("Upload a video first.")
    if not cache and not weights.strip():
        raise gr.Error("Select detector weights or upload cached detections.")
    detector = None
    if not cache:
        from berrytrack.detection.yolo import YoloDetector

        detector = YoloDetector(weights.strip(), confidence=float(confidence))
    directory = Path(tempfile.mkdtemp(prefix="berrytrack-"))
    result = process_video(
        Path(video),
        directory,
        TrackerConfig(algorithm=algorithm),
        detector=detector,
        cached_detections=Path(cache) if cache else None,
        compare=compare if compare != "none" else None,
        ground_truth=Path(gt) if gt else None,
        max_frames=int(frame_limit),
        max_seconds=600,
    )
    events = []
    for row in result["results"]:
        for event in row.get("metrics", {}).get("identity_switch_events", []):
            events.append(
                [
                    row["algorithm"],
                    event["frame"],
                    event["gt_id"],
                    event["previous_id"],
                    event["new_id"],
                ]
            )
    gallery = []
    if events:
        import cv2

        capture = cv2.VideoCapture(result["video"])
        for event in events[:6]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, event[1] - 1)
            valid, image = capture.read()
            if valid:
                gallery.append(
                    (
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                        f"{event[0]} · frame {event[1]} · GT {event[2]}",
                    )
                )
        capture.release()
    files = [result["video"], result["statistics"], *[r["mot"] for r in result["results"]]]
    table = [
        [
            r["algorithm"],
            r["unique_count"],
            round(r["association_ms"], 2),
            round(r["end_to_end_ms"], 2),
            round(r["video_fps"], 2),
        ]
        for r in result["results"]
    ]
    return result["video"], table, events, gallery, files, json.dumps(result, indent=2)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="BerryTrack · Fruit identity lab") as demo:
        gr.HTML(
            "<div class='hero'><span class='eyebrow'>AGRICULTURAL VISION / RESEARCH LAB</span>"
            "<h1>Follow every fruit.</h1><p>Inspect identity association, trajectories and "
            "cumulative counts in dense field scenes.</p>"
            "<span class='mode'>LOCAL CPU · no cloud instances</span></div>"
        )
        with gr.Row():
            with gr.Column(scale=4):
                video = gr.Video(label="Agricultural video", sources=["upload"])
                with gr.Row():
                    algorithm = gr.Dropdown(
                        ["berrytrack", "bytetrack", "botsort", "deepsort"],
                        value="berrytrack",
                        label="Tracker",
                    )
                    compare = gr.Dropdown(
                        ["none", "bytetrack", "botsort", "deepsort"],
                        value="botsort",
                        label="Compare against",
                    )
                weights = gr.Textbox(
                    value="checkpoints/blueberry-via/best.pt",
                    label="Detector weights",
                    info="Use adapted weights; COCO weights are unreliable for blueberries.",
                )
                confidence = gr.Slider(
                    0.01, 0.9, value=0.1, step=0.01, label="Detection confidence"
                )
                frame_limit = gr.Slider(10, 3000, value=300, step=10, label="Maximum frames")
                with gr.Accordion("Cached detections and verified ground truth", open=False):
                    gr.Markdown(
                        "MOT files use 1-based frames and xywh coordinates. Reference boxes "
                        "isolate association and cannot measure detector quality. "
                        "GT must cover every processed frame."
                    )
                    cache = gr.File(
                        label="Cached detections (optional)", file_types=[".txt"], type="filepath"
                    )
                    gt = gr.File(
                        label="Verified tracking GT (optional)",
                        file_types=[".txt"],
                        type="filepath",
                    )
                run = gr.Button("Track & compare", variant="primary")
            with gr.Column(scale=6):
                rendered = gr.Video(label="Persistent IDs & trajectories", interactive=False)
                summary = gr.Dataframe(
                    headers=[
                        "Algorithm",
                        "Unique IDs",
                        "Association ms",
                        "Pipeline ms",
                        "Pipeline FPS",
                    ],
                    label="Measured timing & counts",
                    interactive=False,
                )
                downloads = gr.Files(label="Export video, MOT tracks & statistics")
        with gr.Accordion("Identity switch inspection · requires verified GT", open=False):
            events = gr.Dataframe(
                headers=["Algorithm", "Frame", "GT identity", "Previous ID", "New ID"],
                interactive=False,
            )
            gallery = gr.Gallery(label="Measured switch frames", columns=2, interactive=False)
            gr.Markdown(
                "Use the frame number to inspect the exported video. These events follow "
                "CLEAR matching; a physical identity is known only where verified GT exists."
            )
        with gr.Accordion("Run details", open=False):
            details = gr.Code(language="json", label="Reproducibility record")
        gr.Markdown(
            "**Research status:** counts represent confirmed tracker identities, not "
            "guaranteed physical fruit totals. Occlusion, parallax and re-entry can create "
            "duplicate counts. GPU jobs run separately through bounded Modal commands; "
            "see the repository guide."
        )
        run.click(
            run_demo,
            [video, algorithm, compare, weights, confidence, cache, gt, frame_limit],
            [rendered, summary, events, gallery, downloads, details],
            api_name="track",
        )
    return demo


def main() -> None:
    css = """
    .gradio-container{max-width:1400px!important}
    .hero{padding:30px;border-radius:20px;background:linear-gradient(120deg,#241744,#163a3b);
          color:#f3efff;margin-bottom:22px}
    .hero h1{font-size:46px;margin:8px 0}.hero p{font-size:17px}
    .eyebrow{letter-spacing:3px;font-size:11px;color:#c2a9ed}
    .mode{display:inline-block;border:1px solid #658c83;padding:6px 12px;
          border-radius:20px;font-size:11px}
    """
    build_demo().queue(max_size=4, default_concurrency_limit=1).launch(
        server_name="127.0.0.1", share=False, theme=gr.themes.Soft(primary_hue="violet"), css=css
    )


if __name__ == "__main__":
    main()
