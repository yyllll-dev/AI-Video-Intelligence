# AI-Video-Intelligence
Real-time local video intelligence system for Intel AI PC

## Current pipeline

`VideoPipeline` now connects detector output, optional tracking, and the event
engine. A detector returns dictionaries with `class_name`, `confidence`, and
`bbox`; it may also include `track_id`. If a tracker is supplied, it receives
the normalized detections and the frame timestamp. Every call to
`process_frame(frame, timestamp)` returns the detections, `TrackingResult`
objects, and newly emitted events.

Run the complete entry point from the repository root:

```powershell
python -m src.main --source path\to\video.mp4 --qwen-model-path path\to\Qwen2-VL-2B-Instruct --yolo-device 0
```

Use `--source 0` for the first camera. Use `--no-vlm` only when checking the
YOLO/Tracking/Event half of the pipeline without loading Qwen2-VL. Keyframes
are written to `data/clips/` and the final JSON report is written to
`data/outputs/`.

To use the visual demo, set `QWEN_VL_MODEL_PATH` and run:

```powershell
python demo/app.py
```

Upload a video under "本地视频文件", then click "开始分析". The event cards and
natural-language search use the real runtime after analysis completes.

## Long videos and replay

- Uploaded files are sampled directly from the original video by timestamp, so
  long events do not depend on the 30-second in-memory buffer.
- Camera input is compressed into one-minute MP4 chunks under `data/recordings/`.
- Camera capture/recording runs independently from YOLO and Qwen inference, so
  a slow semantic analysis does not pause the archived video.
- Confirmed short events get a complete replay clip under `data/replays/`.
- Events longer than 60 seconds get a three-part highlight containing their
  beginning, middle, and end instead of duplicating the entire long recording.

The visual demo can analyze either an uploaded video or the computer's first
camera. Camera runs use the duration selected on the page and can be stopped
from the UI.

The event engine accepts a timestamp even for an empty frame, so absence-based
events continue to advance when the detector finds no objects.
