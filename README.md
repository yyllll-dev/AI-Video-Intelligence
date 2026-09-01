# AI-Video-Intelligence
Real-time local video intelligence system for Intel AI PC

## Current pipeline

`VideoPipeline` now connects detector output, optional tracking, and the event
engine. A detector returns dictionaries with `class_name`, `confidence`, and
`bbox`; it may also include `track_id`. If a tracker is supplied, it receives
the normalized detections and the frame timestamp. Every call to
`process_frame(frame, timestamp)` returns the detections, `TrackingResult`
objects, and newly emitted events.

Run the entry point from the repository root:

```powershell
python -m src.main
```

The event engine accepts a timestamp even for an empty frame, so absence-based
events continue to advance when the detector finds no objects.
