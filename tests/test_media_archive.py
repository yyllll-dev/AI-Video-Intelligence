from pathlib import Path

import cv2
import numpy as np

from src.pipeline.media_archive import (
    FileKeyframeExtractor,
    RollingVideoRecorder,
    VideoClipExporter,
    prepare_browser_video,
    video_codec,
)


def make_video(path: Path, *, seconds: int = 12, fps: int = 2) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(seconds * fps):
        frame = np.full((48, 64, 3), index % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def video_duration(path: str) -> float:
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    return frames / fps


def read_image(path: str):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_file_keyframes_seek_across_long_event(tmp_path):
    video_path = tmp_path / "long.mp4"
    make_video(video_path, seconds=12, fps=2)
    extractor = FileKeyframeExtractor(video_path, tmp_path / "frames")

    paths = extractor.extract(0.0, 11.0, count=5)

    assert len(paths) == 5
    assert all(Path(path).is_file() for path in paths)
    first = read_image(paths[0]).mean()
    last = read_image(paths[-1]).mean()
    assert last > first


def test_long_event_replay_is_capped_highlight(tmp_path):
    video_path = tmp_path / "long.mp4"
    make_video(video_path, seconds=12, fps=2)
    exporter = VideoClipExporter(
        tmp_path / "replays",
        output_fps=2,
        max_clip_seconds=6,
    )

    replay_path = exporter.export_file(video_path, 0.0, 11.0)

    assert Path(replay_path).is_file()
    assert video_codec(replay_path) in {"h264", "avc1"}
    assert 4.5 <= video_duration(replay_path) <= 7.0


def test_uploaded_video_is_converted_for_browser(tmp_path):
    video_path = tmp_path / "upload.mp4"
    make_video(video_path, seconds=3, fps=2)

    preview_path = prepare_browser_video(video_path, tmp_path / "previews")

    assert Path(preview_path).is_file()
    assert video_codec(preview_path) in {"h264", "avc1"}
    assert 2.5 <= video_duration(preview_path) <= 3.5


def test_live_recorder_segments_and_exports_replay(tmp_path):
    recorder = RollingVideoRecorder(
        tmp_path / "recordings",
        segment_seconds=2.0,
        fps=2,
        max_width=64,
    )
    for index in range(12):
        recorder.add_frame(
            np.full((48, 64, 3), index, dtype=np.uint8),
            index / 2,
        )
    recorder.close()

    assert len(recorder.segments) == 3
    assert all(Path(segment.path).is_file() for segment in recorder.segments)

    exporter = VideoClipExporter(
        tmp_path / "replays",
        output_fps=2,
        max_clip_seconds=10,
    )
    replay_path = exporter.export_segments(recorder.segments, 0.5, 4.5)

    assert Path(replay_path).is_file()
    assert video_duration(replay_path) >= 3.0
