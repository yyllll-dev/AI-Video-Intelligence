"""长视频抽帧、实时分段录像与事件回放。"""

from __future__ import annotations

import threading
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BROWSER_CODECS = {"avc1", "h264", "x264"}


def _uniform_times(start_time: float, end_time: float, count: int) -> list[float]:
    if count <= 1 or end_time <= start_time:
        return [start_time]
    step = (end_time - start_time) / (count - 1)
    return [start_time + index * step for index in range(count)]


def _resize(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    return cv2.resize(
        frame,
        (max_width, max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _even_size(frame: np.ndarray) -> np.ndarray:
    """H.264 编码要求宽高为偶数。"""
    height, width = frame.shape[:2]
    return frame[: height - height % 2, : width - width % 2]


def _writer_path(path: Path) -> str:
    """规避部分 Windows OpenCV 后端无法写入中文绝对路径的问题。"""
    try:
        return os.path.relpath(path.resolve(), Path.cwd().resolve())
    except ValueError:
        return str(path)


class _RelocatingWriter:
    """编码结束后把英文临时文件移动到原目标路径。"""

    def __init__(self, writer, temporary_path: Path, target_path: Path) -> None:
        self._writer = writer
        self._temporary_path = temporary_path
        self._target_path = target_path

    def isOpened(self) -> bool:
        return self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()
        if self._temporary_path.exists():
            self._target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self._temporary_path), str(self._target_path))


def _open_browser_writer(path: Path, fps: float, frame: np.ndarray):
    frame = _even_size(frame)
    height, width = frame.shape[:2]
    api = cv2.CAP_MSMF if os.name == "nt" else cv2.CAP_ANY
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(
        _writer_path(path), api, fourcc, max(float(fps), 1.0), (width, height)
    )
    if not writer.isOpened() and os.name == "nt":
        writer.release()
        temporary_path = Path.cwd() / f"video_encode_{uuid4().hex}.mp4"
        writer = cv2.VideoWriter(
            temporary_path.name,
            api,
            fourcc,
            max(float(fps), 1.0),
            (width, height),
        )
        if writer.isOpened():
            return _RelocatingWriter(writer, temporary_path, path)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            "无法创建浏览器可播放的 H.264 视频，请确认当前 OpenCV 支持 avc1 编码"
        )
    return writer


def video_codec(path: str | Path) -> str:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return ""
    value = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4)).lower()


def prepare_browser_video(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    max_width: int = 1280,
) -> str:
    """把上传视频转为浏览器可直接播放的 H.264 MP4。"""
    source_path = Path(source_path)
    if source_path.suffix.lower() == ".mp4" and video_codec(source_path) in _BROWSER_CODECS:
        return str(source_path.resolve())

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频文件: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"preview_{uuid4().hex[:12]}.mp4"
    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame = _even_size(_resize(frame, max_width))
            if writer is None:
                writer = _open_browser_writer(output_path, fps, frame)
            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if writer is None or not output_path.exists():
        raise RuntimeError(f"视频转换失败: {source_path}")
    return str(output_path.resolve())


class FileKeyframeExtractor:
    """直接从原视频按事件时间定位，不依赖短期 VideoBuffer。"""

    def __init__(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        *,
        max_width: int = 448,
    ) -> None:
        self.source_path = Path(source_path)
        self.output_dir = Path(output_dir)
        self.max_width = int(max_width)

    def extract(
        self,
        start_time: float,
        end_time: float,
        count: int = 5,
    ) -> list[str]:
        capture = cv2.VideoCapture(str(self.source_path))
        if not capture.isOpened():
            raise ValueError(f"无法打开视频文件: {self.source_path}")

        event_dir = self.output_dir / f"event_{uuid4().hex[:12]}"
        event_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        try:
            for index, timestamp in enumerate(
                _uniform_times(float(start_time), float(end_time), count),
                start=1,
            ):
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame = _resize(frame, self.max_width)
                image_path = event_dir / f"frame_{index:03d}.jpg"
                encoded, image_data = cv2.imencode(".jpg", frame)
                if encoded:
                    image_data.tofile(str(image_path))
                    paths.append(str(image_path.resolve()))
        finally:
            capture.release()
        return paths


@dataclass(frozen=True, slots=True)
class RecordingSegment:
    path: str
    start_time: float
    end_time: float


class VideoClipExporter:
    """生成短事件完整回放或长事件摘要回放。"""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        output_fps: float = 15.0,
        max_width: int = 960,
        max_clip_seconds: float = 60.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_fps = float(output_fps)
        self.max_width = int(max_width)
        self.max_clip_seconds = float(max_clip_seconds)

    def _ranges(self, start_time: float, end_time: float) -> list[tuple[float, float]]:
        start_time = max(0.0, float(start_time))
        end_time = max(start_time, float(end_time))
        duration = end_time - start_time
        if duration <= self.max_clip_seconds:
            return [(start_time, end_time)]

        window = self.max_clip_seconds / 3.0
        centers = [
            start_time + window / 2.0,
            (start_time + end_time) / 2.0,
            end_time - window / 2.0,
        ]
        return [
            (max(start_time, center - window / 2.0), min(end_time, center + window / 2.0))
            for center in centers
        ]

    def export_file(self, source_path: str | Path, start_time: float, end_time: float) -> str:
        return self._export_sources(
            [(str(source_path), 0.0, float("inf"))],
            self._ranges(start_time, end_time),
        )

    def export_segments(
        self,
        segments: list[RecordingSegment],
        start_time: float,
        end_time: float,
    ) -> str:
        sources = [(item.path, item.start_time, item.end_time) for item in segments]
        return self._export_sources(sources, self._ranges(start_time, end_time))

    def _export_sources(
        self,
        sources: list[tuple[str, float, float]],
        ranges: list[tuple[float, float]],
    ) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"replay_{uuid4().hex[:12]}.mp4"
        writer = None
        writer_fps: Optional[float] = None

        try:
            for range_start, range_end in ranges:
                for path, source_start, source_end in sources:
                    overlap_start = max(range_start, source_start)
                    overlap_end = min(range_end, source_end)
                    if overlap_end <= overlap_start:
                        continue

                    capture = cv2.VideoCapture(path)
                    if not capture.isOpened():
                        continue
                    source_fps = capture.get(cv2.CAP_PROP_FPS)
                    source_fps = source_fps if source_fps and source_fps > 0 else self.output_fps
                    local_start = max(0.0, overlap_start - source_start)
                    capture.set(cv2.CAP_PROP_POS_MSEC, local_start * 1000.0)
                    frame_index = 0
                    next_write_time = overlap_start
                    try:
                        while True:
                            ok, frame = capture.read()
                            if not ok or frame is None:
                                break
                            timestamp = overlap_start + frame_index / source_fps
                            frame_index += 1
                            if timestamp >= overlap_end - 1e-6:
                                break
                            if timestamp + 1e-6 < next_write_time:
                                continue
                            frame = _even_size(_resize(frame, self.max_width))
                            if writer is None:
                                writer_fps = min(self.output_fps, source_fps)
                                writer = _open_browser_writer(output_path, writer_fps, frame)
                            writer.write(frame)
                            next_write_time += 1.0 / writer_fps
                    finally:
                        capture.release()
        finally:
            if writer is not None:
                writer.release()

        if writer is None or not output_path.exists():
            return ""
        return str(output_path.resolve())


class RollingVideoRecorder:
    """把实时摄像头压缩为固定时长的 MP4 分段。"""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        segment_seconds: float = 60.0,
        fps: float = 15.0,
        max_width: int = 1280,
    ) -> None:
        self.session_dir = Path(output_dir) / (
            "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.segment_seconds = float(segment_seconds)
        self.fps = float(fps)
        self.max_width = int(max_width)
        self.segments: list[RecordingSegment] = []
        self._writer = None
        self._segment_path: Optional[Path] = None
        self._segment_start: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._last_written: Optional[float] = None
        self._segment_index = 0
        self._lock = threading.RLock()

    def add_frame(self, frame: np.ndarray, timestamp: float) -> None:
        with self._lock:
            if (
                self._last_written is not None
                and timestamp - self._last_written < 1.0 / self.fps
            ):
                return
            if (
                self._writer is not None
                and self._segment_start is not None
                and timestamp - self._segment_start >= self.segment_seconds
            ):
                self._close_segment()

            frame = _even_size(_resize(frame, self.max_width))
            if self._writer is None:
                self._open_segment(frame, timestamp)
            self._writer.write(frame)
            self._last_timestamp = float(timestamp)
            self._last_written = float(timestamp)

    def _open_segment(self, frame: np.ndarray, timestamp: float) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._segment_index += 1
        self._segment_path = self.session_dir / f"chunk_{self._segment_index:05d}.mp4"
        self._writer = _open_browser_writer(self._segment_path, self.fps, frame)
        self._segment_start = float(timestamp)

    def _close_segment(self) -> None:
        if self._writer is None:
            return
        self._writer.release()
        if self._segment_path is not None and self._segment_start is not None:
            self.segments.append(
                RecordingSegment(
                    path=str(self._segment_path.resolve()),
                    start_time=self._segment_start,
                    end_time=self._last_timestamp or self._segment_start,
                )
            )
        self._writer = None
        self._segment_path = None
        self._segment_start = None

    def flush(self) -> None:
        with self._lock:
            self._close_segment()

    def close(self) -> None:
        self.flush()


class LiveEvidenceCollector:
    """长实时事件只保存开始、定时和结束证据帧。"""

    def __init__(self, output_dir: str | Path, interval_seconds: float = 300.0) -> None:
        self.output_dir = Path(output_dir)
        self.interval_seconds = float(interval_seconds)
        self._events: dict[tuple[str, float], list[tuple[float, str]]] = {}

    def observe(
        self,
        event_type: Optional[str],
        start_time: Optional[float],
        frame: np.ndarray,
        timestamp: float,
    ) -> None:
        if event_type is None or start_time is None:
            return
        key = (event_type, float(start_time))
        entries = self._events.setdefault(key, [])
        if entries and timestamp - entries[-1][0] < self.interval_seconds:
            return
        self._save(key, frame, timestamp)

    def finish(self, event_type: str, start_time: float, frame: np.ndarray, timestamp: float) -> list[str]:
        # 同一窗口内候选类型可能多次变化；候选只是弱提示，证据应按窗口
        # 合并，并一次性清理，不能只弹出最终胜出类型而留下陈旧键。
        keys = [
            key for key in self._events
            if abs(key[1] - float(start_time)) <= 1e-6
        ]
        if not keys:
            return []
        key = next((item for item in keys if item[0] == event_type), keys[0])
        entries = self._events[key]
        if not entries or timestamp - entries[-1][0] > 1.0:
            self._save(key, frame, timestamp)
        combined = []
        for candidate_key in keys:
            combined.extend(self._events.pop(candidate_key, []))
        combined.sort(key=lambda item: item[0])
        return list(dict.fromkeys(path for _, path in combined))

    def _save(self, key: tuple[str, float], frame: np.ndarray, timestamp: float) -> None:
        event_dir = self.output_dir / f"live_{key[0]}_{key[1]:.3f}".replace(".", "_")
        event_dir.mkdir(parents=True, exist_ok=True)
        image_path = event_dir / f"evidence_{len(self._events[key]) + 1:03d}.jpg"
        encoded, image_data = cv2.imencode(".jpg", frame)
        if encoded:
            image_data.tofile(str(image_path))
            self._events[key].append((float(timestamp), str(image_path.resolve())))
