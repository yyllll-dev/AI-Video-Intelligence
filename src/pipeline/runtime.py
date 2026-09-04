"""可直接运行的端到端视频分析流程。"""
from __future__ import annotations
import json
import queue
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2

from ..detection.detector import YoloDetector
from ..detection.video_source import VideoFrame, VideoSource, open_source
from ..event.engine import EventEngine
from ..event.schemas import Event
from ..retrieval import HashingEmbedder, InMemoryStore, VideoMemoryService
from ..tracking.tracker import SimpleTracker
from ..vlm.inference import (
    analyze_event,
    infer_activities_from_vlm_meta,
    should_persist_to_memory,
)
from ..vlm.qwen_vlm import load_model
from .frame_stream import FrameStream
from .media_archive import (
    FileKeyframeExtractor,
    LiveEvidenceCollector,
    RollingVideoRecorder,
    VideoClipExporter,
)
from .pipeline import VideoPipeline
from .video_buffer import VideoBuffer


EventAnalyzer = Callable[[Event, list[str]], tuple[Optional[Event], dict[str, Any]]]

_ACTIVITY_EVENT_TYPES = {
    "reading",
    "writing",
    "phone_learning",
    "computer_learning",
    "other_study_behavior",
    "phone_distraction",
    "computer_distraction",
    "communication_distraction",
}

class EndToEndRunner:
    """串联输入、视觉分析、事件、VLM、记忆和检索。"""

    def __init__(
        self,
        source: int | str | VideoSource,
        *,
        detector=None,
        tracker=None,
        event_engine: Optional[EventEngine] = None,
        event_analyzer: Optional[EventAnalyzer] = None,
        use_vlm: bool = True,
        qwen_model_path: Optional[str] = None,
        yolo_device: str = "cpu",
        yolo_confidence: float = 0.5,
        clips_dir: str | Path | None = None,
        recordings_dir: str | Path | None = None,
        replays_dir: str | Path | None = None,
        buffer_duration: float = 30.0,
        buffer_fps: float = 4.0,
        buffer_max_width: int = 448,
        analysis_fps: float = 4.0,
        keyframe_count: int = 9,
        recording_segment_seconds: float = 60.0,
        replay_max_seconds: float = 60.0,
        evidence_interval_seconds: float = 300.0,
    ) -> None:
        if buffer_fps <= 0 or analysis_fps <= 0:
            raise ValueError("buffer_fps 和 analysis_fps 必须大于 0")
        if keyframe_count <= 0:
            raise ValueError("keyframe_count 必须大于 0")

        self.source = source
        self.event_engine = event_engine or EventEngine()
        self.pipeline = VideoPipeline(
            detector=detector or YoloDetector(
                device=yolo_device,
                conf_threshold=yolo_confidence,
            ),
            tracker=tracker or SimpleTracker(),
            event_engine=self.event_engine,
        )

        project_root = Path(__file__).resolve().parents[2]
        clips_path = Path(clips_dir) if clips_dir is not None else project_root / "data" / "clips"
        recordings_path = (
            Path(recordings_dir)
            if recordings_dir is not None
            else project_root / "data" / "recordings"
        )
        replays_path = (
            Path(replays_dir)
            if replays_dir is not None
            else project_root / "data" / "replays"
        )

        self.buffer = VideoBuffer(
            buffer_duration=buffer_duration,
            clips_dir=clips_path,
        )

        self.file_keyframes = (
            FileKeyframeExtractor(source, clips_path)
            if isinstance(source, str)
            else None
        )

        self.replay_exporter = VideoClipExporter(
            replays_path,
            max_clip_seconds=replay_max_seconds,
        )

        is_camera = isinstance(source, int) or type(source).__name__ == "CameraSource"
        self.recorder = (
            RollingVideoRecorder(
                recordings_path,
                segment_seconds=recording_segment_seconds,
            )
            if is_camera
            else None
        )

        self.live_evidence = LiveEvidenceCollector(
            clips_path / "live_evidence",
            interval_seconds=evidence_interval_seconds,
        )

        self.memory_store = InMemoryStore()
        self.memory = VideoMemoryService(
            store=self.memory_store,
            embedder=HashingEmbedder(),
        )

        self.use_vlm = use_vlm
        self.qwen_model_path = qwen_model_path
        self.event_analyzer = event_analyzer

        self.buffer_interval = 1.0 / buffer_fps
        self.analysis_interval = 1.0 / analysis_fps
        self.buffer_max_width = buffer_max_width
        self.keyframe_count = keyframe_count

        self.last_buffer_time: Optional[float] = None
        self.last_analysis_time: Optional[float] = None
        self.last_timestamp = 0.0
        self.current_frame: Optional[VideoFrame] = None
        self.latest_result: dict[str, Any] = {}
        self.events: list[Event] = []
        self.rejected_events: list[dict[str, Any]] = []
        self.errors: list[str] = []

        self._is_realtime = is_camera
        self._stop_event = threading.Event()
        self._active_stream: Optional[FrameStream] = None

    def _ensure_event_analyzer(self) -> Optional[EventAnalyzer]:
        if not self.use_vlm:
            return None
        if self.event_analyzer is None:
            model, processor = load_model(model_path=self.qwen_model_path)

            def run(event: Event, paths: list[str]):
                return analyze_event(
                    model,
                    processor,
                    event,
                    paths,
                    return_meta=True,
                )

            self.event_analyzer = run
        return self.event_analyzer

    def _store_buffer_frame(self, video_frame: VideoFrame) -> None:
        if (
            self.last_buffer_time is not None
            and video_frame.timestamp - self.last_buffer_time < self.buffer_interval
        ):
            return

        frame = video_frame.frame
        if self.buffer_max_width > 0 and frame.shape[1] > self.buffer_max_width:
            scale = self.buffer_max_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                (self.buffer_max_width, max(1, round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self.buffer.add_frame(frame, video_frame.timestamp)
        self.last_buffer_time = video_frame.timestamp

    def _extract_event_frames(self, event: Event) -> list[str]:
        if self.file_keyframes is not None:
            return self.file_keyframes.extract(
                event.start_time,
                event.end_time,
                self.keyframe_count,
            )

        evidence_paths: list[str] = []
        if self.current_frame is not None:
            evidence_paths = self.live_evidence.finish(
                event.event_type,
                event.start_time,
                self.current_frame.frame,
                self.current_frame.timestamp,
            )

        earliest = self.buffer.earliest_timestamp
        latest = self.buffer.latest_timestamp
        if earliest is None or latest is None:
            return evidence_paths

        start_time = max(event.start_time, earliest)
        end_time = min(max(event.end_time, start_time), latest)
        recent_paths = self.buffer.get_frames(
            start_time=start_time,
            end_time=end_time,
            num_frames=self.keyframe_count,
        )

        paths = list(dict.fromkeys(evidence_paths + recent_paths))
        if len(paths) <= self.keyframe_count:
            return paths
        if self.keyframe_count == 1:
            return [paths[len(paths) // 2]]

        indexes = [
            round(index * (len(paths) - 1) / (self.keyframe_count - 1))
            for index in range(self.keyframe_count)
        ]
        return [paths[index] for index in indexes]

    def _create_replay(self, event: Event) -> str:
        return self.create_replay(event.start_time, event.end_time)

    def create_replay(self, start_time: float, end_time: float) -> str:
        """为展示层合并后的时间范围生成完整回放。"""
        try:
            if isinstance(self.source, str):
                return self.replay_exporter.export_file(
                    self.source,
                    start_time,
                    end_time,
                )
            if self.recorder is not None:
                self.recorder.flush()
                return self.replay_exporter.export_segments(
                    self.recorder.segments,
                    start_time,
                    end_time,
                )
        except Exception as exc:
            self.errors.append(f"生成 {start_time:.2f}s-{end_time:.2f}s 回放失败: {exc}")
        return ""

    def _handle_event(self, event: Event) -> None:
        paths = self._extract_event_frames(event)
        if not paths:
            self.errors.append(f"事件 {event.event_type} 关键帧提取失败")
            return

        meta: dict[str, Any] = {"event_confirmed": True, "vlm_skipped": True}
        confirmed_events: list[Event] = [event]
        analyzer = (
            self._ensure_event_analyzer()
            if event.event_type in _ACTIVITY_EVENT_TYPES
            else None
        )

        if analyzer is not None:
            try:
                print(
                    f"[Qwen] 正在识别活动候选 {event.event_type} "
                    f"({event.start_time:.2f}s -> {event.end_time:.2f}s)..."
                )
                confirmed_event, meta = analyzer(event, paths)
            except Exception as exc:
                self.errors.append(f"VLM 分析 {event.event_type} 失败: {exc}")
                return

            activities = infer_activities_from_vlm_meta(meta)
            if activities:
                confirmed_events = self._split_activity_event(
                    event,
                    activities,
                    meta.get("description", event.description),
                    activity_segments=meta.get("activity_segments", []),
                    frame_count=len(paths),
                )
                meta = {
                    **meta,
                    "event_confirmed": True,
                    "reclassified_from": event.event_type,
                    "reclassified_to": activities,
                }
                print(
                    f"[Qwen] 活动分类结果：{' -> '.join(activities)}"
                )
            elif confirmed_event is not None and should_persist_to_memory(meta):
                confirmed_events = [confirmed_event]
            else:
                confirmed_events = []

        if not confirmed_events:
            print(
                f"[事件跳过] Qwen 未确认 {event.event_type} "
                f"({event.start_time:.2f}s -> {event.end_time:.2f}s)"
            )
            self.rejected_events.append(
                {"event": asdict(event), "vlm": meta, "keyframes": paths}
            )
            return

        for confirmed_event in confirmed_events:
            self._remember_event(confirmed_event, paths, meta)

    @staticmethod
    def _split_activity_event(
        event: Event,
        activities: list[str],
        description: str,
        *,
        activity_segments: list[dict[str, Any]] | None = None,
        frame_count: int = 0,
    ) -> list[Event]:
        if not activities:
            return []
        duration = max(0.0, event.end_time - event.start_time)
        segment_starts: dict[str, float] = {}
        if activity_segments and frame_count > 1:
            for segment in activity_segments:
                event_type = segment.get("event_type")
                if event_type not in activities or event_type in segment_starts:
                    continue
                frame_index = max(1, min(frame_count, int(segment.get("start_frame", 1))))
                segment_starts[event_type] = event.start_time + (
                    (frame_index - 1) / (frame_count - 1)
                ) * duration

        if len(segment_starts) == len(activities):
            ordered = sorted(activities, key=segment_starts.get)
            starts = [segment_starts[event_type] for event_type in ordered]
        else:
            ordered = activities
            segment_duration = duration / len(activities)
            starts = [
                event.start_time + index * segment_duration
                for index in range(len(activities))
            ]

        events: list[Event] = []
        for index, event_type in enumerate(ordered):
            start_time = starts[index]
            end_time = (
                event.end_time
                if index == len(ordered) - 1
                else starts[index + 1]
            )
            events.append(
                Event(
                    event_type=event_type,
                    start_time=start_time,
                    end_time=end_time,
                    track_id=event.track_id,
                    confidence=event.confidence,
                    description=description,
                )
            )
        return events

    def _remember_event(
        self,
        event: Event,
        paths: list[str],
        meta: dict[str, Any],
    ) -> None:
        replay_path = self._create_replay(event)
        source_path = str(self.source) if isinstance(self.source, str) else "camera"
        record = self.memory.remember_event(
            event,
            caption=event.description,
            screenshot_path=paths[0],
            video_path=replay_path or source_path,
            metadata={
                "keyframes": paths,
                "vlm": meta,
                "source_video": source_path,
                "replay_mode": (
                    "full_event"
                    if event.end_time - event.start_time <= self.replay_exporter.max_clip_seconds
                    else "three_part_highlight"
                ),
            },
        )
        self.events.append(event)
        print(
            f"[事件] {record.event_type} "
            f"{record.start_time:.2f}s -> {record.end_time:.2f}s | {record.caption}"
        )

    def _on_frame(self, video_frame: VideoFrame) -> None:
        if self.recorder is not None:
            self.recorder.add_frame(video_frame.frame, video_frame.timestamp)
        self._store_buffer_frame(video_frame)

        if (
            self.last_analysis_time is not None
            and video_frame.timestamp - self.last_analysis_time < self.analysis_interval
        ):
            return
        self._analyze_frame(video_frame)

    def _analyze_frame(self, video_frame: VideoFrame) -> None:
        self.current_frame = video_frame
        self.last_timestamp = video_frame.timestamp
        self.last_analysis_time = video_frame.timestamp
        self.latest_result = self.pipeline.process_frame(video_frame)

        for event in self.latest_result["events"]:
            self._handle_event(event)

        self.live_evidence.observe(
            self.event_engine.current_activity,
            self.event_engine.activity_start_time,
            video_frame.frame,
            video_frame.timestamp,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._active_stream is not None:
            self._active_stream.stop()

    def _run_realtime(
        self,
        *,
        max_frames: Optional[int],
        max_duration: Optional[float],
    ) -> dict[str, Any]:
        source = open_source(self.source) if isinstance(self.source, int) else self.source
        owns_source = isinstance(self.source, int)
        frame_queue: queue.Queue = queue.Queue(maxsize=2)
        sentinel = object()
        first_timestamp: Optional[float] = None
        last_timestamp: Optional[float] = None
        processed = 0
        last_enqueued: Optional[float] = None

        def analysis_worker() -> None:
            while True:
                item = frame_queue.get()
                try:
                    if item is sentinel:
                        return
                    self._analyze_frame(item)
                except Exception as exc:
                    self.errors.append(f"实时分析失败: {exc}")
                finally:
                    frame_queue.task_done()

        worker = threading.Thread(target=analysis_worker, name="video-analysis", daemon=True)
        worker.start()

        start_wall = time.monotonic()
        try:
            while not self._stop_event.is_set():
                if max_frames is not None and processed >= max_frames:
                    break
                video_frame = source.read()
                if video_frame is None:
                    break

                if first_timestamp is None:
                    first_timestamp = video_frame.timestamp
                if (
                    max_duration is not None
                    and video_frame.timestamp - first_timestamp >= max_duration
                ):
                    break

                last_timestamp = video_frame.timestamp
                processed += 1
                self.last_timestamp = video_frame.timestamp

                if self.recorder is not None:
                    self.recorder.add_frame(video_frame.frame, video_frame.timestamp)
                self._store_buffer_frame(video_frame)

                if (
                    last_enqueued is not None
                    and video_frame.timestamp - last_enqueued < self.analysis_interval
                ):
                    continue
                last_enqueued = video_frame.timestamp

                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(video_frame)
        except KeyboardInterrupt:
            self._stop_event.set()
        finally:
            if owns_source:
                source.release()
            frame_queue.put(sentinel)
            worker.join()

        return {
            "frames": float(processed),
            "start_time": first_timestamp or 0.0,
            "end_time": last_timestamp or 0.0,
            "duration": (
                last_timestamp - first_timestamp
                if first_timestamp is not None and last_timestamp is not None
                else 0.0
            ),
            "wall_duration": time.monotonic() - start_wall,
        }

    def run(
        self,
        *,
        max_frames: Optional[int] = None,
        max_duration: Optional[float] = None,
    ) -> dict[str, Any]:
        self._stop_event.clear()
        if self._is_realtime:
            stats = self._run_realtime(
                max_frames=max_frames,
                max_duration=max_duration,
            )
        else:
            stream = FrameStream(source=self.source, on_frame=self._on_frame)
            self._active_stream = stream
            try:
                stats = stream.run(max_frames=max_frames, progress_every=30)
            finally:
                self._active_stream = None

        try:
            if self.last_analysis_time is not None:
                for event in self.event_engine.finalize(self.last_timestamp):
                    self._handle_event(event)
        finally:
            if self.recorder is not None:
                self.recorder.close()

        return {
            "stream": stats,
            "events_detected": len(self.events) + len(self.rejected_events),
            "memories_saved": len(self.memory_store),
            "events": [asdict(event) for event in self.events],
            "rejected_events": self.rejected_events,
            "errors": self.errors,
        }

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return [result.to_dict() for result in self.memory.search(query, top_k=top_k)]


def save_run_report(
    result: dict[str, Any],
    search_results: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / (
        "analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    report_path.write_text(
        json.dumps(
            {"analysis": result, "search_results": search_results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path
