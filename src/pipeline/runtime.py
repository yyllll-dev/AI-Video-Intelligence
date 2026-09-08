"""可直接运行的端到端视频分析流程。"""
from __future__ import annotations
from collections import deque
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2

from ..detection.detector import YoloDetector
from ..detection.video_source import VideoFrame, VideoSource, open_source
from ..event.engine import EventEngine
from ..event.event_types import (
    ALL_EVENTS,
)
from ..event.schemas import Event
from ..retrieval import HashingEmbedder, InMemoryStore, VideoMemoryService
from ..tracking.tracker import SimpleTracker
from ..vlm.inference import (
    analyze_event,
    analyze_first_transition_frame,
    analyze_frame_labels,
    analyze_position_transition,
    analyze_window_label,
    description_conflicts_with_event,
    description_has_transition_evidence,
    infer_activity_from_description,
    infer_activities_from_vlm_meta,
    frame_labels_to_segments,
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
BoundaryAnalyzer = Callable[[list[str]], dict[str, Any]]
PositionAnalyzer = Callable[[list[str], str], dict[str, Any]]
WindowLabelAnalyzer = Callable[[list[str]], dict[str, Any]]
TransitionFrameAnalyzer = Callable[[list[str], str, str], dict[str, Any]]
SummaryAnalyzer = Callable[[list[dict[str, Any]]], str]

_SUMMARY_ACTION_CN = {
    "sit_at_study_position": "坐到学习位置",
    "leave_study_position": "离开学习位置",
    "reading": "阅读",
    "writing": "书写",
    "phone_usage": "使用手机",
    "computer_usage": "使用电脑",
    "communication_distraction": "与他人交流",
}

_NON_OBJECTIVE_SUMMARY_TERMS = (
    "可能",
    "似乎",
    "也许",
    "或许",
    "大概",
    "推测",
    "看起来",
    "显示出",
    "表明",
    "体现",
    "兴趣",
    "态度",
    "习惯",
    "专注力",
    "学习效果",
    "确保",
)

_VLM_EVENT_TYPES = frozenset(ALL_EVENTS)
_BEHAVIOR_EVENT_TYPES = frozenset(
    {
        "reading",
        "writing",
        "phone_usage",
        "computer_usage",
        "other_behavior",
        "communication_distraction",
    }
)
_TRANSITION_EVENT_TYPES = frozenset(
    {
        "sit_at_study_position",
        "leave_study_position",
    }
)

_HISTORY_WEIGHT = 0.25
_CURRENT_WINDOW_WEIGHT = 0.75
_SWITCH_THRESHOLD = 1.0
_SHORT_TAIL_SECONDS = 2.0


def _display_description(event_type: str, raw_description: str = "") -> str:
    """生成与最终事件严格一致的 UI 描述；原始描述仍保留在 VLM meta。"""
    if event_type == "sit_at_study_position":
        return "人物走近学习位置并坐下。"
    if event_type == "leave_study_position":
        return "人物从学习位置起身并离开。"
    if event_type == "reading":
        return "人物持续注视书页进行阅读。"
    if event_type == "writing":
        return "人物持笔在纸面持续书写。"
    if event_type == "phone_usage":
        return "人物持续注视或操作手机。"
    if event_type == "computer_usage":
        return "人物持续注视或操作电脑。"
    if event_type == "communication_distraction":
        return "人物与画面中的另一人交谈或回应，注意力偏离原活动。"
    text = str(raw_description or "")
    has_computer = any(word in text for word in ("电脑", "键盘", "鼠标"))
    has_book = any(word in text for word in ("书", "笔袋", "纸", "学习用品"))
    if has_computer and has_book:
        return "人物先整理电脑，随后拿取或整理书本等学习用品。"
    if has_computer:
        return "人物正在收起、移动或整理电脑。"
    if has_book:
        return "人物正在拿取、摆放或整理书本和学习用品。"
    return "人物处于学习准备、整理或动作切换阶段。"


@dataclass
class _PendingBehavior:
    event: Event
    paths: list[str]
    meta: dict[str, Any]

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
        boundary_analyzer: Optional[BoundaryAnalyzer] = None,
        position_analyzer: Optional[PositionAnalyzer] = None,
        window_label_analyzer: Optional[WindowLabelAnalyzer] = None,
        transition_frame_analyzer: Optional[TransitionFrameAnalyzer] = None,
        summary_analyzer: Optional[SummaryAnalyzer] = None,
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
        trace: bool = False,
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
            trace=trace,
        )
        self.trace = trace

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
        self.boundary_analyzer = boundary_analyzer
        self.position_analyzer = position_analyzer
        self.window_label_analyzer = window_label_analyzer
        self.transition_frame_analyzer = transition_frame_analyzer
        self.summary_analyzer = summary_analyzer
        self._vlm_model = None
        self._vlm_processor = None
        self.vlm_main_calls = 0
        self.vlm_boundary_calls = 0
        self.vlm_position_calls = 0
        self.vlm_window_label_calls = 0
        self.vlm_transition_frame_calls = 0
        self.vlm_summary_calls = 0
        self.video_summary = ""

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
        self._analysis_observations: deque[dict[str, Any]] = deque(maxlen=50000)
        self._analysis_observation_lock = threading.Lock()
        self._stable_behavior_by_track: dict[int | None, str] = {}
        self._pending_behavior_by_track: dict[int | None, _PendingBehavior] = {}
        self._boundary_allowed_event_types: list[str] | None = None
        self._window_label_allowed_event_types: list[str] | None = None

        self._is_realtime = is_camera
        self._stop_event = threading.Event()
        self._active_stream: Optional[FrameStream] = None

    def _ensure_event_analyzer(self) -> Optional[EventAnalyzer]:
        if not self.use_vlm:
            return None
        if self.event_analyzer is None:
            model, processor = load_model(model_path=self.qwen_model_path)
            self._vlm_model, self._vlm_processor = model, processor

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

    def _ensure_boundary_analyzer(self) -> Optional[BoundaryAnalyzer]:
        if not self.use_vlm:
            return None
        if self.boundary_analyzer is not None:
            return self.boundary_analyzer
        # 自定义主分析器没有暴露模型时，不擅自再次加载大模型；测试或集成方
        # 可显式传入 boundary_analyzer。正式内置分析器会复用同一模型。
        self._ensure_event_analyzer()
        if self._vlm_model is None or self._vlm_processor is None:
            return None
        model, processor = self._vlm_model, self._vlm_processor

        def run(paths: list[str]) -> dict[str, Any]:
            return analyze_frame_labels(
                model,
                processor,
                paths,
                allowed_event_types=self._boundary_allowed_event_types,
            )

        self.boundary_analyzer = run
        return self.boundary_analyzer

    def _ensure_position_analyzer(self) -> Optional[PositionAnalyzer]:
        if not self.use_vlm:
            return None
        if self.position_analyzer is not None:
            return self.position_analyzer
        self._ensure_event_analyzer()
        if self._vlm_model is None or self._vlm_processor is None:
            return None
        model, processor = self._vlm_model, self._vlm_processor

        def run(paths: list[str], event_type: str) -> dict[str, Any]:
            return analyze_position_transition(
                model, processor, paths, event_type
            )

        self.position_analyzer = run
        return self.position_analyzer

    def _ensure_window_label_analyzer(self) -> Optional[WindowLabelAnalyzer]:
        if not self.use_vlm:
            return None
        if self.window_label_analyzer is not None:
            return self.window_label_analyzer
        self._ensure_event_analyzer()
        if self._vlm_model is None or self._vlm_processor is None:
            return None
        model, processor = self._vlm_model, self._vlm_processor

        def run(paths: list[str]) -> dict[str, Any]:
            return analyze_window_label(
                model,
                processor,
                paths,
                allowed_event_types=self._window_label_allowed_event_types,
            )

        self.window_label_analyzer = run
        return self.window_label_analyzer

    def _ensure_transition_frame_analyzer(self) -> Optional[TransitionFrameAnalyzer]:
        if not self.use_vlm:
            return None
        if self.transition_frame_analyzer is not None:
            return self.transition_frame_analyzer
        self._ensure_event_analyzer()
        if self._vlm_model is None or self._vlm_processor is None:
            return None
        model, processor = self._vlm_model, self._vlm_processor

        def run(paths: list[str], from_event: str, to_event: str) -> dict[str, Any]:
            return analyze_first_transition_frame(
                model, processor, paths, from_event, to_event
            )

        self.transition_frame_analyzer = run
        return self.transition_frame_analyzer

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

    def _record_analysis_observation(self, video_frame: VideoFrame) -> None:
        objects = []
        for detection in self.latest_result.get("detections", []):
            if isinstance(detection, dict):
                class_name = detection.get("class_name", "unknown")
                confidence = detection.get("confidence", 0.0)
            else:
                class_name = getattr(detection, "class_name", "unknown")
                confidence = getattr(detection, "confidence", 0.0)
            objects.append({
                "class_name": str(class_name),
                "confidence": round(float(confidence), 2),
            })
        observation = {
            "frame_id": int(video_frame.frame_id),
            "timestamp": float(video_frame.timestamp),
            "objects": objects,
        }
        with self._analysis_observation_lock:
            self._analysis_observations.append(observation)

    @staticmethod
    def _format_detected_objects(objects: list[dict[str, Any]]) -> str:
        grouped: dict[str, list[float]] = {}
        for item in objects:
            grouped.setdefault(str(item["class_name"]), []).append(
                float(item["confidence"])
            )
        if not grouped:
            return "无"
        parts = []
        for class_name, confidences in grouped.items():
            count = len(confidences)
            count_text = f"×{count}" if count > 1 else ""
            parts.append(f"{class_name}{count_text}({max(confidences):.2f})")
        return ", ".join(parts)

    def _print_window_trace(self, event: Event, paths: list[str]) -> None:
        candidate_reason = {
            "sit_at_study_position": "人物与学习位置关系持续满足入座阈值",
            "leave_study_position": "人物离开证据持续满足离座阈值",
        }.get(event.event_type)
        if candidate_reason is None:
            if event.event_type == "other_behavior" and event.candidate_scores:
                candidate_reason = "具体物体候选并列，等待VLM分类"
            elif event.event_type == "other_behavior":
                candidate_reason = "窗口内没有具体物体候选，等待VLM分类"
            else:
                candidate_reason = "窗口内具体物体候选投票最高项"
        score_text = event.candidate_scores if event.candidate_scores else "无具体候选"
        print(
            f"\n[分析窗口] {event.start_time:.2f}s - {event.end_time:.2f}s | "
            f"Event候选={event.event_type} | "
            f"候选依据={candidate_reason} | "
            f"物体线索出现帧比例={score_text} | "
            f"无具体物体证据帧比例={event.unclassified_ratio:.1%}"
        )
        if not paths:
            print("  [关键帧] 提取失败")
            return

        with self._analysis_observation_lock:
            observations = [
                dict(item)
                for item in self._analysis_observations
                if event.start_time - 0.5
                <= float(item["timestamp"])
                <= event.end_time + 0.5
            ]

        count = len(paths)
        duration = max(0.0, event.end_time - event.start_time)
        for index in range(count):
            keyframe_time = (
                event.start_time
                if count == 1
                else event.start_time + index * duration / (count - 1)
            )
            nearest = (
                min(
                    observations,
                    key=lambda item: abs(float(item["timestamp"]) - keyframe_time),
                )
                if observations
                else None
            )
            if nearest is None:
                print(
                    f"  [关键帧 {index + 1}/{count}] {keyframe_time:.2f}s | "
                    "附近无YOLO分析帧"
                )
                continue
            print(
                f"  [关键帧 {index + 1}/{count}] {keyframe_time:.2f}s | "
                f"最近YOLO源帧#{nearest['frame_id']} "
                f"@{float(nearest['timestamp']):.2f}s | "
                f"识别={self._format_detected_objects(nearest['objects'])}"
            )

    @staticmethod
    def _vlm_decision_reason(meta: dict[str, Any], event_type: str) -> str:
        fallback = meta.get("timeline_fallback")
        fallback_labels = {
            "vlm_description": "结构化字段失效，按VLM客观描述恢复",
            "event_engine_candidate": "结构化字段失效，按Event候选恢复",
            "vlm_description_generic": "按VLM客观描述归为其他行为",
            "unclassified_seated_window": "无具体证据，使用在座窗口兜底",
            "previous_stable_event": "当前窗口无明确分类，延续上一稳定行为",
            "parse_error_boundary_other": "结构化结果失效且存在边界风险，安全归为其他",
            "whole_window_visual_review": "整窗单标签视觉复核",
            "first_transition_frame_review": "首次稳定动作帧定位",
        }
        if fallback in fallback_labels:
            return fallback_labels[fallback]
        segments = meta.get("activity_segments", [])
        if any(
            isinstance(segment, dict)
            and segment.get("event_type") == event_type
            for segment in segments
        ):
            return "VLM events=true 且提供了合法时间分段"
        if meta.get("boundary_labels_valid"):
            return "逐帧单标签边界复核"
        return "VLM确认，且通过事件契约校验"

    @staticmethod
    def _has_strong_action_evidence(
        event_type: str,
        meta: dict[str, Any],
    ) -> bool:
        """判断新事件是否有足以立即推翻历史状态的明确动作证据。"""
        if meta.get("whole_window_label_valid"):
            return True
        segments = meta.get("activity_segments", [])
        segmented_types = {
            str(item.get("event_type", ""))
            for item in segments
            if isinstance(item, dict)
        } if isinstance(segments, list) else set()
        # 一个窗口中模型明确切出了多个动作，优先保留真实的短时切换。
        if len(segmented_types) > 1 and event_type in segmented_types:
            return True
        # 专门的首帧/逐帧边界复核已经给出合法具体动作分段时，它本身就是
        # 强视觉证据。不能再因主 JSON 的描述措辞错误，把视频末尾完整的
        # 电脑使用或写字暂存为“弱切换”，随后在流结束时降级成 other。
        if (
            event_type != "other_behavior"
            and event_type in segmented_types
            and (
                meta.get("first_transition_frame_valid")
                or meta.get("boundary_labels_valid")
            )
        ):
            return True

        description = str(meta.get("objective_description", ""))
        if not description or description == "[VLM未返回有效description]":
            return False

        if event_type == "reading":
            return any(
                word in description
                for word in (
                    "持续看书", "专注看书", "看书", "读书", "阅读",
                    "看资料", "看教材", "注视书页", "浏览文字",
                )
            )
        if event_type == "writing":
            return any(
                word in description
                for word in (
                    "落笔", "写字", "写作", "书写", "做题", "记笔记", "用笔记录"
                )
            )
        if event_type == "phone_usage":
            return "手机" in description and any(
                word in description
                for word in ("使用", "操作", "滑动", "点击", "查看", "看手机", "注视")
            )
        if event_type == "computer_usage":
            return any(word in description for word in ("电脑", "笔记本电脑")) and any(
                word in description
                for word in (
                    "使用", "操作", "键盘", "鼠标", "屏幕", "点击", "敲击",
                    "工作", "用电脑", "用笔记本电脑",
                )
            )
        if event_type == "communication_distraction":
            return any(
                word in description
                for word in (
                    "交谈", "说话", "交流", "对话", "通话", "讨论",
                    "回应", "倾听", "眼神交流", "轮流交流", "轮流互动",
                    "面向另一人", "朝向彼此", "与人交流", "与他人交流",
                )
            )
        if event_type == "other_behavior":
            return (
                description_has_transition_evidence(description)
                or bool(meta.get("blocks_previous_stable"))
                or bool(
                    meta.get("boundary_labels_valid")
                    and not meta.get("boundary_description_conflict")
                )
            )
        return False

    @staticmethod
    def _with_temporal_context(
        meta: dict[str, Any],
        *,
        previous_stable: str | None,
        proposed_event: str,
        decision: str,
        accumulated_current_weight: float,
    ) -> dict[str, Any]:
        result = dict(meta)
        result["temporal_context"] = {
            "previous_stable_event": previous_stable,
            "proposed_event": proposed_event,
            "history_weight": _HISTORY_WEIGHT,
            "current_window_weight": _CURRENT_WINDOW_WEIGHT,
            "accumulated_current_weight": accumulated_current_weight,
            "switch_threshold": _SWITCH_THRESHOLD,
            "decision": decision,
        }
        return result

    def _remember_temporally_corrected(
        self,
        pending: _PendingBehavior,
        stable_event_type: str,
        *,
        reason: str = "单个弱证据窗口被相同的前后稳定事件包围",
        decision: str = "bridge_a_b_a_to_previous",
    ) -> None:
        description = _display_description(
            stable_event_type,
            str(pending.meta.get("objective_description", "")),
        )
        corrected_event = replace(
            pending.event,
            event_type=stable_event_type,
            description=description,
        )
        corrected_events = {event_type: False for event_type in ALL_EVENTS}
        corrected_events[stable_event_type] = True
        corrected_meta = {
            **pending.meta,
            "event_confirmed": True,
            "events": corrected_events,
            "primary_event": stable_event_type,
            "observed_activities": [stable_event_type],
            "final_description": description,
            "description": description,
            "temporal_correction": {
                "from": pending.event.event_type,
                "to": stable_event_type,
                "reason": reason,
            },
        }
        corrected_meta = self._with_temporal_context(
            corrected_meta,
            previous_stable=stable_event_type,
            proposed_event=pending.event.event_type,
            decision=decision,
            accumulated_current_weight=_CURRENT_WINDOW_WEIGHT,
        )
        print(
            f"[时序修正] {pending.event.start_time:.2f}s - "
            f"{pending.event.end_time:.2f}s | "
            f"{pending.event.event_type} -> {stable_event_type} | "
            f"原因={reason}"
        )
        self._remember_event(corrected_event, pending.paths, corrected_meta)

    def _remember_pending_as_other(
        self,
        pending: _PendingBehavior,
        *,
        reason: str,
    ) -> None:
        """把有明确过渡含义或无法确认的短尾段安全归入“其他”。"""
        description = _display_description(
            "other_behavior",
            str(pending.meta.get("objective_description", "")),
        )
        corrected_event = replace(
            pending.event,
            event_type="other_behavior",
            description=description,
        )
        corrected_events = {event_type: False for event_type in ALL_EVENTS}
        corrected_events["other_behavior"] = True
        corrected_meta = {
            **pending.meta,
            "event_confirmed": True,
            "events": corrected_events,
            "primary_event": "other_behavior",
            "observed_activities": ["other_behavior"],
            "activity_segments": [{
                "event_type": "other_behavior",
                "start_frame": 1,
                "end_frame": max(1, len(pending.paths)),
            }],
            "final_description": description,
            "description": description,
            "temporal_correction": {
                "from": pending.event.event_type,
                "to": "other_behavior",
                "reason": reason,
            },
        }
        corrected_meta = self._with_temporal_context(
            corrected_meta,
            previous_stable=self._stable_behavior_by_track.get(0),
            proposed_event=pending.event.event_type,
            decision="resolve_transition_or_short_tail_as_other",
            accumulated_current_weight=_CURRENT_WINDOW_WEIGHT,
        )
        print(
            f"[时序修正] {pending.event.start_time:.2f}s - "
            f"{pending.event.end_time:.2f}s | "
            f"{pending.event.event_type} -> other_behavior | 原因={reason}"
        )
        self._remember_event(corrected_event, pending.paths, corrected_meta)

    def _stage_or_remember_event(
        self,
        event: Event,
        paths: list[str],
        meta: dict[str, Any],
    ) -> None:
        event_type = event.event_type
        # 当前产品只维护一个主要学习者。Tracker 的临时 ID 可能因遮挡变化，
        # 时序状态必须绑定逻辑学习者，而不是易抖动的检测 ID。
        track_key = 0
        if event_type not in _BEHAVIOR_EVENT_TYPES:
            if event_type == "leave_study_position":
                self._flush_temporal_pending(track_key)
                self._stable_behavior_by_track.pop(track_key, None)
            elif event_type == "sit_at_study_position":
                self._flush_temporal_pending(track_key)
                self._stable_behavior_by_track.pop(track_key, None)
            self._remember_event(event, paths, meta)
            return

        stable = self._stable_behavior_by_track.get(track_key)
        pending = self._pending_behavior_by_track.get(track_key)
        duration = max(0.0, event.end_time - event.start_time)
        # finalize 产生的不足 2 秒尾窗必须先判尾部，不能因为标签恰好与
        # stable 相同就直接保存为完整动作。
        if event.is_final_window and duration < _SHORT_TAIL_SECONDS:
            tail_has_transition = description_has_transition_evidence(
                str(meta.get("objective_description", ""))
            )
            safe_stable_tail = (
                stable in _BEHAVIOR_EVENT_TYPES
                and stable != "other_behavior"
                and not tail_has_transition
                and not meta.get("blocks_previous_stable")
            )
            if safe_stable_tail:
                if pending is not None:
                    self._pending_behavior_by_track.pop(track_key, None)
                    if pending.meta.get("blocks_previous_stable"):
                        self._remember_pending_as_other(
                            pending,
                            reason="最终短尾前的明确边界不得被稳定事件覆盖",
                        )
                    else:
                        self._remember_temporally_corrected(pending, str(stable))
                display_description = _display_description(
                    str(stable), str(meta.get("objective_description", ""))
                )
                tail_event = replace(
                    event,
                    event_type=str(stable),
                    description=display_description,
                )
                stable_events = {name: False for name in ALL_EVENTS}
                stable_events[str(stable)] = True
                tail_meta = {
                    **meta,
                    "events": stable_events,
                    "observed_activities": [str(stable)],
                    "primary_event": str(stable),
                    "display_description": display_description,
                    "short_tail_continuity": True,
                }
                tail_meta = self._with_temporal_context(
                    tail_meta,
                    previous_stable=stable,
                    proposed_event=str(stable),
                    decision="continue_stable_without_tail_transition",
                    accumulated_current_weight=_CURRENT_WINDOW_WEIGHT + _HISTORY_WEIGHT,
                )
                if event_type != stable:
                    print(
                        f"[时序修正] {event.start_time:.2f}s - {event.end_time:.2f}s | "
                        f"{event_type} -> {stable} | 原因=最终短尾没有动作变化证据"
                    )
                self._remember_event(tail_event, paths, tail_meta)
                return
            if pending is not None:
                self._pending_behavior_by_track.pop(track_key, None)
                self._remember_pending_as_other(
                    pending,
                    reason="最终短尾到来前的弱切换仍未确认",
                )
            self._remember_pending_as_other(
                _PendingBehavior(event=event, paths=list(paths), meta=meta),
                reason="最终窗口不足2秒，先按不完整收尾处理",
            )
            return
        if stable is None:
            self._stable_behavior_by_track[track_key] = event_type
            meta = self._with_temporal_context(
                meta,
                previous_stable=None,
                proposed_event=event_type,
                decision="initialize_stable_event",
                accumulated_current_weight=_CURRENT_WINDOW_WEIGHT,
            )
            self._remember_event(event, paths, meta)
            return

        if event_type == stable:
            if pending is not None:
                if pending.meta.get("blocks_previous_stable"):
                    self._remember_pending_as_other(
                        pending,
                        reason="前窗边界复核失败或明确发生切换，禁止被旧稳定事件覆盖",
                    )
                else:
                    self._remember_temporally_corrected(pending, stable)
                self._pending_behavior_by_track.pop(track_key, None)
            meta = self._with_temporal_context(
                meta,
                previous_stable=stable,
                proposed_event=event_type,
                decision="continue_stable_event",
                accumulated_current_weight=_CURRENT_WINDOW_WEIGHT + _HISTORY_WEIGHT,
            )
            self._remember_event(event, paths, meta)
            return

        strong_evidence = self._has_strong_action_evidence(event_type, meta)
        # 一两秒的尾片段不足以凭一个新标签立即推翻稳定状态。明确的整理
        # 动作例外，因为它正是需要及时切断上一事件的过渡段。
        if duration <= _SHORT_TAIL_SECONDS and event_type != "other_behavior":
            strong_evidence = False
        if strong_evidence:
            if pending is not None:
                self._pending_behavior_by_track.pop(track_key, None)
                pending_description = str(
                    pending.meta.get("objective_description", "")
                )
                if pending.meta.get("blocks_previous_stable"):
                    self._remember_pending_as_other(
                        pending,
                        reason="前窗已经阻断旧稳定事件，后续具体动作不得倒灌覆盖",
                    )
                elif (
                    pending.event.event_type == event_type
                    and description_has_transition_evidence(pending_description)
                    and stable != event_type
                ):
                    # 非阅读→翻到目标页→阅读、电脑→收电脑→写字：
                    # 当前强动作只确认“现在开始”，不能把前一准备窗倒灌成它。
                    self._remember_pending_as_other(
                        pending,
                        reason="前窗是明确准备/整理动作，后窗才出现具体行为",
                    )
                elif pending.event.event_type == event_type:
                    self._remember_event(pending.event, pending.paths, pending.meta)
                elif stable is not None:
                    self._remember_temporally_corrected(
                        pending,
                        stable,
                        reason="未确认的弱切换被随后出现的强动作终止",
                        decision="discard_weak_pending_before_strong_switch",
                    )
                else:
                    self._remember_pending_as_other(
                        pending,
                        reason="前窗弱事件缺少可用上下文",
                    )
            meta = self._with_temporal_context(
                meta,
                previous_stable=stable,
                proposed_event=event_type,
                decision="switch_on_strong_action_evidence",
                accumulated_current_weight=_SWITCH_THRESHOLD,
            )
            self._stable_behavior_by_track[track_key] = event_type
            self._remember_event(event, paths, meta)
            return


        if pending is not None and pending.event.event_type == event_type:
            # 两次相同的模型标签不能代替动作证据。先把较早的弱窗口按
            # 历史连续性处理，只继续等待当前窗口；真正的阅读/设备使用
            # 一旦出现明确动作词，会走上面的 strong_evidence 分支。
            self._pending_behavior_by_track.pop(track_key, None)
            if stable is not None:
                self._remember_temporally_corrected(
                    pending,
                    stable,
                    reason="连续两个新标签仍缺少明确动作证据",
                    decision="repeated_weak_labels_do_not_force_switch",
                )
            else:
                self._remember_pending_as_other(
                    pending,
                    reason="连续弱标签缺少可确认的具体动作",
                )
            pending_meta = self._with_temporal_context(
                meta,
                previous_stable=stable,
                proposed_event=event_type,
                decision="keep_waiting_for_action_evidence",
                accumulated_current_weight=_CURRENT_WINDOW_WEIGHT * 2,
            )
            self._pending_behavior_by_track[track_key] = _PendingBehavior(
                event=event,
                paths=list(paths),
                meta=pending_meta,
            )
            print(
                f"[时序等待] {event.start_time:.2f}s - {event.end_time:.2f}s | "
                f"新事件={event_type}连续出现，但仍缺少明确动作证据"
            )
            return

        if pending is not None:
            self._pending_behavior_by_track.pop(track_key, None)
            if stable is not None:
                self._remember_temporally_corrected(
                    pending,
                    stable,
                    reason="弱切换尚未确认又出现了不同事件",
                    decision="replace_unconfirmed_pending",
                )
            else:
                self._remember_pending_as_other(
                    pending,
                    reason="弱切换没有稳定历史且未获确认",
                )
        pending_meta = self._with_temporal_context(
            meta,
            previous_stable=stable,
            proposed_event=event_type,
            decision="hold_weak_switch_for_next_window",
            accumulated_current_weight=_CURRENT_WINDOW_WEIGHT,
        )
        self._pending_behavior_by_track[track_key] = _PendingBehavior(
            event=event,
            paths=list(paths),
            meta=pending_meta,
        )
        print(
            f"[时序等待] {event.start_time:.2f}s - {event.end_time:.2f}s | "
            f"稳定事件={stable} | 新事件={event_type} | "
            "弱切换需下一窗口确认"
        )

    def _flush_temporal_pending(
        self,
        track_key: int | None = None,
        *,
        end_of_stream: bool = False,
    ) -> None:
        keys = (
            [track_key]
            if track_key in self._pending_behavior_by_track
            else list(self._pending_behavior_by_track)
            if track_key is None
            else []
        )
        for key in keys:
            pending = self._pending_behavior_by_track.pop(key)
            if end_of_stream:
                duration = max(
                    0.0,
                    pending.event.end_time - pending.event.start_time,
                )
                description = str(
                    pending.meta.get("objective_description", "")
                )
                stable = self._stable_behavior_by_track.get(key)
                if (
                    pending.event.event_type == "other_behavior"
                    or duration <= _SHORT_TAIL_SECONDS
                    or description_has_transition_evidence(description)
                    or pending.meta.get("blocks_previous_stable")
                ):
                    self._remember_pending_as_other(
                        pending,
                        reason=(
                            "视频结束时短窗口没有后续帧确认"
                            if duration <= _SHORT_TAIL_SECONDS
                            else "视频结束前是明确整理/过渡动作"
                        ),
                    )
                elif stable is not None:
                    self._remember_temporally_corrected(
                        pending,
                        stable,
                        reason="视频结束时弱切换没有后续窗口确认",
                        decision="keep_previous_stable_at_unresolved_tail",
                    )
                else:
                    self._remember_pending_as_other(
                        pending,
                        reason="视频结束时弱事件缺少历史和后续证据",
                    )
                continue
            pending.meta = self._with_temporal_context(
                pending.meta,
                previous_stable=self._stable_behavior_by_track.get(key),
                proposed_event=pending.event.event_type,
                decision="flush_unresolved_tail_without_lookahead",
                accumulated_current_weight=_CURRENT_WINDOW_WEIGHT,
            )
            self._stable_behavior_by_track[key] = pending.event.event_type
            self._remember_event(pending.event, pending.paths, pending.meta)

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

    def _should_refine_window(
        self,
        event: Event,
        paths: list[str],
        meta: dict[str, Any] | None = None,
    ) -> bool:
        """主三步判断完成后，判断是否需要一次逐帧边界复核。"""
        if meta is None:
            # 禁止在第一次完整 VLM 判断之前因 YOLO 稀疏而机械细分。
            return False
        duration = max(0.0, event.end_time - event.start_time)
        if duration <= 0.0 or len(paths) < 2:
            return False
        if (
            event.event_type in _TRANSITION_EVENT_TYPES
            and meta.get("position_review_valid")
            and meta.get("position_review", {}).get("confirmed")
        ):
            return False
        if event.is_final_window and duration < _SHORT_TAIL_SECONDS:
            # 短尾由时序层整体判断，禁止边界模型把一秒尾巴切成多个伪事件。
            return False
        if event.is_final_window or event.event_type in _TRANSITION_EVENT_TYPES:
            return True
        # 视频开头直到 12 秒都允许逐帧寻找真正的入座和准备边界。
        if event.start_time < 12.0:
            return True

        segments = meta.get("activity_segments", [])
        if segments:
            return False
        description = str(meta.get("objective_description", ""))
        if description_has_transition_evidence(description) or any(
            word in description
            for word in ("先", "随后", "然后", "接着", "转而", "之后")
        ):
            return True

        events = meta.get("events", {})
        true_types = [
            name for name, value in events.items()
            if name in ALL_EVENTS and bool(value)
        ] if isinstance(events, dict) else []
        if len(true_types) > 1 and not segments:
            return True

        # 外层 JSON 已损坏时，即使抢救到了完整 event_labels，也只把它当成
        # 二次视觉复核的候选，不能因 description 恰好同词就跳过复核。
        if meta.get("parse_error"):
            described = infer_activity_from_description(description)
            if described in _BEHAVIOR_EVENT_TYPES or self._object_clues_changed(event):
                return True

        # 一个合法 true 且客观动作与之吻合时，主判断已经足够。历史 stable
        # 不同不能单独触发边界复核，否则一次错误 stable 会污染后续整段视频。
        if len(true_types) == 1:
            candidate_conflict = self._candidate_conflicts_with_stable(
                event, true_types[0]
            )
            if candidate_conflict:
                return True
            if infer_activity_from_description(description) == true_types[0]:
                return False

        stable = self._stable_behavior_by_track.get(0)
        if len(true_types) == 1 and stable is not None and true_types[0] != stable:
            return True
        return False

    @staticmethod
    def _candidate_conflicts_with_stable(event: Event, stable: str | None) -> bool:
        """足量的另一物体候选只用于提示切换风险，不直接建立事件。"""
        if stable not in _BEHAVIOR_EVENT_TYPES or stable == "other_behavior":
            return False
        competing = {
            name: float(score)
            for name, score in event.candidate_scores.items()
            if name in _BEHAVIOR_EVENT_TYPES and name != stable
        }
        return bool(competing) and max(competing.values()) >= 0.25

    def _object_clues_changed(self, event: Event) -> bool:
        """用窗口首尾 YOLO 类别变化判断解析失败时是否疑似切换。"""
        with self._analysis_observation_lock:
            observations = [
                item for item in self._analysis_observations
                if event.start_time <= float(item["timestamp"]) <= event.end_time
            ]
        if len(observations) < 2:
            # 多种候选或一半以上未分类，也属于明显不稳定的弱线索。
            return len(event.candidate_scores) > 1 or (
                bool(event.candidate_scores) and event.unclassified_ratio >= 0.5
            )
        relevant = {"book", "pen", "laptop", "computer", "keyboard", "mouse", "cell phone", "phone"}
        def classes(item: dict[str, Any]) -> set[str]:
            return {
                str(obj.get("class_name", ""))
                for obj in item.get("objects", [])
                if str(obj.get("class_name", "")) in relevant
            }
        return classes(observations[0]) != classes(observations[-1])

    def _first_transition_target(
        self,
        event: Event,
        meta: dict[str, Any],
        stable: str | None,
        described: str | None,
    ) -> str | None:
        """选择适合用单个首帧号定位的具体目标动作。"""
        concrete = _BEHAVIOR_EVENT_TYPES - {"other_behavior"}
        events = meta.get("events", {})
        true_types = [
            name for name, enabled in events.items()
            if name in concrete and bool(enabled)
        ] if isinstance(events, dict) else []
        if len(true_types) == 1:
            main_type = true_types[0]
            # 主VLM与描述给出同一明确动作时，YOLO 的书/电脑物体候选
            # 不能凭自身触发相反切换；若动作与 stable 相同则无需找边界。
            if main_type == stable:
                return None
            if described == main_type or event.start_time < 12.0:
                return main_type
        if event.start_time < 12.0 and described in concrete and described != stable:
            return str(described)

        if stable in concrete:
            competing = [
                (name, float(score))
                for name, score in event.candidate_scores.items()
                if name in concrete and name != stable and float(score) >= 0.25
            ]
            if competing:
                return max(competing, key=lambda item: item[1])[0]
        return None

    def _review_boundary_edges(
        self,
        event: Event,
        paths: list[str],
        *,
        allowed_event_types: list[str] | None = None,
        edge_names: tuple[str, ...] = ("start", "end"),
    ) -> dict[str, str]:
        """分别复核窗口首尾稳定动作，防止多数帧掩盖边缘切换。"""
        if len(paths) < 5 or not (event.start_time < 12.0 or event.is_final_window):
            return {}
        analyzer = self._ensure_window_label_analyzer()
        if analyzer is None:
            return {}
        self._window_label_allowed_event_types = list(
            allowed_event_types or _BEHAVIOR_EVENT_TYPES
        )
        result: dict[str, str] = {}
        edge_groups = {
            "start": paths[:3],
            "end": paths[-2:],
        }
        for edge_name, edge_paths in edge_groups.items():
            if edge_name not in edge_names:
                continue
            self.vlm_window_label_calls += 1
            try:
                review = analyzer(edge_paths)
            except Exception as exc:
                print(f"[边缘复核] {edge_name}=失败 | 原因={exc}")
                continue
            label = str(review.get("label", ""))
            valid = bool(review.get("valid")) and label in _BEHAVIOR_EVENT_TYPES
            # 限定为“目标/其他”的边缘复核若输出了第三个合法枚举，说明
            # 目标动作尚未开始；相对于当前目标应安全记为 other。
            if not valid and allowed_event_types and "other_behavior" in allowed_event_types:
                raw_label = str(review.get("_vlm_raw", "")).strip().strip("`\"'")
                if (
                    raw_label in _BEHAVIOR_EVENT_TYPES
                    and raw_label not in allowed_event_types
                ):
                    label = "other_behavior"
                    valid = True
            print(
                f"[边缘复核] {event.start_time:.2f}s - {event.end_time:.2f}s | "
                f"{edge_name}={label if valid else '无效'} | "
                f"原始={str(review.get('_vlm_raw', ''))[:100].replace(chr(10), ' ')}"
            )
            if valid:
                result[edge_name] = label
        return result

    @staticmethod
    def _frame_motion_scores(paths: list[str]) -> list[float]:
        """计算相邻关键帧的平均灰度变化；无法读取时返回空列表。"""
        if len(paths) < 2:
            return []
        gray_frames = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                return []
            gray_frames.append(
                cv2.cvtColor(
                    cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2GRAY,
                )
            )
        return [
            float(cv2.absdiff(before, after).mean())
            for before, after in zip(gray_frames, gray_frames[1:])
        ]

    @staticmethod
    def _stable_motion_span(paths: list[str]) -> tuple[int, int, list[float]] | None:
        """用连续关键帧变化定位稳定动作区间，帧号为 1-based。"""
        if len(paths) < 5:
            return None
        scores = EndToEndRunner._frame_motion_scores(paths)
        if not scores:
            return None
        low = min(scores)
        high = max(scores)
        # 画面变化近似均匀时没有可靠边界，禁止机械切分稳定窗口。
        # 最低变化量本身仍很高，说明整段都在走动、拿取或整理；此时所谓
        # “最低的一段”只是相对较慢，并不代表已经进入稳定阅读/书写。
        if low > 8.0 or high - low < 3.0 or high < max(4.0, low * 1.8):
            return None
        # 使用最低变化量的 2 倍作为“稳定”上限。中位数在一个窗口后半段
        # 持续整理时会被抬得过高，把最先发生变化的关键帧也吞进稳定动作；
        # 以局部最低噪声为基准，能让 3s 阅读→7s 整理这类边界落在真正
        # 开始变化的位置，同时仍容忍轻微手部动作与视频压缩噪声。
        threshold = max(0.75, low * 2.0)
        best_start = -1
        best_end = -1
        run_start = -1
        for index, score in enumerate(scores + [float("inf")]):
            if score <= threshold:
                if run_start < 0:
                    run_start = index
                continue
            if run_start >= 0:
                run_end = index - 1
                if run_end - run_start > best_end - best_start:
                    best_start, best_end = run_start, run_end
                run_start = -1
        if best_start < 0 or best_end - best_start + 1 < 2:
            return None
        # 差分 i 连接第 i+1 与第 i+2 帧，因此连续低变化边对应的
        # 稳定图像区间为 [best_start+1, best_end+2]（1-based）。
        return best_start + 1, best_end + 2, scores

    def _review_first_transition_frame(
        self,
        paths: list[str],
        meta: dict[str, Any],
        from_event: str,
        to_event: str,
        edge_labels: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        if from_event == to_event:
            print(f"[首帧定位] 跳过无效同类切换：{from_event} -> {to_event}")
            return None
        analyzer = self._ensure_transition_frame_analyzer()
        if analyzer is None:
            return None
        print(
            f"[首帧定位] {from_event} -> {to_event} | "
            f"关键帧={len(paths)}张"
        )
        try:
            review = analyzer(paths, from_event, to_event)
        except Exception as exc:
            self.vlm_transition_frame_calls += 1
            print(f"[首帧定位] 失败 | 原因={exc}")
            return None
        self.vlm_transition_frame_calls += max(
            1, int(review.get("model_calls", 1))
        )
        if not review.get("valid") or not review.get("activity_segments"):
            print(
                f"[首帧定位] 无效 | "
                f"原始={str(review.get('_vlm_raw', ''))[:120].replace(chr(10), ' ')}"
            )
            return None
        labels = list(review.get("frame_labels", []))
        edges = dict(edge_labels or {})
        if len(labels) >= 2:
            start_label = edges.get("start")
            end_label = edges.get("end")
            if start_label in {to_event, "other_behavior"}:
                start_span = min(3, len(labels))
                labels[:start_span] = [str(start_label)] * start_span
            if end_label in {to_event, "other_behavior"}:
                labels[-2:] = [str(end_label)] * 2
        target_ratio = (
            labels.count(to_event) / len(labels) if labels else 0.0
        )
        motion_span = self._stable_motion_span(paths)
        if motion_span is not None and target_ratio >= 0.65:
            motion_start, motion_end, motion_scores = motion_span
            labels = ["other_behavior"] * len(labels)
            labels[motion_start - 1:motion_end] = [to_event] * (
                motion_end - motion_start + 1
            )
            print(
                f"[运动边界] 稳定{to_event}=第{motion_start}-{motion_end}帧 | "
                f"变化分数={[round(score, 2) for score in motion_scores]}"
            )
        segments = frame_labels_to_segments(labels, len(labels))
        if not segments:
            return None
        events = {name: False for name in ALL_EVENTS}
        for label in labels:
            if label in events:
                events[label] = True
        print(
            f"[首帧定位] 首个{to_event}=第{review.get('first_frame')}帧 | "
            f"原始={str(review.get('_vlm_raw', ''))[:120].replace(chr(10), ' ')} | "
            f"二值复核={str(review.get('presence_vlm_raw', ''))[:120].replace(chr(10), ' ')}"
        )
        return {
            **meta,
            "event_confirmed": True,
            "events": events,
            "observed_activities": list(dict.fromkeys(labels)),
            "activity_segments": segments,
            "frame_labels": labels,
            "boundary_labels_valid": True,
            "first_transition_frame_valid": True,
            "first_transition_frame": int(review["first_frame"]),
            "transition_frame_vlm_raw": review.get("_vlm_raw", ""),
            "transition_presence_vlm_raw": review.get("presence_vlm_raw", ""),
            "boundary_edge_labels": edges,
            "timeline_fallback": "first_transition_frame_review",
            "blocks_previous_stable": True,
        }

    def _review_stable_motion_core(
        self,
        event: Event,
        paths: list[str],
        motion_span: tuple[int, int, list[float]] | None,
    ) -> str | None:
        """只给 VLM 看低运动稳定区，独立判断该区真正执行的动作。"""
        if motion_span is None:
            return None
        start_frame, end_frame, _ = motion_span
        core_paths = paths[start_frame - 1:end_frame]
        if len(core_paths) < 2:
            return None
        analyzer = self._ensure_window_label_analyzer()
        if analyzer is None:
            return None
        self._window_label_allowed_event_types = list(_BEHAVIOR_EVENT_TYPES)
        self.vlm_window_label_calls += 1
        try:
            review = analyzer(core_paths)
        except Exception as exc:
            print(f"[稳定区复核] 失败 | 原因={exc}")
            return None
        label = str(review.get("label", ""))
        valid = bool(review.get("valid")) and label in _BEHAVIOR_EVENT_TYPES
        print(
            f"[稳定区复核] {event.start_time:.2f}s - {event.end_time:.2f}s | "
            f"帧={start_frame}-{end_frame} | "
            f"动作={label if valid else '无效'} | "
            f"原始={str(review.get('_vlm_raw', ''))[:100].replace(chr(10), ' ')}"
        )
        return label if valid else None

    def _observed_communication_labels(
        self,
        event: Event,
        meta: dict[str, Any],
        frame_count: int,
    ) -> list[str] | None:
        """主 VLM 确认交流时，用双人检测确定交流覆盖的关键帧。"""
        events = meta.get("events", {})
        description = str(
            meta.get("objective_description", meta.get("description", ""))
        )
        if (
            frame_count <= 0
            or meta.get("parse_error")
            or not isinstance(events, dict)
            or not events.get("communication_distraction")
            or infer_activity_from_description(description)
            != "communication_distraction"
        ):
            return None
        with self._analysis_observation_lock:
            observations = [
                item for item in self._analysis_observations
                if event.start_time - 0.25
                <= float(item["timestamp"])
                <= event.end_time + 0.25
            ]
        if not observations:
            return None
        duration = max(0.0, event.end_time - event.start_time)
        labels: list[str] = []
        for index in range(frame_count):
            ratio = index / max(1, frame_count - 1)
            timestamp = event.start_time + ratio * duration
            nearest = min(
                observations,
                key=lambda item: abs(float(item["timestamp"]) - timestamp),
            )
            person_count = sum(
                1 for item in nearest.get("objects", [])
                if str(item.get("class_name", "")) == "person"
            )
            labels.append(
                "communication_distraction"
                if person_count >= 2
                else "other_behavior"
            )
        if "communication_distraction" not in labels:
            return None
        print(
            "[交流复核] 主VLM确认交流且检测到双人 | "
            f"逐帧={labels}"
        )
        return labels

    @staticmethod
    def _apply_reviewed_labels(
        meta: dict[str, Any],
        labels: list[str],
        *,
        timeline_fallback: str,
    ) -> dict[str, Any]:
        segments = frame_labels_to_segments(labels, len(labels))
        events = {name: False for name in ALL_EVENTS}
        for label in labels:
            if label in events:
                events[label] = True
        return {
            **meta,
            "event_confirmed": bool(segments),
            "events": events,
            "observed_activities": list(dict.fromkeys(labels)),
            "activity_segments": segments,
            "frame_labels": labels,
            "boundary_labels_valid": bool(segments),
            "timeline_fallback": timeline_fallback,
            "blocks_previous_stable": True,
        }

    def _review_boundary_window(
        self,
        event: Event,
        paths: list[str],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        print(
            f"[边界复核] {event.start_time:.2f}s - {event.end_time:.2f}s | "
            "使用同一组关键帧逐帧单标签定位切换"
        )
        result = {**meta, "boundary_review_required": True}

        def boundary_failure_as_other(reason: str) -> dict[str, Any]:
            safe_events = {name: False for name in ALL_EVENTS}
            safe_events["other_behavior"] = True
            result.update({
                "boundary_review_error": reason,
                "event_confirmed": True,
                "events": safe_events,
                "observed_activities": ["other_behavior"],
                "activity_segments": [{
                    "event_type": "other_behavior",
                    "start_frame": 1,
                    "end_frame": len(paths),
                }],
                "blocks_previous_stable": True,
                "timeline_fallback": "parse_error_boundary_other",
            })
            return result

        stable = self._stable_behavior_by_track.get(0)
        described = infer_activity_from_description(
            str(meta.get("objective_description", meta.get("description", "")))
        )
        description = str(
            meta.get("objective_description", meta.get("description", ""))
        )
        motion_span = self._stable_motion_span(paths)
        stable_core_label = self._review_stable_motion_core(
            event, paths, motion_span
        )
        main_events = meta.get("events", {})
        main_concrete = [
            name for name in (_BEHAVIOR_EVENT_TYPES - {"other_behavior"})
            if isinstance(main_events, dict) and main_events.get(name)
        ]
        main_label = main_concrete[0] if len(main_concrete) == 1 else None
        communication_labels = self._observed_communication_labels(
            event, meta, len(paths)
        )
        # 整窗或稳定核心本身就是交流时，主VLM语义 + 双人检测已经构成
        # 完整证据，不再允许容易被书本干扰的首尾单标签复核覆盖它。
        if communication_labels is not None and stable_core_label in (
            None, "communication_distraction"
        ):
            return self._apply_reviewed_labels(
                meta,
                communication_labels,
                timeline_fallback="structured_communication_with_two_people",
            )
        if event.start_time >= 12.0 and main_label and stable_core_label:
            if main_label == stable_core_label:
                return self._apply_reviewed_labels(
                    meta,
                    [main_label] * len(paths),
                    timeline_fallback="main_and_stable_core_agree",
                )
            # 两次独立视觉判断给出不同具体动作时，不能任选一个铺满窗口；
            # 这通常是拿笔、翻书、整理等过渡画面。
            return self._apply_reviewed_labels(
                meta,
                ["other_behavior"] * len(paths),
                timeline_fallback="main_and_stable_core_conflict",
            )
        follows_sitting = any(
            previous.event_type == "sit_at_study_position"
            and abs(previous.end_time - event.start_time) <= 0.35
            for previous in self.events[-3:]
        )
        motion_scores = self._frame_motion_scores(paths)
        persistently_high_motion = bool(motion_scores) and min(motion_scores) > 8.0
        if (
            event.start_time < 12.0
            and follows_sitting
            and motion_span is None
            and persistently_high_motion
        ):
            return self._apply_reviewed_labels(
                meta,
                ["other_behavior"] * len(paths),
                timeline_fallback="post_sitting_high_motion_preparation",
            )
        # 先只看末尾两帧，用来发现主描述遗漏的新稳定动作；目标确定后，
        # 再以“目标/其他”二选一检查开头，避免额外做一次无用的全类别首帧判断。
        edge_labels = self._review_boundary_edges(
            event, paths, edge_names=("end",)
        )
        transition_target = self._first_transition_target(
            event, meta, stable, described
        )
        if stable_core_label in (_BEHAVIOR_EVENT_TYPES - {"other_behavior"}):
            transition_target = stable_core_label
        end_label = edge_labels.get("end")
        if (
            stable_core_label not in (_BEHAVIOR_EVENT_TYPES - {"other_behavior"})
            and
            end_label in (_BEHAVIOR_EVENT_TYPES - {"other_behavior"})
            and end_label != stable
        ):
            transition_target = end_label
        if transition_target is not None:
            compatible_edges = {
                name: label
                for name, label in edge_labels.items()
                if label in {transition_target, "other_behavior"}
            }
            # 开头只允许“目标动作/其他”，静态摆在桌上的电脑或书本不能
            # 把准备动作误导成第三种正式事件。
            compatible_edges.update(
                self._review_boundary_edges(
                    event,
                    paths,
                    allowed_event_types=[
                        transition_target, "other_behavior"
                    ],
                    edge_names=("start",),
                )
            )
            located = self._review_first_transition_frame(
                paths,
                meta,
                str(stable) if stable in _BEHAVIOR_EVENT_TYPES else "other_behavior",
                transition_target,
                compatible_edges,
            )
            if located is not None:
                # 混合窗口中，运动核心可能是随后开始的阅读/电脑动作；核心
                # 之前若主VLM确认交流且YOLO确有两人，应保留为交流，而不是
                # 被二元边界定位统一写成 other。核心之后仍按实际边界结果。
                if communication_labels is not None:
                    combined_labels = [
                        "communication_distraction"
                        if (
                            current == "other_behavior"
                            and communication == "communication_distraction"
                        )
                        else current
                        for current, communication in zip(
                            located.get("frame_labels", []),
                            communication_labels,
                        )
                    ]
                    located = self._apply_reviewed_labels(
                        located,
                        combined_labels,
                        timeline_fallback="communication_then_stable_action_review",
                    )
                return located
        # 主 JSON 失败、但描述没有明确的先后切换时，不要求 2B 模型一次生成
        # 九个标签。先用同一组图片做一次更可靠的整窗单标签视觉复核。
        # description 只决定走哪种复核，不直接产生正式事件。
        if (
            meta.get("parse_error")
            and event.start_time >= 12.0
            and not description_has_transition_evidence(description)
            and not any(word in description for word in ("先", "随后", "然后", "接着", "转而", "之后"))
        ):
            return self._review_whole_window(event, paths, result)
        # JSON失败时 description 只能缩小二次视觉复核的候选范围，不能
        # 直接创建事件。这样既能从 sticky-other 恢复，又不违反解析契约。
        allowed = set(_BEHAVIOR_EVENT_TYPES)
        if event.start_time < 12.0:
            allowed.add("sit_at_study_position")
        if not meta.get("parse_error") and event.event_type in ALL_EVENTS:
            allowed.add(event.event_type)
        if not meta.get("parse_error") and stable in ALL_EVENTS:
            allowed.add(str(stable))
        for event_map_name in ("events", "raw_events"):
            event_map = meta.get(event_map_name, {})
            if isinstance(event_map, dict):
                allowed.update(
                    name for name, enabled in event_map.items()
                    if name in ALL_EVENTS and bool(enabled)
                )
        if described in ALL_EVENTS:
            allowed.add(str(described))
        self._boundary_allowed_event_types = [
            name for name in ALL_EVENTS if name in allowed
        ]
        analyzer = self._ensure_boundary_analyzer()
        if analyzer is None:
            # 自定义/测试分析器可能没有二次视觉能力；此时保留已经得到的
            # 主结构结果。正式内置流程始终会复用已加载模型创建复核器。
            result["boundary_review_error"] = "没有可用的逐帧边界分析器"
            return result
        self.vlm_boundary_calls += 1
        try:
            boundary = analyzer(paths)
        except Exception as exc:
            return boundary_failure_as_other(str(exc))
        labels = list(boundary.get("frame_labels", []))
        boundary_segments = list(boundary.get("activity_segments", []))
        print(
            f"[边界标签] 允许={self._boundary_allowed_event_types} | "
            f"输出={labels}"
        )
        pathological = (
            len(set(labels)) > 3
            or len(boundary_segments) > 3
            or any(label not in allowed for label in labels)
        )
        if pathological:
            boundary = {
                **boundary,
                "valid": False,
                "parse_error": "边界标签种类/切换次数异常，疑似机械枚举",
            }
        if not boundary.get("valid") or not boundary.get("activity_segments"):
            result["boundary_review_error"] = str(
                boundary.get("parse_error", "frame_labels 无效")
            )
            main_events = meta.get("events", {})
            main_true = [
                name for name, enabled in main_events.items()
                if name in ALL_EVENTS and bool(enabled)
            ] if isinstance(main_events, dict) else []
            main_type = main_true[0] if len(main_true) == 1 else None
            main_matches_description = (
                main_type is not None
                and infer_activity_from_description(description) == main_type
            )
            main_has_no_conflict = (
                main_type is not None
                and not self._candidate_conflicts_with_stable(event, main_type)
                and not description_has_transition_evidence(description)
            )
            if (
                not meta.get("parse_error")
                and main_matches_description
                and main_has_no_conflict
            ):
                # 只有主判断、描述和候选证据三者没有冲突时，才允许在边界
                # 模型格式失败后保留主判断。
                result["boundary_review_rejected"] = True
                return result
            return boundary_failure_as_other(result["boundary_review_error"])

        labels = list(boundary["frame_labels"])
        segments = list(boundary["activity_segments"])
        boundary_events = {name: False for name in ALL_EVENTS}
        for label in labels:
            boundary_events[str(label)] = True
        result.update({
            "event_confirmed": True,
            "events": boundary_events,
            "observed_activities": list(dict.fromkeys(labels)),
            "activity_segments": segments,
            "frame_labels": labels,
            "boundary_labels_valid": True,
            "boundary_vlm_raw": boundary.get("_vlm_raw", ""),
            "boundary_description_conflict": bool(
                described in _BEHAVIOR_EVENT_TYPES
                and labels
                and str(described) not in labels
            ),
        })
        return result

    def _review_whole_window(
        self,
        event: Event,
        paths: list[str],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        """以一个短标签复核整窗，避免 2B 模型机械枚举九帧标签。"""
        result = dict(meta)
        allowed = [
            name for name in ALL_EVENTS
            if name in _BEHAVIOR_EVENT_TYPES
        ]
        self._window_label_allowed_event_types = allowed
        analyzer = self._ensure_window_label_analyzer()
        if analyzer is None:
            review: dict[str, Any] = {
                "valid": False,
                "parse_error": "没有可用的整窗单标签分析器",
            }
        else:
            self.vlm_window_label_calls += 1
            print(
                f"[整窗复核] {event.start_time:.2f}s - {event.end_time:.2f}s | "
                "主JSON失败，使用单标签视觉判断"
            )
            try:
                review = analyzer(paths)
            except Exception as exc:
                review = {"valid": False, "parse_error": str(exc)}

        label = str(review.get("label", ""))
        if not review.get("valid") or label not in _BEHAVIOR_EVENT_TYPES:
            label = "other_behavior"
            result["whole_window_review_error"] = str(
                review.get("parse_error", "整窗单标签无效")
            )
        events = {name: False for name in ALL_EVENTS}
        events[label] = True
        result.update({
            "event_confirmed": True,
            "events": events,
            "observed_activities": [label],
            "activity_segments": [{
                "event_type": label,
                "start_frame": 1,
                "end_frame": len(paths),
            }],
            "whole_window_label_valid": bool(review.get("valid")),
            "whole_window_vlm_raw": review.get("_vlm_raw", ""),
            "timeline_fallback": (
                "whole_window_visual_review"
                if review.get("valid")
                else "parse_error_boundary_other"
            ),
            # 二次结构化复核（包括安全 other）必须结束旧稳定状态，不能再被
            # A-B-A 桥接改回 reading。
            "blocks_previous_stable": True,
        })
        print(
            f"[整窗标签] 允许={allowed} | 输出={label} | "
            f"有效={bool(review.get('valid'))} | "
            f"原始={str(review.get('_vlm_raw', ''))[:160].replace(chr(10), ' ')}"
        )
        return result

    def _review_position_window(
        self,
        event: Event,
        paths: list[str],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        """独立验证动态入座/离座，不依赖普通 description 措辞。"""
        current_events = meta.get("events", {})
        if isinstance(current_events, dict) and current_events.get(event.event_type):
            return meta
        analyzer = self._ensure_position_analyzer()
        if analyzer is None:
            return meta
        print(
            f"[位置复核] {event.start_time:.2f}s - {event.end_time:.2f}s | "
            f"候选={event.event_type}"
        )
        try:
            review = analyzer(paths, event.event_type)
        except Exception as exc:
            self.vlm_position_calls += 1
            return {**meta, "position_review_error": str(exc)}
        self.vlm_position_calls += max(1, int(review.get("model_calls", 1)))
        print(
            f"[位置复核结果] change={review.get('position_change', 'none')} | "
            f"start={review.get('start_state', '') or '未知'} | "
            f"end={review.get('end_state', '') or '未知'} | "
            f"confirmed={bool(review.get('confirmed'))}"
        )
        result = {
            **meta,
            "position_review_valid": bool(review.get("valid")),
            "position_review": review,
        }
        sequence_confirms_early_sit = bool(
            review.get("valid")
            and event.event_type == "sit_at_study_position"
            and event.start_time < 5.0
            and review.get("position_change") == "standing_to_sitting"
            and review.get("start_state") != "sitting"
        )
        position_confirmed = bool(
            review.get("confirmed") or sequence_confirms_early_sit
        )
        if sequence_confirms_early_sit and not review.get("confirmed"):
            print(
                "[位置复核修正] 连续帧确认人物走近并执行入座动作，"
                "采用序列变化证据，不因单张末帧姿态误判而删除入座"
            )
            result["position_review"] = {
                **review,
                "confirmed": True,
                "confirmed_by_sequence": True,
            }
        if not review.get("valid") or not position_confirmed:
            return result
        events = {name: False for name in ALL_EVENTS}
        events[event.event_type] = True
        description = (
            "人物从站立或走近转为在学习位置坐下。"
            if event.event_type == "sit_at_study_position"
            else "人物从学习位置起身并离开。"
        )
        result.update({
            "event_confirmed": True,
            "events": events,
            "primary_event": event.event_type,
            "observed_activities": [event.event_type],
            "activity_segments": [{
                "event_type": event.event_type,
                "start_frame": 1,
                "end_frame": len(paths),
            }],
            "objective_description": description,
            "final_description": description,
            "description": description,
            "parse_error": "",
        })
        return result

    def _handle_event(
        self,
        event: Event,
        paths: Optional[list[str]] = None,
        *,
        allow_refine: bool = True,
    ) -> None:
        if self.trace:
            print(f"[TRACE][EVENT_TO_VLM] {json.dumps(asdict(event), ensure_ascii=False)}")
        if paths is None:
            paths = self._extract_event_frames(event)
        self._print_window_trace(event, paths)
        if self.trace:
            print(f"[TRACE][VLM_KEYFRAMES] {json.dumps(paths, ensure_ascii=False)}")
        if not paths:
            error = f"事件 {event.event_type} 关键帧提取失败"
            self.errors.append(error)
            print(f"[VLM最终] 未分析 | 原因={error}")
            self.rejected_events.append(
                {"event": asdict(event), "vlm": {"error": error}, "keyframes": []}
            )
            return

        meta: dict[str, Any] = {"event_confirmed": True, "vlm_skipped": True}
        confirmed_events: list[Event] = [event]
        analyzer = (
            self._ensure_event_analyzer()
            if event.event_type in _VLM_EVENT_TYPES
            else None
        )

        if analyzer is not None:
            try:
                print(f"[VLM请求] 候选={event.event_type} | 关键帧={len(paths)}张")
                self.vlm_main_calls += 1
                confirmed_event, meta = analyzer(event, paths)
            except Exception as exc:
                error = f"VLM 分析 {event.event_type} 失败: {exc}"
                self.errors.append(error)
                print(f"[VLM最终] 分析失败 | 原因={error}")
                self.rejected_events.append(
                    {
                        "event": asdict(event),
                        "vlm": {"error": error, "event_confirmed": False},
                        "keyframes": paths,
                    }
                )
                return

            if event.event_type in _TRANSITION_EVENT_TYPES:
                meta = self._review_position_window(event, paths, meta)

            if allow_refine and self._should_refine_window(event, paths, meta):
                meta = self._review_boundary_window(event, paths, meta)

            # 位置候选若被动态复核否决，只有经过逐帧边界复核得到的行为
            # 分段可以保留。这样静态“坐着”不会确认入座，同时也不会把
            # 同一开头窗口后半段真正开始的阅读一并丢弃。
            if event.event_type in _TRANSITION_EVENT_TYPES:
                transition = event.event_type
                reviewed_behaviors = bool(meta.get("boundary_labels_valid"))
                filtered_segments = [
                    item for item in meta.get("activity_segments", [])
                    if isinstance(item, dict)
                    and (
                        item.get("event_type") == transition
                        or (
                            reviewed_behaviors
                            and item.get("event_type") in _BEHAVIOR_EVENT_TYPES
                        )
                    )
                ]
                transition_confirmed = bool(
                    isinstance(meta.get("events"), dict)
                    and meta["events"].get(transition)
                )
                retained_types = {
                    str(item.get("event_type")) for item in filtered_segments
                }
                meta = {
                    **meta,
                    "events": {
                        name: bool(
                            (name == transition and transition_confirmed)
                            or name in retained_types
                        )
                        for name in ALL_EVENTS
                    },
                    "activity_segments": filtered_segments,
                    "observed_activities": list(retained_types),
                    "event_confirmed": bool(transition_confirmed or retained_types),
                }

            if self.trace:
                trace_meta = dict(meta)
                raw_text = str(trace_meta.pop("_vlm_raw", ""))
                if raw_text:
                    trace_meta["vlm_raw_summary"] = {
                        "characters": len(raw_text),
                        "preview": raw_text[:160].replace("\n", " "),
                    }
                print(
                    "[TRACE][VLM_NORMALIZED_OUTPUT] "
                    + json.dumps(trace_meta, ensure_ascii=False, default=str)
                )
            activities = infer_activities_from_vlm_meta(meta)
            if (
                not activities
                and event.event_type in _BEHAVIOR_EVENT_TYPES
            ):
                description = str(meta.get("description", "")).strip()
                stable_activity = self._stable_behavior_by_track.get(0)
                boundary_window = bool(meta.get("boundary_review_required"))
                parse_error = bool(meta.get("parse_error"))
                transition = description_has_transition_evidence(description)
                stable_conflict = (
                    stable_activity in _BEHAVIOR_EVENT_TYPES
                    and (
                        description_conflicts_with_event(
                            description, str(stable_activity)
                        )
                        or self._candidate_conflicts_with_stable(
                            event, str(stable_activity)
                        )
                    )
                )
                may_extend_stable = (
                    stable_activity in _BEHAVIOR_EVENT_TYPES
                    and stable_activity != "other_behavior"
                    and not boundary_window
                    and (
                        not event.is_final_window
                        or (event.end_time - event.start_time) < _SHORT_TAIL_SECONDS
                    )
                    and event.start_time >= 12.0
                    and not transition
                    and not stable_conflict
                    and not self._object_clues_changed(event)
                )
                if may_extend_stable:
                    fallback_activity = str(stable_activity)
                    fallback_source = "previous_stable_event"
                else:
                    # JSON 失败、边界/整理/新事件/首尾窗口都采用安全的 other；
                    # 抢救 description 和 EventEngine 物体候选不得制造具体事件。
                    fallback_activity = "other_behavior"
                    fallback_source = (
                        "parse_error_boundary_other"
                        if parse_error or boundary_window
                        else "unclassified_seated_window"
                    )

                if not description or description == "[VLM未返回有效description]":
                    description = "人物仍在学习位置，但未识别出可归入具体类别的动作。"

                fallback_events = {
                    event_type: False
                    for event_type in ALL_EVENTS
                }
                fallback_events[fallback_activity] = True
                meta = {
                    **meta,
                    "event_confirmed": True,
                    "description": description,
                    "events": fallback_events,
                    "primary_event": fallback_activity,
                    "observed_activities": [fallback_activity],
                    "activity_segments": [{
                        "event_type": fallback_activity,
                        "start_frame": 1,
                        "end_frame": len(paths),
                    }],
                    "timeline_fallback": fallback_source,
                    "blocks_previous_stable": bool(
                        fallback_activity == "other_behavior"
                        and stable_activity not in (None, "other_behavior")
                        and (transition or stable_conflict or boundary_window)
                    ),
                    "contract_warnings": [
                        *meta.get("contract_warnings", []),
                        "结构化事件没有有效行为，已按 "
                        f"{fallback_source} 恢复为 {fallback_activity}",
                    ],
                }
                activities = [fallback_activity]
            if activities:
                confirmed_events = self._split_activity_event(
                    event,
                    activities,
                    meta.get("description", event.description),
                    activity_segments=meta.get("activity_segments", []),
                    frame_count=len(paths),
                    primary_activity=meta.get("primary_event"),
                )
                meta = {
                    **meta,
                    "event_confirmed": True,
                    "reclassified_from": event.event_type,
                    "reclassified_to": [item.event_type for item in confirmed_events],
                }
            elif confirmed_event is not None and should_persist_to_memory(meta):
                confirmed_events = [confirmed_event]
            else:
                confirmed_events = []
        elif self.trace:
            print(
                "[TRACE][VLM_SKIPPED] "
                + json.dumps(
                    {
                        "reason": "VLM 已禁用或事件类型不在当前统一事件清单中",
                        "event_type": event.event_type,
                        "meta": meta,
                    },
                    ensure_ascii=False,
                )
            )

        if not confirmed_events:
            warnings = meta.get("contract_warnings", [])
            reason = "；".join(str(item) for item in warnings[-3:]) or "events全部为false"
            print(f"[VLM最终] 未确认事件 | 原因={reason}")
            self.rejected_events.append(
                {"event": asdict(event), "vlm": meta, "keyframes": paths}
            )
            return

        for confirmed_event in confirmed_events:
            raw_description = str(
                meta.get("objective_description", meta.get("description", ""))
            )
            display_description = _display_description(
                confirmed_event.event_type, raw_description
            )
            confirmed_event = replace(
                confirmed_event,
                description=display_description,
            )
            event_meta = {
                **meta,
                "display_description": display_description,
                "raw_objective_description": raw_description,
            }
            if self.trace:
                print(
                    "[TRACE][EVENT_AFTER_VLM] "
                    + json.dumps(asdict(confirmed_event), ensure_ascii=False)
                )
            print(
                f"[VLM最终] {confirmed_event.start_time:.2f}s - "
                f"{confirmed_event.end_time:.2f}s | "
                f"事件={confirmed_event.event_type} | "
                f"依据={self._vlm_decision_reason(event_meta, confirmed_event.event_type)} | "
                f"描述={confirmed_event.description}"
            )
            self._stage_or_remember_event(confirmed_event, paths, event_meta)

    def _split_activity_event(
        self,
        event: Event,
        activities: list[str],
        description: str,
        *,
        activity_segments: list[dict[str, Any]] | None = None,
        frame_count: int = 0,
        primary_activity: str | None = None,
    ) -> list[Event]:
        activities = list(dict.fromkeys(
            event_type
            for event_type in activities
            if event_type in ALL_EVENTS
        ))
        primary_activity = (
            primary_activity
            if (
                isinstance(primary_activity, str)
                and primary_activity in ALL_EVENTS
            )
            else None
        )
        if not activities:
            return []
        duration = max(0.0, event.end_time - event.start_time)
        events: list[Event] = []

        # 有分段时严格按每一个 segment 生成事件，保留 A→B→A 中重复的 A，
        # 并同时使用 start_frame/end_frame，不再把动作强行铺到下一段起点。
        represented_types: set[str] = set()
        if activity_segments and frame_count > 0:
            for segment in activity_segments:
                event_type = str(segment.get("event_type", ""))
                if event_type not in activities:
                    continue
                try:
                    start_frame = max(1, min(frame_count, int(segment["start_frame"])))
                    end_frame = max(start_frame, min(frame_count, int(segment["end_frame"])))
                except (KeyError, TypeError, ValueError):
                    continue
                denominator = max(1, frame_count - 1)
                start_time = event.start_time + ((start_frame - 1) / denominator) * duration
                end_time = (
                    event.end_time
                    if end_frame >= frame_count
                    else event.start_time + (end_frame / denominator) * duration
                )
                events.append(
                    Event(
                        event_type=event_type,
                        start_time=start_time,
                        end_time=end_time,
                        track_id=event.track_id,
                        confidence=event.confidence,
                        description=description,
                        is_final_window=False,
                    )
                )
                represented_types.add(event_type)

        # 不再把每一个缺少 segment 的 true 都铺满整个候选窗口。2B 模型
        # 偶尔会把开始、结束、入座、离座同时设为 true；旧逻辑会把一次
        # 协议错误放大成多条覆盖全窗口的假事件。
        unsegmented = [
            event_type
            for event_type in activities
            if event_type not in represented_types
        ]

        # primary_event 只作摘要。无分段时只有唯一事件可以铺满整窗；
        # 多事件必须由逐帧复核或 activity_segments 给出实际边界。
        fallback_types = unsegmented if len(activities) == 1 else []

        if self.trace:
            dropped_unsegmented = [
                event_type
                for event_type in unsegmented
                if event_type not in fallback_types
            ]
            if dropped_unsegmented:
                print(
                    "[TRACE][VLM_UNSEGMENTED_DROPPED] "
                    + json.dumps(dropped_unsegmented, ensure_ascii=False)
                )

        for event_type in fallback_types:
            events.append(
                Event(
                    event_type=event_type,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    track_id=event.track_id,
                    confidence=event.confidence,
                    description=description,
                    is_final_window=False,
                )
            )
        if event.is_final_window and events:
            final_index = max(
                range(len(events)), key=lambda index: events[index].end_time
            )
            events[final_index] = replace(
                events[final_index], is_final_window=True
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
            screenshot_path=paths[0] if paths else "",
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
        if self.trace:
            trace_record = record.to_dict()
            embedding = trace_record.pop("embedding", [])
            metadata = trace_record.get("metadata")
            if isinstance(metadata, dict):
                metadata = dict(metadata)
                vlm_meta = metadata.get("vlm")
                if isinstance(vlm_meta, dict):
                    vlm_meta = dict(vlm_meta)
                    raw_text = str(vlm_meta.pop("_vlm_raw", ""))
                    if raw_text:
                        vlm_meta["vlm_raw_summary"] = {
                            "characters": len(raw_text),
                            "preview": raw_text[:160].replace("\n", " "),
                        }
                    metadata["vlm"] = vlm_meta
                trace_record["metadata"] = metadata
            trace_record["embedding_summary"] = {
                "dimension": len(embedding),
                "first_values": embedding[:8],
            }
            print(
                "[TRACE][MEMORY_OUTPUT] "
                + json.dumps(trace_record, ensure_ascii=False, default=str)
            )
        print(
            f"[Memory写入] {record.event_type} "
            f"{record.start_time:.2f}s -> {record.end_time:.2f}s | {record.caption}"
        )

    def _fill_uncovered_timeline(
        self,
        start_time: float,
        end_time: float,
        *,
        minimum_gap: float = 1.0,
    ) -> None:
        """把没有产生 Event 的明显时间空档补为 other_behavior。"""
        start = max(0.0, float(start_time))
        end = max(start, float(end_time))
        ordered = sorted(
            self.events,
            key=lambda item: (float(item.start_time), float(item.end_time)),
        )
        cursor = start
        gaps: list[tuple[float, float]] = []
        for item in ordered:
            item_start = max(start, float(item.start_time))
            item_end = min(end, float(item.end_time))
            if item_start - cursor >= minimum_gap:
                gaps.append((cursor, item_start))
            cursor = max(cursor, item_end)
        if end - cursor >= minimum_gap:
            gaps.append((cursor, end))

        for gap_start, gap_end in gaps:
            description = "该时段未检测到可确认的学习动作。"
            gap_event = Event(
                event_type="other_behavior",
                start_time=gap_start,
                end_time=gap_end,
                track_id=None,
                confidence=1.0,
                description=description,
            )
            try:
                paths = self._extract_event_frames(gap_event)
            except Exception:
                paths = []
            self._remember_event(
                gap_event,
                paths,
                {
                    "event_confirmed": True,
                    "events": {
                        name: name == "other_behavior" for name in ALL_EVENTS
                    },
                    "activity_segments": [],
                    "timeline_fallback": "uncovered_timeline_other",
                    "display_description": description,
                    "raw_objective_description": "",
                    "synthetic_timeline_gap": True,
                },
            )

        if gaps:
            self.events.sort(
                key=lambda item: (float(item.start_time), float(item.end_time))
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

    def _analyze_frame(
        self,
        video_frame: VideoFrame,
        event_handler: Optional[Callable[[Event], None]] = None,
    ) -> None:
        self.current_frame = video_frame
        self.last_timestamp = video_frame.timestamp
        self.last_analysis_time = video_frame.timestamp
        self.latest_result = self.pipeline.process_frame(video_frame)
        self._record_analysis_observation(video_frame)

        handler = event_handler or self._handle_event
        for event in self.latest_result["events"]:
            handler(event)

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

    @staticmethod
    def _summary_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按时间排序并合并相邻同类窗口，供全事件总结使用。"""
        ordered = sorted(records, key=lambda item: float(item["start_time"]))
        merged: list[dict[str, Any]] = []
        for record in ordered:
            current = dict(record)
            if merged and str(merged[-1].get("event_type")) == str(
                current.get("event_type")
            ):
                merged[-1]["end_time"] = max(
                    float(merged[-1]["end_time"]),
                    float(current["end_time"]),
                )
                if not str(merged[-1].get("caption", "")).strip():
                    merged[-1]["caption"] = current.get("caption", "")
                continue
            merged.append(current)
        return merged

    @staticmethod
    def _fallback_video_summary(records: list[dict[str, Any]]) -> str:
        if not records:
            return "本次视频中没有确认到可总结的具体事件。"
        ordered = EndToEndRunner._summary_records(records)
        event_types = [str(record["event_type"]) for record in ordered]
        has_other = "other_behavior" in event_types
        actions = [
            _SUMMARY_ACTION_CN[event_type]
            for event_type in event_types
            if event_type in _SUMMARY_ACTION_CN
        ]

        if not actions:
            return "视频中，人物进行了学习准备、整理或动作切换。"
        if len(actions) == 1:
            process = f"视频中，人物{actions[0]}"
        elif len(actions) == 2:
            process = f"视频中，人物先{actions[0]}，随后{actions[1]}"
        elif len(actions) == 3:
            process = (
                f"视频中，人物先{actions[0]}，随后{actions[1]}，"
                f"最后{actions[2]}"
            )
        elif len(actions) == 4:
            process = (
                f"视频中，人物先{actions[0]}，随后{actions[1]}，"
                f"接着{actions[2]}，最后{actions[3]}"
            )
        else:
            process = "视频中，人物依次" + "、".join(actions[:-1]) + f"和{actions[-1]}"

        if has_other:
            return process + "；过程中还包含学习准备、整理或动作切换。"
        return process + "。"

    @staticmethod
    def _is_objective_video_summary(summary: str) -> bool:
        text = str(summary or "").strip()
        if not text or any(term in text for term in _NON_OBJECTIVE_SUMMARY_TERMS):
            return False
        repeated_other_phrases = (
            "准备、整理或动作切换",
            "准备、整理和动作切换",
        )
        return not any(text.count(phrase) > 1 for phrase in repeated_other_phrases)

    def _generate_video_summary(self) -> str:
        records = self._summary_records(
            [record.to_dict() for record in self.memory_store.list_all()]
        )
        if not records:
            return self._fallback_video_summary(records)
        # 正式产品路径使用结构化事件直接生成总结，确保总结不能扩写出
        # 兴趣、态度、身份、原因或画面中没有发生的动作。
        if self.summary_analyzer is None:
            return self._fallback_video_summary(records)
        try:
            summary = self.summary_analyzer(records).strip()
            if self._is_objective_video_summary(summary):
                return summary
        except Exception as exc:
            self.errors.append(f"全事件总结失败: {exc}")
        return self._fallback_video_summary(records)

    def _run_realtime(
        self,
        *,
        max_frames: Optional[int],
        max_duration: Optional[float],
    ) -> dict[str, Any]:
        source = open_source(self.source) if isinstance(self.source, int) else self.source
        owns_source = isinstance(self.source, int)
        # 为 YOLO 短时抖动保留数秒余量；仍有限界，避免摄像头长时间积压原始帧。
        frame_queue: queue.Queue = queue.Queue(maxsize=8)
        event_queue: queue.Queue = queue.Queue()
        frame_sentinel = object()
        event_sentinel = object()
        first_timestamp: Optional[float] = None
        last_timestamp: Optional[float] = None
        processed = 0
        last_enqueued: Optional[float] = None

        def enqueue_event(event: Event) -> None:
            # 必须在事件产生时立刻固化关键帧；等待 VLM 时内存缓冲仍会滚动。
            paths = self._extract_event_frames(event)
            event_queue.put((event, paths))

        def analysis_worker() -> None:
            while True:
                item = frame_queue.get()
                try:
                    if item is frame_sentinel:
                        return
                    self._analyze_frame(item, event_handler=enqueue_event)
                except Exception as exc:
                    self.errors.append(f"实时分析失败: {exc}")
                finally:
                    frame_queue.task_done()

        def event_worker() -> None:
            while True:
                item = event_queue.get()
                try:
                    if item is event_sentinel:
                        return
                    event, paths = item
                    self._handle_event(event, paths=paths)
                except Exception as exc:
                    self.errors.append(f"实时 VLM/记忆处理失败: {exc}")
                finally:
                    event_queue.task_done()

        worker = threading.Thread(target=analysis_worker, name="video-analysis", daemon=True)
        semantic_worker = threading.Thread(
            target=event_worker,
            name="video-semantic-analysis",
            daemon=True,
        )
        worker.start()
        semantic_worker.start()

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
            frame_queue.put(frame_sentinel)
            worker.join()
            # Detection/Event 已停止产生新事件，再等待全部 VLM 任务完成。
            event_queue.put(event_sentinel)
            semantic_worker.join()

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
            self._flush_temporal_pending(end_of_stream=True)
            self._fill_uncovered_timeline(
                float(stats.get("start_time", 0.0)),
                float(stats.get("end_time", self.last_timestamp)),
            )
        finally:
            if self.recorder is not None:
                self.recorder.close()

        self.video_summary = self._generate_video_summary()

        return {
            "stream": stats,
            "vlm_calls": {
                "main": self.vlm_main_calls,
                "boundary": self.vlm_boundary_calls,
                "position": self.vlm_position_calls,
                "window_label": self.vlm_window_label_calls,
                "transition_frame": self.vlm_transition_frame_calls,
                "summary": self.vlm_summary_calls,
                "total": (
                    self.vlm_main_calls
                    + self.vlm_boundary_calls
                    + self.vlm_position_calls
                    + self.vlm_window_label_calls
                    + self.vlm_transition_frame_calls
                    + self.vlm_summary_calls
                ),
            },
            "events_detected": len(self.events) + len(self.rejected_events),
            "memories_saved": len(self.memory_store),
            "events": [asdict(event) for event in self.events],
            "rejected_events": self.rejected_events,
            "errors": self.errors,
            "video_summary": self.video_summary,
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
