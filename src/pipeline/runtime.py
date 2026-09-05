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
    description_has_transition_evidence,
    infer_activity_from_description,
    infer_activities_from_vlm_meta,
    should_persist_to_memory,
)
from ..vlm.qwen_vlm import load_model
from ..vlm.prompt import EVENT_TYPE_CN
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
        if meta.get("primary_event") == event_type:
            return "VLM events=true 且判为主事件"
        return "VLM确认，且通过事件契约校验"

    @staticmethod
    def _has_strong_action_evidence(
        event_type: str,
        meta: dict[str, Any],
    ) -> bool:
        """判断新事件是否有足以立即推翻历史状态的明确动作证据。"""
        segments = meta.get("activity_segments", [])
        segmented_types = {
            str(item.get("event_type", ""))
            for item in segments
            if isinstance(item, dict)
        } if isinstance(segments, list) else set()
        # 一个窗口中模型明确切出了多个动作，优先保留真实的短时切换。
        if len(segmented_types) > 1 and event_type in segmented_types:
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
                for word in ("落笔", "写字", "书写", "做题", "记笔记", "用笔记录")
            )
        if event_type == "phone_usage":
            return "手机" in description and any(
                word in description
                for word in ("使用", "操作", "滑动", "点击", "查看", "看手机", "注视")
            )
        if event_type == "computer_usage":
            return any(word in description for word in ("电脑", "笔记本电脑")) and any(
                word in description
                for word in ("使用", "操作", "键盘", "鼠标", "屏幕", "点击", "敲击")
            )
        if event_type == "communication_distraction":
            return any(word in description for word in ("交谈", "说话", "通话", "与人交流"))
        if event_type == "other_behavior":
            return description_has_transition_evidence(description)
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
        label = EVENT_TYPE_CN[stable_event_type]
        description = f"结合前后窗口连续性，时序校正为{label}。"
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
        description = "人物处于动作切换、整理或视频收尾阶段，归为其他。"
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

        duration = max(0.0, event.end_time - event.start_time)
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
                if (
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
        """只细分容易混入过渡动作的窗口，避免把所有 VLM 调用翻倍。"""
        if event.event_type not in _BEHAVIOR_EVENT_TYPES:
            return False
        duration = max(0.0, event.end_time - event.start_time)
        if duration < 6.0 or len(paths) < 7:
            return False
        if meta is not None:
            if meta.get("activity_segments"):
                return False
            description = str(meta.get("objective_description", ""))
            if meta.get("parse_error") or description_has_transition_evidence(description):
                return True

        # candidate_scores 已按全部分析帧计数。低于 50% 且与稳定历史冲突
        # 时，单个 8 秒标签不可靠，拆成两个约 4 秒的小窗口再判断。
        support = max(event.candidate_scores.values(), default=0.0)
        stable = self._stable_behavior_by_track.get(0)
        conflicts_with_context = stable is None or event.event_type != stable
        return support < 0.5 and conflicts_with_context

    def _refine_window_halves(
        self,
        event: Event,
        paths: list[str],
    ) -> None:
        midpoint = (event.start_time + event.end_time) / 2.0
        middle_index = len(paths) // 2
        halves = (
            (event.start_time, midpoint, paths[: middle_index + 1]),
            (midpoint, event.end_time, paths[middle_index:]),
        )
        print(
            f"[窗口细分] {event.start_time:.2f}s - {event.end_time:.2f}s | "
            "物体线索较弱、与上下文冲突或包含整理动作，改用两个短窗口复核"
        )
        for start_time, end_time, sub_paths in halves:
            if not sub_paths or end_time <= start_time:
                continue
            # 细分窗口不继承原来的单一行为候选，避免错误候选继续锚定 VLM。
            sub_event = replace(
                event,
                event_type="other_behavior",
                start_time=start_time,
                end_time=end_time,
                description="待 VLM 细分复核",
                candidate_scores={},
                unclassified_ratio=1.0,
            )
            self._handle_event(sub_event, list(sub_paths), allow_refine=False)

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
            if allow_refine and self._should_refine_window(event, paths):
                self._refine_window_halves(event, paths)
                return
            try:
                print(f"[VLM请求] 候选={event.event_type} | 关键帧={len(paths)}张")
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

            if allow_refine and self._should_refine_window(event, paths, meta):
                self._refine_window_halves(event, paths)
                return

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
                description_activity = infer_activity_from_description(description)
                candidate_activity = event.event_type
                if (
                    description_activity in _BEHAVIOR_EVENT_TYPES
                    and description_activity != "other_behavior"
                ):
                    fallback_activity = description_activity
                    fallback_source = "vlm_description"
                elif description_activity == "other_behavior":
                    fallback_activity = description_activity
                    fallback_source = "vlm_description_generic"
                else:
                    stable_activity = self._stable_behavior_by_track.get(0)
                    if stable_activity in _BEHAVIOR_EVENT_TYPES and stable_activity != "other_behavior":
                        fallback_activity = stable_activity
                        fallback_source = "previous_stable_event"
                    elif candidate_activity != "other_behavior":
                        fallback_activity = candidate_activity
                        fallback_source = "event_engine_candidate"
                    else:
                        fallback_activity = "other_behavior"
                        fallback_source = "unclassified_seated_window"

                if not description or description == "[VLM未返回有效description]":
                    description = "人物仍在学习位置，但未识别出可归入具体类别的动作。"
                description = description.rstrip("。！？!?；;，, ")
                fallback_label = EVENT_TYPE_CN[fallback_activity]
                if not description.endswith(fallback_label):
                    description += f"，{fallback_label}"
                description += "。"

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
            if self.trace:
                print(
                    "[TRACE][EVENT_AFTER_VLM] "
                    + json.dumps(asdict(confirmed_event), ensure_ascii=False)
                )
            print(
                f"[VLM最终] {confirmed_event.start_time:.2f}s - "
                f"{confirmed_event.end_time:.2f}s | "
                f"事件={confirmed_event.event_type} | "
                f"依据={self._vlm_decision_reason(meta, confirmed_event.event_type)} | "
                f"描述={confirmed_event.description}"
            )
            self._stage_or_remember_event(confirmed_event, paths, meta)

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
        candidate_event_type = event.event_type
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
                start_time = event.start_time + ((start_frame - 1) / frame_count) * duration
                end_time = event.start_time + (end_frame / frame_count) * duration
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
                represented_types.add(event_type)

        # 不再把每一个缺少 segment 的 true 都铺满整个候选窗口。2B 模型
        # 偶尔会把开始、结束、入座、离座同时设为 true；旧逻辑会把一次
        # 协议错误放大成多条覆盖全窗口的假事件。
        unsegmented = [
            event_type
            for event_type in activities
            if event_type not in represented_types
        ]

        fallback_types: list[str] = []

        def add_fallback(event_type: str | None) -> None:
            if (
                isinstance(event_type, str)
                and event_type in unsegmented
                and event_type not in fallback_types
            ):
                fallback_types.append(event_type)

        candidate_available = candidate_event_type in unsegmented
        primary_available = primary_activity in unsegmented

        # 明确的位置/生命周期候选已有 EventEngine 时序证据；缺少 segment
        # 时它优先，不能再把不同的 VLM 主事件铺成同一个完整窗口。
        if candidate_available and candidate_event_type != "other_behavior":
            add_fallback(candidate_event_type)
        else:
            # 通用 other 只是占位候选，优先采用 VLM 的具体主事件。
            if (
                primary_available
                and (
                    primary_activity not in _TRANSITION_EVENT_TYPES
                    or primary_activity == candidate_event_type
                )
            ):
                add_fallback(primary_activity)
            if not fallback_types:
                add_fallback(candidate_event_type)

        # 兼容只返回一个明确普通事件、但漏写 primary/segment 的小模型输出。
        if len(activities) == 1 and activities[0] not in _TRANSITION_EVENT_TYPES:
            add_fallback(activities[0])

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
