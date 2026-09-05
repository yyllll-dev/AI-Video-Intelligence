from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, is_dataclass
import json
from typing import Any, Optional

from ..detection.detector import DetectionResult
from ..detection.video_source import VideoFrame
from ..event.engine import EventEngine
from ..event.schemas import TrackingResult


class VideoPipeline:
    """
    视频智能分析主 Pipeline

    Video Input
        ↓
    Detection
        ↓
    Tracking
        ↓
    Event
        ↓
    VLM
        ↓
    Memory
        ↓
    Retrieval
        ↓
    Replay

    当前阶段：
    VideoFrame
        ↓
    Detector
        ↓
    DetectionResult
        ↓
    Tracker
        ↓
    TrackingResult
        ↓
    EventEngine
    """

    def __init__(
        self,
        detector: Optional[Callable[[VideoFrame], Iterable[Any]]] = None,
        tracker: Optional[
            Callable[[list[dict[str, Any]], float], Iterable[Any]]
        ] = None,
        event_engine: Optional[EventEngine] = None,
        trace: bool = False,
    ):
        """
        创建视频分析 Pipeline。

        detector：
            接收 VideoFrame，返回 DetectionResult 列表。

        tracker：
            接收标准化后的 detection 字典列表和 timestamp，
            返回 TrackingResult 列表。

        event_engine：
            接收 TrackingResult 列表并生成 Event。
        """
        self.detector = detector
        self.tracker = tracker
        self.event_engine = event_engine or EventEngine()
        self.trace = trace

        # 后续模块暂时预留
        self.vlm = None
        self.memory = None
        self.retriever = None

    def process_frame(self, video_frame: VideoFrame) -> dict[str, Any]:
        """
        处理一帧视频。

        数据流：

        VideoFrame
            ↓
        Detector
            ↓
        DetectionResult
            ↓
        Tracker
            ↓
        TrackingResult
            ↓
        EventEngine
            ↓
        Event
        """

        # =====================================================
        # 1. Detection
        # =====================================================

        self._trace(
            "FRAME",
            {
                "frame_id": video_frame.frame_id,
                "timestamp": round(video_frame.timestamp, 3),
                "shape": list(video_frame.frame.shape),
            },
        )

        if self.detector:
            detections = list(self.detector(video_frame))
        else:
            detections = []
        self._trace("YOLO_OUTPUT", detections)

        # =====================================================
        # 2. 将 DetectionResult 转成 Tracker 能理解的格式
        # =====================================================

        normalized_detections = [
            self._as_detection(item)
            for item in detections
        ]

        # =====================================================
        # 3. Tracking
        # =====================================================

        if self.tracker:
            # Tracker 负责真正产生 track_id
            tracked_items = list(
                self.tracker(
                    normalized_detections,
                    video_frame.timestamp,
                )
            )
        else:
            # 当前 Tracker 尚未接入时，暂时不生成假的 track_id。
            # 等 C 完成 tracker.py 后，这里会进入上面的分支。
            tracked_items = []

        # =====================================================
        # 4. 转换成统一 TrackingResult
        # =====================================================

        tracking_results = [
            self._as_tracking_result(
                item,
                video_frame,
                index,
            )
            for index, item in enumerate(tracked_items)
        ]
        self._trace("TRACKER_OUTPUT", tracking_results)

        # =====================================================
        # 5. Event
        # =====================================================

        state_before = self.event_engine.debug_state()
        events = self.event_engine.update(
            tracking_results,
            timestamp=video_frame.timestamp,
        )
        self._trace(
            "EVENT_OUTPUT",
            {
                "state_before": state_before,
                "state_after": self.event_engine.debug_state(),
                "emitted_events": events,
            },
        )

        # =====================================================
        # 6. 后续 VLM / Memory 等模块暂时预留
        # =====================================================

        vlm_results = None
        memory_result = None

        return {
            "detections": detections,
            "tracking_results": tracking_results,
            "events": events,
            "vlm_results": vlm_results,
            "memory_result": memory_result,
        }

    def _trace(self, stage: str, payload: Any) -> None:
        if not self.trace:
            return

        def default(value: Any):
            if is_dataclass(value):
                return asdict(value)
            return repr(value)

        print(
            f"[TRACE][{stage}] "
            + json.dumps(payload, ensure_ascii=False, default=default)
        )

    @staticmethod
    def _as_detection(item: Any) -> dict[str, Any]:
        """
        将 Detector 输出统一转换成 Tracker 的输入格式。

        Detector 输出：
            DetectionResult

        Tracker 输入：
            dict

        注意：
        这里绝对不生成 track_id。
        track_id 必须由 Tracker 负责。
        """

        if isinstance(item, DetectionResult):
            return {
                "frame_id": item.frame_id,
                "timestamp": item.timestamp,
                "class_name": item.class_name,
                "confidence": item.confidence,
                "bbox": item.bbox,
            }

        # 兼容测试代码或其他 Detector
        if isinstance(item, dict):
            return item

        raise TypeError(
            "A detection must be a dictionary or DetectionResult"
        )

    @staticmethod
    def _as_tracking_result(
        item: Any,
        video_frame: VideoFrame,
        index: int,
    ) -> TrackingResult:
        """
        将 Tracker 输出统一转换成 TrackingResult。

        正式情况下，Tracker 应该自己提供 track_id。
        Pipeline 不负责生成真正的 track_id。
        """

        if isinstance(item, TrackingResult):
            return item

        if not isinstance(item, dict):
            raise TypeError(
                "A tracker result must be a dictionary or TrackingResult"
            )

        class_name = item.get(
            "class_name",
            item.get("class"),
        )

        if not class_name:
            raise ValueError(
                "A tracking result requires class_name (or class)"
            )

        if "track_id" not in item:
            raise ValueError(
                "A tracking result requires track_id; "
                "track_id must be generated by Tracker"
            )

        try:
            return TrackingResult(
                frame_id=int(
                    item.get(
                        "frame_id",
                        video_frame.frame_id,
                    )
                ),
                timestamp=float(
                    item.get(
                        "timestamp",
                        video_frame.timestamp,
                    )
                ),
                track_id=int(item["track_id"]),
                class_name=str(class_name),
                confidence=float(
                    item.get("confidence", 1.0)
                ),
                bbox=list(
                    item.get("bbox", [])
                ),
            )

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid tracking result: {item!r}"
            ) from exc


def create_pipeline(**kwargs) -> VideoPipeline:
    """统一创建 VideoPipeline。"""
    return VideoPipeline(**kwargs)
