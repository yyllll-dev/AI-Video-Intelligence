"""轻量级多目标跟踪器

本文件：src/tracking/tracker.py

职责：
    接收 YOLO Detector 输出的 DetectionResult / detection dict，
    为连续视频中的目标分配并维持 track_id，
    输出统一的 TrackingResult。

数据流：

    VideoFrame
        ↓
    YOLO Detector
        ↓
    DetectionResult
        ↓
    Tracker
        ↓
    TrackingResult
        ↓
    EventEngine

当前版本：
    使用 IoU + 中心点距离进行轻量级多目标跟踪。

说明：
    1. Tracker 不负责目标检测。
    2. Tracker 不负责事件判断。
    3. Tracker 唯一核心任务是：
       判断当前帧的目标是不是上一帧已经存在的目标。
    4. track_id 由 Tracker 负责生成和维护。
    5. 与当前 pipeline.py 的接口直接兼容。
"""

from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, List, Optional

from ..detection.detector import DetectionResult
from ..event.schemas import TrackingResult


# ============================================================
# Tracker 内部状态
# ============================================================

@dataclass
class Track:
    """Tracker 内部维护的一个目标。"""

    track_id: int
    class_name: str
    bbox: List[float]
    confidence: float
    last_seen: float
    missed_frames: int = 0


# ============================================================
# 轻量级多目标跟踪器
# ============================================================

class SimpleTracker:
    """基于 IoU + 中心点距离的轻量级多目标跟踪器。

    输入：
        Pipeline 传入的 detection 列表。

    支持：
        DetectionResult
        dict

    输出：
        TrackingResult 列表。
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        distance_threshold: float = 100.0,
        max_missed_frames: int = 10,
    ):
        """
        参数：

        iou_threshold：
            两个目标框的 IoU 达到该阈值时，
            可以认为两个框可能属于同一个目标。

        distance_threshold：
            两个目标中心点的最大允许距离。

        max_missed_frames：
            一个目标连续多少帧没有检测到后，
            才将该目标从 Tracker 中删除。
        """

        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.max_missed_frames = max_missed_frames

        # 当前正在跟踪的目标
        self.tracks: Dict[int, Track] = {}

        # 下一个可使用的 track_id
        self.next_track_id = 1

    # ========================================================
    # Pipeline 调用入口
    # ========================================================

    def __call__(
        self,
        detections: List[Any],
        timestamp: float,
    ) -> List[TrackingResult]:
        """允许 Pipeline 直接调用 Tracker。

        Pipeline 中的调用形式：

            tracker(detections, timestamp)

        实际执行：

            tracker.update(detections, timestamp)
        """

        return self.update(
            detections,
            timestamp,
        )

    # ========================================================
    # Tracker 主逻辑
    # ========================================================

    def update(
        self,
        detections: List[Any],
        timestamp: float,
    ) -> List[TrackingResult]:
        """处理当前帧的检测结果。

        参数：
            detections：
                DetectionResult 或 detection dict 列表。

            timestamp：
                当前视频帧时间戳。

        返回：
            当前帧所有检测目标对应的 TrackingResult。
        """

        # ----------------------------------------------------
        # 1. 统一 Detection 数据格式
        # ----------------------------------------------------

        normalized_detections = [
            self._normalize_detection(item)
            for item in detections
        ]

        # ----------------------------------------------------
        # 2. 当前帧没有检测到目标
        # ----------------------------------------------------

        if not normalized_detections:

            self._mark_all_tracks_missed()

            self._remove_lost_tracks()

            return []

        # ----------------------------------------------------
        # 3. 当前检测与历史 Track 进行匹配
        # ----------------------------------------------------

        detection_to_track: Dict[int, int] = {}

        matched_track_ids = set()

        # 为每一个 detection 寻找最合适的历史 Track
        for detection_index, detection in enumerate(
            normalized_detections
        ):

            best_track_id: Optional[int] = None
            best_score = -1.0

            for track_id, track in self.tracks.items():

                # 一个 Track 在当前帧只能匹配一次
                if track_id in matched_track_ids:
                    continue

                # 类别不同，不认为是同一个目标
                if track.class_name != detection["class_name"]:
                    continue

                # ------------------------------------------------
                # 计算 IoU
                # ------------------------------------------------

                iou = self._calculate_iou(
                    track.bbox,
                    detection["bbox"],
                )

                # ------------------------------------------------
                # 计算中心点距离
                # ------------------------------------------------

                distance = self._center_distance(
                    track.bbox,
                    detection["bbox"],
                )

                # ------------------------------------------------
                # 判断是否可能是同一个目标
                # ------------------------------------------------

                if (
                    iou < self.iou_threshold
                    and distance > self.distance_threshold
                ):
                    continue

                # ------------------------------------------------
                # 综合匹配分数
                # ------------------------------------------------

                distance_score = max(
                    0.0,
                    1.0
                    - (
                        distance
                        / self.distance_threshold
                    ),
                )

                score = (
                    0.7 * iou
                    + 0.3 * distance_score
                )

                if score > best_score:

                    best_score = score
                    best_track_id = track_id

            # ------------------------------------------------
            # 找到了历史 Track
            # ------------------------------------------------

            if best_track_id is not None:

                detection_to_track[
                    detection_index
                ] = best_track_id

                matched_track_ids.add(
                    best_track_id
                )

        # ----------------------------------------------------
        # 4. 更新已经匹配上的 Track
        # ----------------------------------------------------

        for detection_index, track_id in (
            detection_to_track.items()
        ):

            detection = normalized_detections[
                detection_index
            ]

            track = self.tracks[track_id]

            track.bbox = list(
                detection["bbox"]
            )

            track.confidence = float(
                detection["confidence"]
            )

            track.last_seen = timestamp

            track.missed_frames = 0

        # ----------------------------------------------------
        # 5. 没有匹配上的 Detection → 创建新 Track
        # ----------------------------------------------------

        for detection_index, detection in enumerate(
            normalized_detections
        ):

            if detection_index in detection_to_track:
                continue

            track_id = self._create_track(
                detection,
                timestamp,
            )

            detection_to_track[
                detection_index
            ] = track_id

            matched_track_ids.add(
                track_id
            )

        # ----------------------------------------------------
        # 6. 没有匹配上的旧 Track → missed_frames + 1
        # ----------------------------------------------------

        for track_id, track in self.tracks.items():

            if track_id not in matched_track_ids:

                track.missed_frames += 1

        # ----------------------------------------------------
        # 7. 删除连续消失太久的 Track
        # ----------------------------------------------------

        self._remove_lost_tracks()

        # ----------------------------------------------------
        # 8. 转换成 TrackingResult
        # ----------------------------------------------------

        tracking_results: List[TrackingResult] = []

        for detection_index, detection in enumerate(
            normalized_detections
        ):

            track_id = detection_to_track[
                detection_index
            ]

            track = self.tracks.get(track_id)

            if track is None:
                continue

            tracking_results.append(
                TrackingResult(
                    frame_id=int(
                        detection.get(
                            "frame_id",
                            0,
                        )
                    ),
                    timestamp=float(
                        detection.get(
                            "timestamp",
                            timestamp,
                        )
                    ),
                    track_id=track_id,
                    class_name=track.class_name,
                    confidence=track.confidence,
                    bbox=list(track.bbox),
                )
            )

        return tracking_results

    # ========================================================
    # Detection 格式统一
    # ========================================================

    @staticmethod
    def _normalize_detection(
        item: Any,
    ) -> Dict[str, Any]:
        """将 DetectionResult / dict
        统一转换成 Tracker 内部使用的 dict。
        """

        # ----------------------------------------------------
        # DetectionResult
        # ----------------------------------------------------

        if isinstance(item, DetectionResult):

            return {
                "frame_id": item.frame_id,
                "timestamp": item.timestamp,
                "class_name": item.class_name,
                "confidence": item.confidence,
                "bbox": list(item.bbox),
            }

        # ----------------------------------------------------
        # dict
        # ----------------------------------------------------

        if isinstance(item, dict):

            class_name = item.get(
                "class_name",
                item.get("class"),
            )

            if not class_name:

                raise ValueError(
                    "A detection requires class_name"
                )

            if "bbox" not in item:

                raise ValueError(
                    "A detection requires bbox"
                )

            return {
                "frame_id": item.get(
                    "frame_id",
                    0,
                ),
                "timestamp": item.get(
                    "timestamp",
                    0.0,
                ),
                "class_name": str(
                    class_name
                ),
                "confidence": float(
                    item.get(
                        "confidence",
                        1.0,
                    )
                ),
                "bbox": list(
                    item["bbox"]
                ),
            }

        # ----------------------------------------------------
        # 不支持的数据类型
        # ----------------------------------------------------

        raise TypeError(
            "A detection must be a "
            "DetectionResult or dict"
        )

    # ========================================================
    # 创建新的 Track
    # ========================================================

    def _create_track(
        self,
        detection: Dict[str, Any],
        timestamp: float,
    ) -> int:
        """为一个新的检测目标创建 Track。"""

        track_id = self.next_track_id

        self.next_track_id += 1

        self.tracks[track_id] = Track(
            track_id=track_id,
            class_name=detection["class_name"],
            bbox=list(
                detection["bbox"]
            ),
            confidence=float(
                detection["confidence"]
            ),
            last_seen=timestamp,
            missed_frames=0,
        )

        return track_id

    # ========================================================
    # 没有检测到目标
    # ========================================================

    def _mark_all_tracks_missed(self) -> None:
        """当前帧没有任何检测结果时，
        所有历史 Track 都暂时标记为未检测到。
        """

        for track in self.tracks.values():

            track.missed_frames += 1

    # ========================================================
    # 删除丢失目标
    # ========================================================

    def _remove_lost_tracks(self) -> None:
        """删除连续多帧没有出现的 Track。"""

        lost_track_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames
            > self.max_missed_frames
        ]

        for track_id in lost_track_ids:

            del self.tracks[track_id]

    # ========================================================
    # IoU
    # ========================================================

    @staticmethod
    def _calculate_iou(
        bbox1: List[float],
        bbox2: List[float],
    ) -> float:
        """计算两个 Bounding Box 的 IoU。

        bbox 格式：

            [x1, y1, x2, y2]
        """

        if (
            len(bbox1) != 4
            or len(bbox2) != 4
        ):
            return 0.0

        # ----------------------------------------------------
        # 交集区域
        # ----------------------------------------------------

        x1 = max(
            bbox1[0],
            bbox2[0],
        )

        y1 = max(
            bbox1[1],
            bbox2[1],
        )

        x2 = min(
            bbox1[2],
            bbox2[2],
        )

        y2 = min(
            bbox1[3],
            bbox2[3],
        )

        intersection_width = max(
            0.0,
            x2 - x1,
        )

        intersection_height = max(
            0.0,
            y2 - y1,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        # ----------------------------------------------------
        # 两个框各自面积
        # ----------------------------------------------------

        area1 = (
            max(
                0.0,
                bbox1[2] - bbox1[0],
            )
            *
            max(
                0.0,
                bbox1[3] - bbox1[1],
            )
        )

        area2 = (
            max(
                0.0,
                bbox2[2] - bbox2[0],
            )
            *
            max(
                0.0,
                bbox2[3] - bbox2[1],
            )
        )

        # ----------------------------------------------------
        # 并集面积
        # ----------------------------------------------------

        union_area = (
            area1
            + area2
            - intersection_area
        )

        if union_area <= 0:

            return 0.0

        return (
            intersection_area
            / union_area
        )

    # ========================================================
    # 中心点距离
    # ========================================================

    @staticmethod
    def _center_distance(
        bbox1: List[float],
        bbox2: List[float],
    ) -> float:
        """计算两个 Bounding Box 中心点的欧氏距离。"""

        if (
            len(bbox1) != 4
            or len(bbox2) != 4
        ):
            return float("inf")

        center1_x = (
            bbox1[0]
            + bbox1[2]
        ) / 2.0

        center1_y = (
            bbox1[1]
            + bbox1[3]
        ) / 2.0

        center2_x = (
            bbox2[0]
            + bbox2[2]
        ) / 2.0

        center2_y = (
            bbox2[1]
            + bbox2[3]
        ) / 2.0

        return sqrt(
            (
                center1_x
                - center2_x
            ) ** 2
            +
            (
                center1_y
                - center2_y
            ) ** 2
        )


# ============================================================
# 统一创建入口
# ============================================================

def create_tracker(
    **kwargs,
) -> SimpleTracker:
    """统一创建 Tracker。"""

    return SimpleTracker(**kwargs)