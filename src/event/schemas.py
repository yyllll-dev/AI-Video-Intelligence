from dataclasses import dataclass
from typing import List


@dataclass
class TrackingResult:
    """目标检测与跟踪结果"""

    frame_id: int
    timestamp: float
    track_id: int
    class_name: str
    confidence: float
    bbox: List[float]


@dataclass
class Event:
    """视频事件"""

    event_type: str
    start_time: float
    end_time: float
    track_id: int | None
    confidence: float
    description: str = ""