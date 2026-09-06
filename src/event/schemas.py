from dataclasses import dataclass, field
from typing import Dict, List


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
    # 当前窗口内全部平等候选在“所有分析帧”中的出现比例。旧调用方无需提供。
    # 该字段只是物体线索，不代表人物已经执行了对应事件。
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    # 没有提供具体物体候选的分析帧比例，仅用于诊断，不是“其他”的票数。
    unclassified_ratio: float = 0.0
    # 由 EventEngine.finalize() 关闭的最后一个窗口。Runtime 必须先执行
    # 尾窗口规则，再考虑与上一稳定行为合并。
    is_final_window: bool = False
