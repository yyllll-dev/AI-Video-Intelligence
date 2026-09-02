"""demo/core —— UI 的核心常量与模拟数据。

独立成文件夹的原因：这里的「事件映射」和「模拟数据」是全新的、可复用的内容，
与 app.py 的 UI 布局职责分离。后续接入真实后端时，只需替换 mock_data 里的函数。
"""

from .events import EVENT_LABELS, CLASS_LABELS, event_label, class_label
from .mock_data import (
    mock_detections,
    mock_current_event,
    mock_timeline,
    mock_vlm_result,
    mock_search,
)

__all__ = [
    "EVENT_LABELS",
    "CLASS_LABELS",
    "event_label",
    "class_label",
    "mock_detections",
    "mock_current_event",
    "mock_timeline",
    "mock_vlm_result",
    "mock_search",
]
