"""高嘉沐负责模块使用的统一数据结构。

本文件不重复定义队长负责的 Event 类，而是通过 EventLike 协议接收它。
只要队长的 Event 对象具有截图中约定的字段，就可以直接传入本模块。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

EmbeddingVector = tuple[float, ...]


@runtime_checkable
class EventLike(Protocol):
    """A模块 Event 接口的最小约定。

    对应队长截图中的字段：
    event_type、start_time、end_time、track_id、confidence、description。
    """

    event_type: str
    start_time: float
    end_time: float
    track_id: int | None
    confidence: float
    description: str


@dataclass(slots=True)
class MemoryRecord:
    """一条可被保存和检索的视频记忆。"""

    event_id: str
    event_type: str
    timestamp: float
    start_time: float
    end_time: float
    track_id: int | None
    confidence: float
    caption: str
    screenshot_path: str
    video_path: str
    embedding: EmbeddingVector
    embedding_model: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("event_type 不能为空")
        if self.start_time < 0:
            raise ValueError("start_time 不能小于 0")
        if self.end_time < self.start_time:
            raise ValueError("end_time 不能早于 start_time")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须在 0 到 1 之间")
        if not self.caption.strip():
            raise ValueError("caption 不能为空")
        if not self.embedding:
            raise ValueError("embedding 不能为空")

    @classmethod
    def from_event(
        cls,
        event: EventLike,
        *,
        embedding: list[float] | tuple[float, ...],
        embedding_model: str,
        caption: str | None = None,
        screenshot_path: str = "",
        video_path: str = "",
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> "MemoryRecord":
        """把队长的 Event 转换为本模块的 MemoryRecord。

        当前接口中的 description 会作为默认 caption；Day 4 接入 Qwen-VL 后，
        可以通过 caption 参数传入更完整的事件描述。
        """

        final_caption = (caption or event.description or event.event_type).strip()
        return cls(
            event_id=event_id or uuid4().hex,
            event_type=event.event_type.strip(),
            timestamp=float(event.start_time),
            start_time=float(event.start_time),
            end_time=float(event.end_time),
            track_id=event.track_id,
            confidence=float(event.confidence),
            caption=final_caption,
            screenshot_path=screenshot_path,
            video_path=video_path,
            embedding=tuple(float(value) for value in embedding),
            embedding_model=embedding_model,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为便于数据库写入、JSON返回和UI对接的字典。"""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """一次检索返回给 Pipeline 或 UI 的结果。"""

    record: MemoryRecord
    similarity_score: float

    def to_dict(self) -> dict[str, Any]:
        result = self.record.to_dict()
        result["similarity_score"] = self.similarity_score
        return result
