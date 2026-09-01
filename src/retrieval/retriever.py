"""把 Memory、Embedding 和 Retrieval 串起来的主服务。"""

from __future__ import annotations

from typing import Any, Protocol

from .embedding import EmbeddingProvider, cosine_similarity
from .memory import MemoryStore
from .models import EventLike, MemoryRecord, SearchResult


class QueryProcessor(Protocol):
    """查询理解接口。

    Day 1 直接返回原问题；后续可接入 Qwen，将“有没有玩手机”等问题
    统一改写为“学生使用手机”。
    """

    def process(self, query: str) -> str:
        """返回用于生成查询向量的规范化文本。"""
        ...


class IdentityQueryProcessor:
    """不改写查询的 Day 1 默认实现。"""

    def process(self, query: str) -> str:
        normalized = query.strip()
        if not normalized:
            raise ValueError("query 不能为空")
        return normalized


class VideoMemoryService:
    """高嘉沐模块对外提供的统一服务。"""

    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingProvider,
        query_processor: QueryProcessor | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._query_processor = query_processor or IdentityQueryProcessor()

    def remember_event(
        self,
        event: EventLike,
        *,
        caption: str | None = None,
        screenshot_path: str = "",
        video_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """接收 A/B 模块事件，生成向量并保存为一条视频记忆。"""

        memory_caption = (caption or event.description or event.event_type).strip()
        embedding_text = f"事件类型：{event.event_type}；事件描述：{memory_caption}"
        embedding = self._embedder.encode(embedding_text)
        record = MemoryRecord.from_event(
            event,
            embedding=embedding,
            embedding_model=self._embedder.model_name,
            caption=memory_caption,
            screenshot_path=screenshot_path,
            video_path=video_path,
            metadata=metadata,
        )
        self._store.add(record)
        return record

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """使用自然语言搜索最相似的视频记忆。"""

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        processed_query = self._query_processor.process(query)
        query_embedding = self._embedder.encode(processed_query)

        results = [
            SearchResult(
                record=record,
                similarity_score=cosine_similarity(query_embedding, record.embedding),
            )
            for record in self._store.list_all()
        ]
        results.sort(key=lambda result: result.similarity_score, reverse=True)
        return [
            result for result in results if result.similarity_score >= min_score
        ][:top_k]

    def get_event(self, event_id: str) -> MemoryRecord | None:
        """根据事件编号读取完整的视频记忆。"""

        return self._store.get(event_id)
