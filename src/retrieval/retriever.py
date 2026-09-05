"""把 Memory、Embedding 和 Retrieval 串起来的主服务。"""

from __future__ import annotations

import math
from typing import Any, Protocol

from .embedding import EmbeddingProvider, cosine_similarity
from .event_catalog import build_event_embedding_text, match_event_type
from .memory import MemoryStore
from .models import EventLike, MemoryRecord, SearchResult
from ..event.event_types import ALL_EVENTS


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


class EventQueryProcessor:
    """把常见中文问题补充为正式事件名称，再交给 Embedding。

    这里只做轻量、可解释的查询规范化。以后接入 Qwen 查询改写时，仍然只需
    提供同一个 ``process(query) -> str`` 接口，不需要修改检索主流程。
    """

    def process(self, query: str) -> str:
        normalized = query.strip()
        if not normalized:
            raise ValueError("query 不能为空")

        event_type = match_event_type(normalized)
        if event_type is None:
            return normalized
        return build_event_embedding_text(event_type, normalized)


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
        self._query_processor = query_processor or EventQueryProcessor()

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

        original_event_type = event.event_type.strip()
        final_event_type = original_event_type
        if final_event_type not in ALL_EVENTS:
            raise ValueError(f"未知正式事件类型：{original_event_type}")
        memory_caption = (caption or event.description or final_event_type).strip()
        embedding_text = build_event_embedding_text(
            final_event_type,
            memory_caption,
        )
        embedding = self._encode(embedding_text)
        final_metadata = dict(metadata or {})
        record = MemoryRecord.from_event(
            event,
            embedding=embedding,
            embedding_model=self._embedder.model_name,
            caption=memory_caption,
            screenshot_path=screenshot_path,
            video_path=video_path,
            metadata=final_metadata,
        )
        record.event_type = final_event_type
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
        query_embedding = self._encode(processed_query)

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

    def _encode(self, text: str) -> list[float]:
        """校验任意 Embedding 实现的输出，避免错误向量进入检索流程。"""

        try:
            embedding = [float(value) for value in self._embedder.encode(text)]
            expected_dimension = int(self._embedder.dimension)
        except (TypeError, ValueError) as exc:
            raise ValueError("EmbeddingProvider 必须返回数值向量") from exc

        if expected_dimension <= 0:
            raise ValueError("EmbeddingProvider.dimension 必须大于 0")
        if len(embedding) != expected_dimension:
            raise ValueError(
                "EmbeddingProvider 返回向量维度错误："
                f"expected={expected_dimension}, actual={len(embedding)}"
            )
        if not all(math.isfinite(value) for value in embedding):
            raise ValueError("EmbeddingProvider 返回向量必须全部为有限数值")
        return embedding
