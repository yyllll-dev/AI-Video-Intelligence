"""Video Memory、Embedding 与 Retrieval 的 Day 1 公共接口。"""

from .embedding import EmbeddingProvider, HashingEmbedder, cosine_similarity
from .event_catalog import (
    EVENT_SEARCH_TERMS,
    SUPPORTED_EVENT_TYPES,
    EventSearchTerms,
    build_event_embedding_text,
    match_event_type,
)
from .memory import InMemoryStore, MemoryStore
from .models import EventLike, MemoryRecord, SearchResult
from .presentation import merge_events_for_display
from .retriever import (
    EventQueryProcessor,
    IdentityQueryProcessor,
    QueryProcessor,
    VideoMemoryService,
)

__all__ = [
    "EmbeddingProvider",
    "EVENT_SEARCH_TERMS",
    "EventLike",
    "EventQueryProcessor",
    "EventSearchTerms",
    "HashingEmbedder",
    "IdentityQueryProcessor",
    "InMemoryStore",
    "MemoryRecord",
    "MemoryStore",
    "QueryProcessor",
    "SearchResult",
    "SUPPORTED_EVENT_TYPES",
    "VideoMemoryService",
    "build_event_embedding_text",
    "cosine_similarity",
    "match_event_type",
    "merge_events_for_display",
]
