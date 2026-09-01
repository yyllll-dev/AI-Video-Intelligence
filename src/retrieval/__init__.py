"""Video Memory、Embedding 与 Retrieval 的 Day 1 公共接口。"""

from .embedding import EmbeddingProvider, HashingEmbedder, cosine_similarity
from .memory import InMemoryStore, MemoryStore
from .models import EventLike, MemoryRecord, SearchResult
from .retriever import IdentityQueryProcessor, QueryProcessor, VideoMemoryService

__all__ = [
    "EmbeddingProvider",
    "EventLike",
    "HashingEmbedder",
    "IdentityQueryProcessor",
    "InMemoryStore",
    "MemoryRecord",
    "MemoryStore",
    "QueryProcessor",
    "SearchResult",
    "VideoMemoryService",
    "cosine_similarity",
]
