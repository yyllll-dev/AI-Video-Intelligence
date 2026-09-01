"""视频记忆存储层接口与 Day 1 内存实现。"""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from .models import MemoryRecord


class MemoryStore(Protocol):
    """所有记忆存储实现都必须遵守的接口。

    Day 1 使用 InMemoryStore；Day 2 可以新增 SQLiteMemoryStore，
    只要实现相同方法，Retriever 无需修改。
    """

    def add(self, record: MemoryRecord) -> None:
        """保存一条视频记忆。"""
        ...

    def get(self, event_id: str) -> MemoryRecord | None:
        """根据 event_id 读取一条记忆。"""
        ...

    def list_all(self) -> tuple[MemoryRecord, ...]:
        """返回当前全部记忆。"""
        ...

    def __len__(self) -> int:
        """返回记忆条数。"""
        ...


class InMemoryStore:
    """Day 1 用于验证接口的内存存储。

    它不会写硬盘，程序退出后数据会消失。持久化数据库属于 Day 2。
    """

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._lock = RLock()

    def add(self, record: MemoryRecord) -> None:
        with self._lock:
            if record.event_id in self._records:
                raise ValueError(f"event_id 已存在: {record.event_id}")
            self._records[record.event_id] = record

    def get(self, event_id: str) -> MemoryRecord | None:
        with self._lock:
            return self._records.get(event_id)

    def list_all(self) -> tuple[MemoryRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
