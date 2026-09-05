"""面向 UI 的安全事件聚合，不改变底层窗口记录。"""

from __future__ import annotations

from typing import Any, Iterable

from ..event.event_types import ALL_EVENTS


def _source_key(item: dict[str, Any]) -> str:
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("source_video", ""))
    return ""


def merge_events_for_display(
    records: Iterable[Any],
    *,
    max_gap_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    """合并同视频、同类型且真正相邻的事件。

    Tracker ID 是短期检测标识，同一学习者遮挡后可能换 ID，因此它不作为
    展示合并边界。函数只与当前展示序列中的上一条记录合并，确保中间一旦
    出现其他确认事件或入座/离座边界，就不会跨事件桥接。
    """
    normalized = [
        item.to_dict() if hasattr(item, "to_dict") else dict(item)
        for item in records
    ]
    normalized.sort(key=lambda item: (float(item["start_time"]), float(item["end_time"])))

    merged: list[dict[str, Any]] = []
    for item in normalized:
        current = dict(item)
        event_type = str(current.get("event_type", ""))
        if event_type not in ALL_EVENTS:
            continue
        current["event_type"] = event_type
        current["merged_event_count"] = 1
        current["merged_track_ids"] = [current.get("track_id")]

        previous = merged[-1] if merged else None
        continuous = (
            previous is not None
            and previous["event_type"] == current["event_type"]
            and _source_key(previous) == _source_key(current)
            and float(current["start_time"])
            <= float(previous["end_time"]) + float(max_gap_seconds)
        )
        if not continuous:
            merged.append(current)
            continue

        previous["end_time"] = max(
            float(previous["end_time"]), float(current["end_time"])
        )
        previous["confidence"] = max(
            float(previous.get("confidence", 0.0)),
            float(current.get("confidence", 0.0)),
        )
        previous["merged_event_count"] += 1
        if current.get("track_id") not in previous["merged_track_ids"]:
            previous["merged_track_ids"].append(current.get("track_id"))
        if "similarity_score" in current:
            previous["similarity_score"] = max(
                float(previous.get("similarity_score", 0.0)),
                float(current["similarity_score"]),
            )

    return merged
