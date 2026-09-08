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
    max_other_bridge_seconds: float = 3.0,
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

    # VLM 完成全部窗口判断后再做时间线去抖。短暂“其他”如果仅夹在
    # 两个同类正式事件之间，就属于这一持续行为并桥接合并。
    # 回退索引使 A-other-A-other-A 也能被连续处理。
    index = 0
    while index + 2 < len(merged):
        left, middle, right = merged[index : index + 3]
        middle_duration = max(
            0.0,
            float(middle["end_time"]) - float(middle["start_time"]),
        )
        same_source = (
            _source_key(left) == _source_key(middle) == _source_key(right)
        )
        continuous = (
            float(middle["start_time"])
            <= float(left["end_time"]) + float(max_gap_seconds)
            and float(right["start_time"])
            <= float(middle["end_time"]) + float(max_gap_seconds)
        )
        bridge = (
            left["event_type"] == right["event_type"]
            and left["event_type"] != "other_behavior"
            and middle["event_type"] == "other_behavior"
            and middle_duration <= float(max_other_bridge_seconds) + 1e-9
            and same_source
            and continuous
        )
        if not bridge:
            index += 1
            continue

        combined = dict(left)
        combined["end_time"] = max(
            float(left["end_time"]), float(right["end_time"])
        )
        combined["confidence"] = max(
            float(left.get("confidence", 0.0)),
            float(right.get("confidence", 0.0)),
        )
        combined["merged_event_count"] = (
            int(left.get("merged_event_count", 1))
            + int(middle.get("merged_event_count", 1))
            + int(right.get("merged_event_count", 1))
        )
        combined["merged_track_ids"] = list(
            dict.fromkeys(
                list(left.get("merged_track_ids", []))
                + list(right.get("merged_track_ids", []))
            )
        )
        if "similarity_score" in left or "similarity_score" in right:
            combined["similarity_score"] = max(
                float(left.get("similarity_score", 0.0)),
                float(right.get("similarity_score", 0.0)),
            )
        merged[index : index + 3] = [combined]
        index = max(0, index - 2)

    return merged
