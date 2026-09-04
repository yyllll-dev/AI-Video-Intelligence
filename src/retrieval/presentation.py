"""面向 UI 的事件聚合，不改变检测、事件和 VLM 原始结果。"""

from __future__ import annotations

from typing import Any, Iterable


_ACTIVITY_TYPES = {
    "reading",
    "writing",
    "phone_learning",
    "computer_learning",
    "other_study_behavior",
    "phone_distraction",
    "computer_distraction",
    "communication_distraction",
}

_PROCESS_TYPES = {
    "study_preparation",
    "start_study",
    "end_study",
    "study_end_cleanup",
}


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        float(left["start_time"]) < float(right["end_time"])
        and float(right["start_time"]) < float(left["end_time"])
    )


def merge_events_for_display(
    records: Iterable[Any],
    *,
    max_gap_seconds: float = 2.5,
) -> list[dict[str, Any]]:
    """合并同类型、同人物且时间连续的记忆，仅供页面展示。"""
    normalized = [
        item.to_dict() if hasattr(item, "to_dict") else dict(item)
        for item in records
    ]
    activities = [item for item in normalized if item["event_type"] in _ACTIVITY_TYPES]
    normalized = [
        item
        for item in normalized
        if not (
            item["event_type"] in _PROCESS_TYPES
            and any(
                item.get("track_id") == activity.get("track_id")
                and _overlaps(item, activity)
                for activity in activities
            )
        )
    ]
    normalized.sort(key=lambda item: (float(item["start_time"]), float(item["end_time"])))

    exclusive: list[dict[str, Any]] = []
    for item in normalized:
        current = dict(item)
        if exclusive and float(current["start_time"]) < float(exclusive[-1]["end_time"]):
            previous = exclusive[-1]
            previous_is_activity = previous["event_type"] in _ACTIVITY_TYPES
            current_is_activity = current["event_type"] in _ACTIVITY_TYPES
            if current_is_activity and not previous_is_activity:
                previous["end_time"] = float(current["start_time"])
                if float(previous["end_time"]) <= float(previous["start_time"]):
                    exclusive.pop()
            else:
                current["start_time"] = float(previous["end_time"])
        if float(current["end_time"]) > float(current["start_time"]):
            exclusive.append(current)

    merged: list[dict[str, Any]] = []
    for item in exclusive:
        current = dict(item)
        current["merged_event_count"] = 1
        previous = merged[-1] if merged else None
        continuous = (
            previous is not None
            and previous["event_type"] == current["event_type"]
            and previous.get("track_id") == current.get("track_id")
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
        if "similarity_score" in current:
            previous["similarity_score"] = max(
                float(previous.get("similarity_score", 0.0)),
                float(current["similarity_score"]),
            )

    return merged
