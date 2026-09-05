from dataclasses import dataclass

import pytest

from src.event.event_types import ALL_EVENTS
from src.event.schemas import Event
from src.retrieval import (
    HashingEmbedder,
    InMemoryStore,
    VideoMemoryService,
    match_event_type,
    merge_events_for_display,
)
from src.retrieval.event_catalog import EVENT_SEARCH_TERMS, SUPPORTED_EVENT_TYPES


def _record(event_type, start, end, track_id=1, source="video-a.mp4"):
    return {
        "event_type": event_type,
        "start_time": start,
        "end_time": end,
        "track_id": track_id,
        "confidence": 0.8,
        "caption": event_type,
        "video_path": source,
        "metadata": {"source_video": source},
    }


def test_event_catalog_exactly_matches_root_event_protocol():
    assert tuple(ALL_EVENTS) == SUPPORTED_EVENT_TYPES
    assert set(EVENT_SEARCH_TERMS) == set(ALL_EVENTS)


def test_common_queries_match_current_event_types():
    assert match_event_type("什么时候在看书？") == "reading"
    assert match_event_type("什么时候使用笔记本电脑？") == "computer_usage"
    assert match_event_type("有没有收拾东西之类的其他行为？") == "other_behavior"


def test_display_merges_contiguous_type_across_tracker_id_changes():
    raw = [
        _record("reading", 12.0, 68.02, track_id=4),
        _record("reading", 68.02, 76.02, track_id=9),
    ]
    displayed = merge_events_for_display(raw)

    assert len(displayed) == 1
    assert displayed[0]["start_time"] == 12.0
    assert displayed[0]["end_time"] == 76.02
    assert displayed[0]["merged_track_ids"] == [4, 9]
    assert len(raw) == 2


def test_display_does_not_bridge_across_a_different_confirmed_event():
    displayed = merge_events_for_display([
        _record("computer_usage", 0.0, 8.0),
        _record("reading", 8.0, 16.0),
        _record("computer_usage", 16.0, 24.0),
    ])
    assert [item["event_type"] for item in displayed] == [
        "computer_usage",
        "reading",
        "computer_usage",
    ]


def test_display_does_not_merge_across_leave_and_resit_boundary():
    displayed = merge_events_for_display([
        _record("reading", 0.0, 8.0),
        _record("leave_study_position", 8.0, 10.0),
        _record("sit_at_study_position", 10.0, 12.0),
        _record("reading", 12.0, 20.0),
    ])
    assert len(displayed) == 4


def test_display_does_not_merge_records_from_different_videos():
    displayed = merge_events_for_display([
        _record("reading", 0.0, 8.0, source="a.mp4"),
        _record("reading", 8.0, 16.0, source="b.mp4"),
    ])
    assert len(displayed) == 2


def test_memory_accepts_each_root_event_without_mapping():
    service = VideoMemoryService(InMemoryStore(), HashingEmbedder())
    for index, event_type in enumerate(ALL_EVENTS):
        record = service.remember_event(
            Event(event_type, index, index + 1, 1, 0.9, event_type)
        )
        assert record.event_type == event_type
        assert "remapped_from" not in record.metadata


def test_memory_rejects_removed_or_unknown_event_names():
    service = VideoMemoryService(InMemoryStore(), HashingEmbedder())
    for removed in ("start_study", "other_study_behavior", "computer_learning"):
        with pytest.raises(ValueError, match="未知正式事件类型"):
            service.remember_event(Event(removed, 0.0, 1.0, 1, 0.9, removed))


@dataclass
class BadEmbedder:
    model_name: str = "bad"
    dimension: int = 3

    def encode(self, text: str):
        return [1.0]


def test_invalid_embedder_output_is_rejected_before_storage():
    store = InMemoryStore()
    service = VideoMemoryService(store, BadEmbedder())
    with pytest.raises(ValueError, match="向量维度错误"):
        service.remember_event(Event("reading", 0.0, 1.0, 1, 0.9, "阅读"))
    assert len(store) == 0
