from src.pipeline.runtime import EndToEndRunner
from src.vlm.inference import (
    description_has_transition_evidence,
    infer_activity_from_description,
    normalize_vlm_result,
)
from demo.app import status_html
from src.retrieval import merge_events_for_display


def _record(event_type: str, start: float, end: float) -> dict:
    return {
        "event_type": event_type,
        "start_time": start,
        "end_time": end,
        "caption": "",
    }


def test_summary_collapses_consecutive_duplicate_events():
    records = [
        _record("reading", 0.0, 4.0),
        _record("reading", 4.0, 8.0),
        _record("reading", 8.0, 12.0),
        _record("writing", 12.0, 16.0),
    ]

    merged = EndToEndRunner._summary_records(records)
    summary = EndToEndRunner._fallback_video_summary(records)

    assert [item["event_type"] for item in merged] == ["reading", "writing"]
    assert merged[0]["start_time"] == 0.0
    assert merged[0]["end_time"] == 12.0
    assert summary.count("阅读") == 1


def test_summary_collapses_reading_runs_separated_only_by_other():
    records = [
        _record("reading", 0.0, 4.0),
        _record("other_behavior", 4.0, 5.0),
        _record("reading", 5.0, 9.0),
        _record("other_behavior", 9.0, 10.0),
        _record("reading", 10.0, 14.0),
    ]

    summary = EndToEndRunner._fallback_video_summary(records)

    assert summary.count("阅读") == 1
    assert summary.count("准备、整理或动作切换") == 1


def test_repeated_generated_action_is_rejected_after_timeline_merge():
    records = [
        _record("reading", 0.0, 4.0),
        _record("reading", 4.0, 8.0),
    ]

    assert not EndToEndRunner._is_objective_video_summary(
        "人物先阅读，随后继续阅读。",
        records,
    )


def test_normal_page_turn_is_reading_not_transition():
    description = "人物翻到下一页继续查看书中内容。"

    assert infer_activity_from_description(description) == "reading"
    assert not description_has_transition_evidence(description)


def test_fanyue_books_is_reading_not_transition():
    description = "人物在桌子上翻阅书籍。"

    assert infer_activity_from_description(description) == "reading"
    assert not description_has_transition_evidence(description)


def test_page_search_remains_other_behavior():
    description = "人物寻找页码并翻到目标页。"

    assert infer_activity_from_description(description) == "other_behavior"
    assert description_has_transition_evidence(description)


def test_valid_json_other_page_turn_is_corrected_to_reading():
    parsed = {
        "objective_description": "人物翻动书页并继续查看内容。",
        "events": {"other_behavior": True},
        "primary_event": "other_behavior",
        "observed_activities": ["other_behavior"],
        "activity_segments": [
            {"event_type": "other_behavior", "start_frame": 1, "end_frame": 9}
        ],
    }

    result = normalize_vlm_result(parsed, "other_behavior", frame_count=9)

    assert result["events"]["reading"] is True
    assert result["events"]["other_behavior"] is False
    assert result["primary_event"] == "reading"
    assert result["activity_segments"] == [
        {"event_type": "reading", "start_frame": 1, "end_frame": 9}
    ]


def test_recording_wording_is_camera_only():
    upload_loading = status_html("loading")
    camera_loading = status_html("loading_camera")

    assert "正在加载模型" in upload_loading
    assert "录制" not in upload_loading
    assert "尚未开始录制" in camera_loading


def test_short_other_between_same_events_is_removed_and_merged():
    records = [
        _record("reading", 0.0, 5.0),
        _record("other_behavior", 5.0, 8.0),
        _record("reading", 8.0, 12.0),
    ]

    merged = merge_events_for_display(records)

    assert len(merged) == 1
    assert merged[0]["event_type"] == "reading"
    assert merged[0]["start_time"] == 0.0
    assert merged[0]["end_time"] == 12.0


def test_repeated_short_other_bridges_are_removed_iteratively():
    records = [
        _record("reading", 0.0, 5.0),
        _record("other_behavior", 5.0, 7.0),
        _record("reading", 7.0, 11.0),
        _record("other_behavior", 11.0, 14.0),
        _record("reading", 14.0, 20.0),
    ]

    merged = merge_events_for_display(records)

    assert len(merged) == 1
    assert merged[0]["event_type"] == "reading"
    assert merged[0]["start_time"] == 0.0
    assert merged[0]["end_time"] == 20.0


def test_other_longer_than_three_seconds_is_preserved():
    records = [
        _record("reading", 0.0, 5.0),
        _record("other_behavior", 5.0, 8.01),
        _record("reading", 8.01, 12.0),
    ]

    merged = merge_events_for_display(records)

    assert [item["event_type"] for item in merged] == [
        "reading",
        "other_behavior",
        "reading",
    ]
