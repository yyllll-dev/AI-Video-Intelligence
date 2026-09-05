import json

from src.event.event_types import ALL_EVENTS
from src.vlm.inference import (
    infer_activity_from_description,
    infer_activities_from_vlm_meta,
    normalize_vlm_result,
    parse_json,
)
from src.vlm.prompt import build_activity_prompt, build_caption_prompt


def _events(**confirmed):
    return {name: bool(confirmed.get(name, False)) for name in ALL_EVENTS}


def test_prompt_uses_only_the_eight_root_event_types():
    prompt = build_activity_prompt("computer_usage", frame_count=9)

    assert "不提供预设事件结论" in prompt
    assert "上游状态：computer_usage" not in prompt
    assert "以下八类" in prompt
    assert set(ALL_EVENTS) == {
        "sit_at_study_position",
        "leave_study_position",
        "reading",
        "writing",
        "phone_usage",
        "computer_usage",
        "communication_distraction",
        "other_behavior",
    }
    for removed in (
        "study_preparation",
        "start_study",
        "end_study",
        "study_end_cleanup",
        "phone_learning",
        "phone_distraction",
        "computer_learning",
        "computer_distraction",
        "other_study_behavior",
    ):
        assert removed not in prompt


def test_caption_prompt_routes_to_the_same_three_stage_contract():
    prompt = build_caption_prompt("writing")
    assert "第一步 objective_description" in prompt
    assert "第二步 events" in prompt
    assert "第三步 final_description" in prompt


def test_other_candidate_is_presented_as_unclassified_not_confirmed():
    prompt = build_activity_prompt(
        "other_behavior",
        frame_count=9,
        candidate_scores={"reading": 0.5, "computer_usage": 0.5},
        unclassified_ratio=0.75,
    )
    assert "书本相关物体约出现在50%的分析帧" in prompt
    assert "电脑/键盘/鼠标约出现在50%的分析帧" in prompt
    assert "75%" in prompt
    assert "准备和切换过程" in prompt


def test_explicit_computer_description_repairs_empty_structured_events():
    result = normalize_vlm_result(
        {
            "objective_description": "一位女士坐在桌前使用笔记本电脑。",
            "final_description": "一位女士坐在桌前使用笔记本电脑。",
            "primary_event": "none",
            "observed_activities": [],
            "activity_segments": [],
            "event_confirmed": False,
            "events": _events(),
        },
        event_type="other_behavior",
        frame_count=9,
    )

    assert result["events"]["computer_usage"] is True
    assert result["events"]["other_behavior"] is False
    assert result["primary_event"] == "computer_usage"
    assert "events 与 objective_description 不一致" in "；".join(result["contract_warnings"])


def test_concrete_event_excludes_unsegmented_other():
    result = normalize_vlm_result(
        {
            "objective_description": "人物正在写字。",
            "final_description": "人物正在书写。",
            "primary_event": "writing",
            "events": _events(writing=True, other_behavior=True),
        },
        event_type="other_behavior",
        frame_count=9,
    )
    assert result["events"]["writing"] is True
    assert result["events"]["other_behavior"] is False


def test_other_can_coexist_with_reading_in_separate_segments():
    result = normalize_vlm_result(
        {
            "objective_description": "人物先拿出书并翻到目标页，随后持续阅读。",
            "final_description": "人物先整理书本，随后持续阅读。",
            "primary_event": "reading",
            "observed_activities": ["other_behavior", "reading"],
            "events": _events(other_behavior=True, reading=True),
            "activity_segments": [
                {"event_type": "other_behavior", "start_frame": 1, "end_frame": 4},
                {"event_type": "reading", "start_frame": 5, "end_frame": 9},
            ],
        },
        event_type="other_behavior",
        frame_count=9,
    )

    assert result["events"]["other_behavior"] is True
    assert result["events"]["reading"] is True
    assert infer_activities_from_vlm_meta(result) == ["other_behavior", "reading"]


def test_other_remains_valid_when_no_specific_action_is_available():
    result = normalize_vlm_result(
        {
            "objective_description": "人物整理桌面上的物品。",
            "final_description": "人物整理桌面上的物品，其他。",
            "primary_event": "other_behavior",
            "observed_activities": ["other_behavior"],
            "events": _events(other_behavior=True),
        },
        event_type="other_behavior",
        frame_count=9,
    )
    assert result["events"]["other_behavior"] is True
    assert result["primary_event"] == "other_behavior"


def test_removed_event_names_are_ignored_instead_of_mapped():
    result = normalize_vlm_result(
        {
            "objective_description": "人物坐在桌前。",
            "events": {"study_preparation": True, "computer_learning": True},
            "primary_event": "study_preparation",
        },
        event_type="other_behavior",
        frame_count=9,
    )
    assert set(result["events"]) == set(ALL_EVENTS)
    assert not any(result["events"].values())


def test_activity_segments_are_normalized_and_ordered():
    result = normalize_vlm_result(
        {
            "objective_description": "人物先写字，随后使用手机。",
            "final_description": "人物先书写，随后使用手机。",
            "primary_event": "writing",
            "observed_activities": ["phone_usage", "writing"],
            "events": _events(writing=True, phone_usage=True),
            "activity_segments": [
                {"event_type": "phone_usage", "start_frame": 6, "end_frame": 9},
                {"event_type": "writing", "start_frame": 1, "end_frame": 5},
            ],
        },
        event_type="other_behavior",
        frame_count=9,
    )
    assert [item["event_type"] for item in result["activity_segments"]] == [
        "writing",
        "phone_usage",
    ]
    assert infer_activities_from_vlm_meta(result) == ["writing", "phone_usage"]


def test_out_of_range_segment_is_rejected():
    result = normalize_vlm_result(
        {
            "objective_description": "人物正在写字。",
            "primary_event": "writing",
            "events": _events(writing=True),
            "activity_segments": [
                {"event_type": "writing", "start_frame": 1, "end_frame": 12}
            ],
        },
        event_type="writing",
        frame_count=9,
    )
    assert result["events"]["writing"] is True
    assert result["activity_segments"] == []


def test_description_fallback_handles_notebook_computer_phrasing():
    assert infer_activity_from_description("人物正在使用笔记本电脑。") == "computer_usage"
    assert infer_activity_from_description("人物注视屏幕并操作键盘。") == "computer_usage"


def test_description_fallback_distinguishes_page_turning_context():
    assert infer_activity_from_description("人物拿出书本并翻到上次阅读的位置。") == "other_behavior"
    assert infer_activity_from_description("人物继续阅读书本并短暂翻页。") == "reading"
    assert infer_activity_from_description("人物收好电脑并拿出笔袋。") == "other_behavior"


def test_explicit_transition_description_overrides_unsegmented_concrete_event():
    result = normalize_vlm_result(
        {
            "objective_description": "人物收好电脑并拿出笔袋。",
            "primary_event": "computer_usage",
            "events": _events(computer_usage=True),
            "activity_segments": [],
        },
        event_type="computer_usage",
        frame_count=9,
    )

    assert result["events"]["other_behavior"] is True
    assert result["events"]["computer_usage"] is False


def test_description_fallback_ignores_negated_device_action():
    assert infer_activity_from_description("人物没有使用手机，只是在写字。") == "writing"


def test_phone_event_repairs_summary_boolean():
    result = normalize_vlm_result(
        {
            "objective_description": "人物正在使用手机。",
            "final_description": "人物正在使用手机。",
            "events": _events(phone_usage=True),
            "primary_event": "phone_usage",
            "is_phone_usage": False,
        },
        event_type="phone_usage",
        frame_count=9,
    )
    assert result["is_phone_usage"] is True


def test_parser_extracts_nested_events_json():
    payload = {
        "objective_description": "人物写字。",
        "events": _events(writing=True),
    }
    parsed = parse_json("前缀\n```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```\n后缀")
    assert parsed["events"]["writing"] is True


def test_truncated_json_salvages_objective_description():
    parsed = parse_json('{"objective_description":"人物正在阅读一本书。","events":{')
    assert "阅读一本书" in parsed["objective_description"]
    assert parsed["events"] == {}
