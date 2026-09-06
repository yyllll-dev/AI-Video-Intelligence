import json
import pytest
import src.vlm.inference as vlm_inference

from src.event.event_types import ALL_EVENTS
from src.vlm.inference import (
    frame_labels_to_segments,
    analyze_window_label,
    infer_activity_from_description,
    infer_activities_from_vlm_meta,
    normalize_vlm_result,
    parse_json,
)
from src.vlm.prompt import (
    build_activity_prompt,
    build_caption_prompt,
    build_frame_labels_prompt,
    build_window_label_prompt,
)


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


def test_window_label_prompt_distinguishes_laptop_from_book_and_writing():
    prompt = build_window_label_prompt()
    assert "不要 JSON" in prompt
    assert "笔记本电脑" in prompt
    assert "不是书" in prompt
    assert "笔尖接触纸面" in prompt
    assert "前后动作不一致" in prompt


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("computer_usage", "computer_usage"),
        ('"writing"', "writing"),
        ("label: reading", "reading"),
    ],
)
def test_window_label_accepts_strict_bare_enum(monkeypatch, raw, expected):
    monkeypatch.setattr(vlm_inference, "run_vlm", lambda **kwargs: raw)
    result = analyze_window_label(object(), object(), ["one.jpg"])
    assert result["valid"] is True
    assert result["label"] == expected


def test_parser_salvages_only_complete_structured_event_labels():
    result = parse_json(
        '{"objective_description":"人物使用笔记本电脑",'
        '"event_labels":["computer_usage"],"activity_segments":['
    )
    assert result["_parse_error"]
    assert result["_structured_event_labels_salvaged"] is True
    assert result["event_labels"] == ["computer_usage"]


def test_parser_does_not_salvage_incomplete_event_labels():
    result = parse_json(
        '{"objective_description":"人物写字", "event_labels":["writing"'
    )
    assert result["_parse_error"]
    assert result["_structured_event_labels_salvaged"] is False
    assert result["event_labels"] == []


def test_caption_prompt_routes_to_the_same_three_stage_contract():
    prompt = build_caption_prompt("writing")
    assert "第一步 objective_description" in prompt
    assert "第二步 events" in prompt
    assert "第三步 activity_segments" in prompt
    assert "禁止描述性别、年龄、外貌、眼镜、服装、颜色、背景" in prompt


def test_boundary_prompt_only_exposes_plausible_labels():
    prompt = build_frame_labels_prompt(
        9, ["reading", "other_behavior", "computer_usage"]
    )
    assert "reading, other_behavior, computer_usage" in prompt
    assert "phone_usage" not in prompt
    assert "不要按标签列表顺序轮流填写" in prompt


def test_normalization_removes_irrelevant_visual_details():
    result = normalize_vlm_result(
        {
            "objective_description": (
                "一位戴眼镜的女性穿着白色衣服，坐在火车车厢内，随后专注写字。"
            ),
            "events": _events(writing=True),
            "primary_event": "writing",
            "activity_segments": [],
        },
        "writing",
        frame_count=9,
    )
    description = result["objective_description"]
    assert "写字" in description
    for irrelevant in ("女性", "眼镜", "白色", "衣服", "火车"):
        assert irrelevant not in description


@pytest.mark.parametrize(
    "description",
    [
        "人物用笔记本电脑工作。",
        "人物在笔记本电脑上工作。",
        "人物用电脑处理任务。",
    ],
)
def test_laptop_work_phrases_are_computer_usage(description):
    assert infer_activity_from_description(description) == "computer_usage"


def test_compact_event_labels_schema_normalizes_to_eight_event_contract():
    result = normalize_vlm_result(
        {
            "objective_description": "人物使用笔记本电脑工作。",
            "event_labels": ["computer_usage"],
            "primary_event": "computer_usage",
            "activity_segments": [],
        },
        "other_behavior",
        frame_count=9,
    )
    assert result["events"]["computer_usage"] is True
    assert sum(result["events"].values()) == 1


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


def test_unsegmented_other_and_concrete_remain_ambiguous_for_boundary_review():
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
    assert result["events"]["other_behavior"] is True


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


def test_static_seated_description_cannot_confirm_sit_transition():
    result = normalize_vlm_result(
        {
            "objective_description": "一个人坐在桌前看书。",
            "events": _events(sit_at_study_position=True),
            "primary_event": "sit_at_study_position",
        },
        event_type="sit_at_study_position",
        frame_count=9,
    )
    assert result["events"]["sit_at_study_position"] is False
    assert result["primary_event"] == "none"


def test_standing_to_sitting_confirms_sit_without_primary_event():
    result = normalize_vlm_result(
        {
            "objective_description": "人物先站着走近书桌，随后坐下。",
            "events": _events(sit_at_study_position=True),
            "primary_event": "none",
        },
        event_type="sit_at_study_position",
        frame_count=9,
    )
    assert result["events"]["sit_at_study_position"] is True


@pytest.mark.parametrize("event_type,word", [("reading", "看书"), ("writing", "写字")])
def test_parse_error_description_never_manufactures_concrete_event(event_type, word):
    parsed = parse_json(f'{{"objective_description":"人物正在{word}。","events":{{')
    result = normalize_vlm_result(parsed, event_type="other_behavior", frame_count=9)
    assert result["parse_error"]
    assert not any(result["events"].values())
    assert result["primary_event"] == "none"


def test_frame_labels_are_merged_into_continuous_segments():
    labels = [
        "computer_usage", "computer_usage", "other_behavior",
        "other_behavior", "other_behavior", "writing", "writing",
        "writing", "writing",
    ]
    assert frame_labels_to_segments(labels, 9) == [
        {"event_type": "computer_usage", "start_frame": 1, "end_frame": 2},
        {"event_type": "other_behavior", "start_frame": 3, "end_frame": 5},
        {"event_type": "writing", "start_frame": 6, "end_frame": 9},
    ]
