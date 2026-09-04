from src.vlm.inference import (
    infer_activity_from_description,
    infer_activity_from_vlm_meta,
    normalize_vlm_result,
)
from src.vlm.prompt import build_activity_prompt, build_caption_prompt
from src.vlm.qwen_vlm import load_model


def test_caption_prompt_identifies_the_requested_candidate():
    prompt = build_caption_prompt("writing")

    assert "event_type：writing" in prompt
    assert "本次唯一待确认事件" in prompt
    assert "events.writing" in prompt


def test_activity_prompt_treats_candidate_as_correctable_hint():
    prompt = build_activity_prompt("phone_distraction")

    assert "可能完全错误" in prompt
    assert "observed_activities" in prompt
    assert "activity_segments" in prompt


def test_normalization_never_upgrades_model_rejection_from_keywords():
    result = normalize_vlm_result(
        {
            "description": "学生坐下后开始写字。",
            "event_confirmed": False,
            "is_phone_usage": False,
            "is_studying": True,
        },
        event_type="writing",
    )

    assert result["event_confirmed"] is False


def test_description_can_reclassify_reading_candidate_as_writing():
    assert (
        infer_activity_from_description("女孩坐在桌前，用笔在笔记本上写字。")
        == "writing"
    )


def test_structured_events_can_recover_reading_without_description_keyword():
    result = normalize_vlm_result(
        {
            "description": "女孩坐在桌前，手里拿着书。",
            "events": {"reading": True},
            "event_confirmed": False,
            "is_phone_usage": False,
            "is_studying": True,
        },
        event_type="study_preparation",
    )

    assert result["events"]["reading"] is True
    assert infer_activity_from_vlm_meta(result) == "reading"


def test_structured_events_are_filtered_by_description_evidence():
    result = normalize_vlm_result(
        {
            "description": "女孩坐在桌前，手里拿着书，然后拿起手机查看内容。",
            "events": {
                "reading": True,
                "writing": True,
                "computer_learning": True,
                "communication_distraction": True,
            },
            "event_confirmed": False,
            "is_phone_usage": True,
            "is_studying": True,
        },
        event_type="study_preparation",
    )

    assert result["events"]["reading"] is True
    assert result["events"]["writing"] is False
    assert result["events"]["computer_learning"] is False
    assert result["events"]["communication_distraction"] is False


def test_activity_segments_are_normalized_and_ordered():
    result = normalize_vlm_result(
        {
            "description": "女孩先看书，随后拿起手机查看内容。",
            "primary_event": "reading",
            "observed_activities": ["reading", "phone_distraction"],
            "activity_segments": [
                {
                    "event_type": "phone_distraction",
                    "start_frame": 6,
                    "end_frame": 9,
                },
                {"event_type": "reading", "start_frame": 1, "end_frame": 5},
            ],
            "event_confirmed": False,
            "is_phone_usage": True,
            "is_studying": True,
        },
        event_type="other_study_behavior",
    )

    assert [item["event_type"] for item in result["activity_segments"]] == [
        "reading",
        "phone_distraction",
    ]


def test_local_model_path_is_validated_before_loading(tmp_path):
    missing = tmp_path / "missing-model"

    try:
        load_model(model_path=str(missing))
        assert False, "应当拒绝不存在的本地模型目录"
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
