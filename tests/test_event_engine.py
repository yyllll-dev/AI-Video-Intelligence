from src.event.engine import EventEngine
from src.event.event_types import (
    EVENT_LEAVE_STUDY_POSITION,
    EVENT_READING,
    EVENT_SIT_AT_STUDY_POSITION,
)
from src.event.schemas import TrackingResult


def make_result(
    timestamp,
    track_id,
    class_name,
    bbox=None,
    confidence=0.95,
):
    return TrackingResult(
        frame_id=int(timestamp * 30),
        timestamp=timestamp,
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox or [100, 100, 300, 500],
    )


def study_position_frame(timestamp, extra_objects=None):
    objects = [
        make_result(
            timestamp,
            1,
            "person",
            [100, 100, 300, 500],
        ),
        make_result(
            timestamp,
            2,
            "chair",
            [120, 240, 320, 560],
        ),
    ]

    objects.extend(extra_objects or [])
    return objects


def event_types(events):
    return [event.event_type for event in events]


def test_emits_sit_at_study_position_after_duration():
    engine = EventEngine()

    assert engine.update(
        study_position_frame(0.0),
        timestamp=0.0,
    ) == []

    events = engine.update(
        study_position_frame(2.0),
        timestamp=2.0,
    )

    assert event_types(events) == [
        EVENT_SIT_AT_STUDY_POSITION,
    ]

    assert events[0].track_id == 1
    assert events[0].start_time == 0.0
    assert events[0].end_time == 2.0


def test_emits_study_flow_and_reading_when_activity_changes():
    engine = EventEngine(semantic_window_duration=5.0)

    engine.update(
        study_position_frame(0.0),
        timestamp=0.0,
    )

    engine.update(
        study_position_frame(2.0),
        timestamp=2.0,
    )

    # 第一次检测到 book：内部进入学习准备，但不输出重叠的流程事件。
    events = engine.update(
        study_position_frame(
            3.0,
            [
                make_result(
                    3.0,
                    3,
                    "book",
                    [330, 250, 420, 330],
                )
            ],
        ),
        timestamp=3.0,
    )

    assert events == []

    # 持续检测到 book，内部达到稳定学习状态，仍只累计实际动作。
    events = engine.update(
        study_position_frame(
            6.0,
            [
                make_result(
                    6.0,
                    3,
                    "book",
                    [330, 250, 420, 330],
                )
            ],
        ),
        timestamp=6.0,
    )

    assert events == []

    # 语义观察窗口到期后，输出该窗口内票数最多的宽松候选。
    events = engine.update(
        study_position_frame(
            7.0,
            [
                make_result(
                    7.0,
                    4,
                    "pen",
                    [330, 250, 360, 280],
                )
            ],
        ),
        timestamp=7.0,
    )

    assert event_types(events) == [
        EVENT_READING,
    ]

    assert events[0].start_time == 2.0
    assert events[0].end_time == 7.0
    assert events[0].track_id == 1


def test_emits_leave_study_position_after_absence_duration():
    engine = EventEngine()

    engine.update(
        study_position_frame(0.0),
        timestamp=0.0,
    )

    engine.update(
        study_position_frame(2.0),
        timestamp=2.0,
    )

    assert engine.update(
        [],
        timestamp=3.0,
    ) == []

    assert engine.update([], timestamp=4.0) == []

    events = engine.update([], timestamp=5.0)

    assert event_types(events) == [
        EVENT_LEAVE_STUDY_POSITION,
    ]

    assert events[0].track_id == 1


def test_missing_furniture_is_unknown_and_does_not_reset_session():
    engine = EventEngine(semantic_window_duration=10.0)
    engine.update(study_position_frame(0.0), timestamp=0.0)
    engine.update(study_position_frame(2.0), timestamp=2.0)
    engine.update(study_position_frame(3.0), timestamp=3.0)

    person_only = [make_result(4.0, 1, "person", [100, 100, 300, 500])]
    assert engine.update(person_only, timestamp=4.0) == []
    person_only = [make_result(12.0, 1, "person", [100, 100, 300, 500])]
    events = engine.update(person_only, timestamp=12.0)

    assert event_types(events) == ["other_behavior"]
    assert engine.state == "studying"
    assert engine.leave_start_time is None


def test_stable_person_can_start_session_when_furniture_is_never_detected():
    engine = EventEngine()
    person_only = lambda timestamp: [
        make_result(timestamp, 1, "person", [100, 100, 300, 500])
    ]

    assert engine.update(person_only(0.0), timestamp=0.0) == []
    assert engine.update(person_only(3.0), timestamp=3.0) == []
    events = engine.update(person_only(4.0), timestamp=4.0)

    assert event_types(events) == [EVENT_SIT_AT_STUDY_POSITION]
    assert events[0].start_time == 0.0


def test_short_person_occlusion_preserves_activity_context():
    engine = EventEngine(semantic_window_duration=6.0)
    engine.update(study_position_frame(0.0), timestamp=0.0)
    engine.update(study_position_frame(2.0), timestamp=2.0)
    engine.update(study_position_frame(3.0), timestamp=3.0)

    assert engine.update([], timestamp=4.0) == []
    events = engine.update(study_position_frame(5.0), timestamp=5.0)
    assert events == []
    assert engine.activity_start_time == 2.0
    assert engine.activity_votes == {}
    assert engine.activity_unclassified_count == 3


def test_emits_generic_candidate_without_detected_study_object():
    engine = EventEngine(semantic_window_duration=2.0)
    engine.update(study_position_frame(0.0), timestamp=0.0)
    engine.update(study_position_frame(2.0), timestamp=2.0)
    engine.update(study_position_frame(3.0), timestamp=3.0)

    events = engine.update(study_position_frame(4.0), timestamp=4.0)

    assert event_types(events) == ["other_behavior"]
    assert events[0].start_time == 2.0
    assert events[0].end_time == 4.0
    assert events[0].candidate_scores == {}
    assert events[0].unclassified_ratio == 1.0


def test_object_candidates_are_collected_without_priority():
    evidence = EventEngine._basic_activity_from_objects([
        make_result(1.0, 1, "book", [10, 10, 30, 30]),
        make_result(1.0, 2, "laptop", [40, 10, 100, 70]),
        make_result(1.0, 3, "cell phone", [110, 10, 130, 40]),
    ])

    assert evidence == {
        "reading": 1.0,
        "phone_usage": 1.0,
        "computer_usage": 1.0,
    }


def test_person_only_frames_do_not_outvote_intermittent_laptop_evidence():
    engine = EventEngine(semantic_window_duration=4.0)
    engine.activity_start_time = 0.0
    person = make_result(4.0, 1, "person", [100, 100, 300, 500])

    engine._handle_activity({}, person, 1.0)
    engine._handle_activity({"computer_usage": 1.0}, person, 2.0)
    engine._handle_activity({}, person, 3.0)
    events = engine._handle_activity({}, person, 4.0)

    assert event_types(events) == ["computer_usage"]
    assert events[0].candidate_scores == {"computer_usage": 0.25}
    assert events[0].unclassified_ratio == 0.75


def test_keyboard_and_mouse_are_computer_candidates():
    evidence = EventEngine._basic_activity_from_objects([
        make_result(1.0, 2, "keyboard", [40, 10, 100, 70]),
        make_result(1.0, 3, "mouse", [110, 10, 130, 40]),
    ])
    assert evidence == {"computer_usage": 1.0}


def test_equal_window_votes_emit_neutral_candidate_with_all_scores():
    engine = EventEngine(semantic_window_duration=2.0)
    engine.activity_start_time = 0.0
    engine.activity_votes = {"reading": 3.0, "computer_usage": 3.0}
    engine.activity_observation_count = 6

    event = engine._finish_activity_window(2.0)

    assert event is not None
    assert event.event_type == "other_behavior"
    assert event.candidate_scores == {"reading": 0.5, "computer_usage": 0.5}


def test_finalize_closes_activity_without_fabricating_end_study():
    engine = EventEngine(semantic_window_duration=30.0)
    engine.update(study_position_frame(0.0), timestamp=0.0)
    engine.update(
        study_position_frame(
            2.0,
            [make_result(2.0, 3, "book", [330, 250, 420, 330])],
        ),
        timestamp=2.0,
    )
    engine.update(
        study_position_frame(
            3.0,
            [make_result(3.0, 3, "book", [330, 250, 420, 330])],
        ),
        timestamp=3.0,
    )

    events = engine.finalize(10.0)

    assert event_types(events) == [EVENT_READING]
    assert events[0].start_time == 2.0
    assert events[0].end_time == 10.0


def test_finalize_does_not_extend_activity_through_unconfirmed_absence():
    engine = EventEngine(semantic_window_duration=30.0)
    engine.update(study_position_frame(0.0), timestamp=0.0)
    engine.update(study_position_frame(2.0), timestamp=2.0)
    engine.update(study_position_frame(3.0), timestamp=3.0)
    engine.update([], timestamp=4.0)

    events = engine.finalize(4.0)

    assert event_types(events) == ["other_behavior"]
    assert events[0].end_time == 3.0


def test_rebinds_person_after_tracker_changes_id():
    engine = EventEngine()
    engine.update(study_position_frame(0.0), timestamp=0.0)
    engine.update(study_position_frame(2.0), timestamp=2.0)

    rebound = [
        make_result(3.0, 99, "person", [100, 100, 300, 500]),
        make_result(3.0, 2, "chair", [120, 240, 320, 560]),
    ]
    engine.update(rebound, timestamp=3.0)

    assert engine.person_track_id == 99


def test_does_not_rebind_active_session_to_distant_bystander():
    engine = EventEngine()
    engine.update(study_position_frame(0.0), timestamp=0.0)
    engine.update(study_position_frame(2.0), timestamp=2.0)

    bystander = [make_result(3.0, 99, "person", [900, 100, 1100, 500])]
    engine.update(bystander, timestamp=3.0)

    assert engine.person_track_id == 1
    assert engine.leave_start_time == 2.0
