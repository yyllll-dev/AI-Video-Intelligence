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

    events = engine.update(
        [],
        timestamp=4.0,
    )

    assert event_types(events) == [
        EVENT_LEAVE_STUDY_POSITION,
    ]

    assert events[0].track_id == 1


def test_emits_generic_candidate_without_detected_study_object():
    engine = EventEngine(semantic_window_duration=2.0)
    engine.update(study_position_frame(0.0), timestamp=0.0)
    engine.update(study_position_frame(2.0), timestamp=2.0)
    engine.update(study_position_frame(3.0), timestamp=3.0)

    events = engine.update(study_position_frame(4.0), timestamp=4.0)

    assert event_types(events) == ["other_study_behavior"]
    assert events[0].start_time == 2.0
    assert events[0].end_time == 4.0


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
