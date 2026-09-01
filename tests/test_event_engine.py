from src.event.engine import EventEngine
from src.event.event_types import (
    EVENT_LEAVE_STUDY_POSITION,
    EVENT_READING,
    EVENT_SIT_AT_STUDY_POSITION,
    EVENT_START_STUDY,
    EVENT_STUDY_PREPARATION,
)
from src.event.schemas import TrackingResult


def make_result(timestamp, track_id, class_name, bbox=None, confidence=0.95):
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
        make_result(timestamp, 1, "person", [100, 100, 300, 500]),
        make_result(timestamp, 2, "chair", [120, 240, 320, 560]),
    ]
    objects.extend(extra_objects or [])
    return objects


def event_types(events):
    return [event.event_type for event in events]


def test_emits_sit_at_study_position_after_duration():
    engine = EventEngine()

    assert engine.update(study_position_frame(0.0), timestamp=0.0) == []
    events = engine.update(study_position_frame(2.0), timestamp=2.0)

    assert event_types(events) == [EVENT_SIT_AT_STUDY_POSITION]
    assert events[0].track_id == 1
    assert events[0].start_time == 0.0
    assert events[0].end_time == 2.0


def test_emits_study_flow_and_reading_when_activity_changes():
    engine = EventEngine()

    engine.update(study_position_frame(0.0), timestamp=0.0)
    engine.update(study_position_frame(2.0), timestamp=2.0)

    events = engine.update(
        study_position_frame(
            3.0,
            [make_result(3.0, 3, "book", [330, 250, 420, 330])],
        ),
        timestamp=3.0,
    )
    assert event_types(events) == [EVENT_STUDY_PREPARATION]

    events = engine.update(
        study_position_frame(
            6.0,
            [make_result(6.0, 3, "book", [330, 250, 420, 330])],
        ),
        timestamp=6.0,
    )
    assert event_types(events) == [EVENT_START_STUDY]

    events = engine.update(
        study_position_frame(
            7.0,
            [make_result(7.0, 4, "pen", [330, 250, 360, 280])],
        ),
        timestamp=7.0,
    )
    assert event_types(events) == [EVENT_READING]


def test_emits_leave_study_position_after_absence_duration():
    engine = EventEngine()

    engine.update(study_position_frame(0.0), timestamp=0.0)
    engine.update(study_position_frame(2.0), timestamp=2.0)

    assert engine.update([], timestamp=3.0) == []
    events = engine.update([], timestamp=4.0)

    assert event_types(events) == [EVENT_LEAVE_STUDY_POSITION]
    assert events[0].track_id == 1