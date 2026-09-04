from src.tracking.tracker import SimpleTracker


def detection(class_name, timestamp, frame_id, bbox):
    return {
        "class_name": class_name,
        "confidence": 0.9,
        "bbox": bbox,
        "timestamp": timestamp,
        "frame_id": frame_id,
    }


def test_tracker_keeps_person_identity_without_emitting_stale_detection():
    tracker = SimpleTracker(max_missed_frames=2)
    first = tracker.update(
        [detection("person", 0.0, 0, [0, 0, 30, 40])],
        timestamp=0.0,
    )

    second = tracker.update(
        [detection("book", 0.5, 1, [10, 10, 20, 20])],
        timestamp=0.5,
    )

    assert all(item.class_name != "person" for item in second)

    third = tracker.update(
        [detection("person", 1.0, 2, [0, 0, 30, 40])],
        timestamp=1.0,
    )
    assert third[0].track_id == first[0].track_id


def test_tracker_drops_person_after_configured_gap():
    tracker = SimpleTracker(max_missed_frames=1)
    tracker.update(
        [detection("person", 0.0, 0, [0, 0, 30, 40])],
        timestamp=0.0,
    )

    tracker.update(
        [detection("book", 0.5, 1, [10, 10, 20, 20])],
        timestamp=0.5,
    )
    last = tracker.update(
        [detection("book", 1.0, 2, [10, 10, 20, 20])],
        timestamp=1.0,
    )

    assert all(item.class_name != "person" for item in last)
