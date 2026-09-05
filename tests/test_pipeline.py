import numpy as np
import pytest

from src.detection.video_source import VideoFrame
from src.event.event_types import (
    EVENT_LEAVE_STUDY_POSITION,
    EVENT_SIT_AT_STUDY_POSITION,
)
from src.pipeline.pipeline import VideoPipeline


def make_frame(timestamp: float, frame_id: int = 0) -> VideoFrame:
    return VideoFrame(
        frame=np.zeros((20, 20, 3), dtype=np.uint8),
        frame_id=frame_id,
        timestamp=timestamp,
        width=20,
        height=20,
    )


def test_pipeline_uses_tracker_track_ids_and_emits_sit_event():
    def detector(video_frame):
        return [
            {
                "frame_id": video_frame.frame_id,
                "timestamp": video_frame.timestamp,
                "class_name": "person",
                "confidence": 0.9,
                "bbox": [0, 0, 10, 10],
            },
            {
                "frame_id": video_frame.frame_id,
                "timestamp": video_frame.timestamp,
                "class_name": "chair",
                "confidence": 0.8,
                "bbox": [1, 1, 11, 11],
            },
        ]

    def tracker(detections, timestamp):
        assert all("track_id" not in detection for detection in detections)
        return [
            {**detections[0], "track_id": 7, "timestamp": timestamp},
            {**detections[1], "track_id": 8, "timestamp": timestamp},
        ]

    pipeline = VideoPipeline(detector=detector, tracker=tracker)

    first = pipeline.process_frame(make_frame(0.0, 0))
    second = pipeline.process_frame(make_frame(2.0, 60))

    assert len(first["detections"]) == 2
    assert second["tracking_results"][0].track_id == 7
    assert [event.event_type for event in second["events"]] == [
        EVENT_SIT_AT_STUDY_POSITION
    ]


def test_pipeline_advances_absence_with_empty_frame():
    detections = [
        {
            "class_name": "person",
            "confidence": 0.9,
            "bbox": [0, 0, 10, 10],
        },
        {
            "class_name": "chair",
            "confidence": 0.8,
            "bbox": [1, 1, 11, 11],
        },
    ]

    def tracker(items, timestamp):
        if not items:
            return []

        return [
            {**item, "track_id": 7 + index, "timestamp": timestamp}
            for index, item in enumerate(items)
    ]
    pipeline = VideoPipeline(
        detector=lambda _frame: detections,
        tracker=tracker,
    )

    pipeline.process_frame(make_frame(0.0, 0))
    pipeline.process_frame(make_frame(2.0, 60))
    pipeline.detector = lambda _frame: []

    assert pipeline.process_frame(make_frame(4.0, 120))["events"] == []
    assert pipeline.process_frame(make_frame(5.0, 150))["events"] == []
    result = pipeline.process_frame(make_frame(6.0, 180))

    assert [event.event_type for event in result["events"]] == [
        EVENT_LEAVE_STUDY_POSITION
    ]


def test_pipeline_rejects_tracker_output_without_track_id():
    pipeline = VideoPipeline(
        detector=lambda _frame: [
            {
                "class_name": "person",
                "confidence": 0.9,
                "bbox": [0, 0, 10, 10],
            }
        ],
        tracker=lambda detections, _timestamp: detections,
    )

    with pytest.raises(ValueError, match="track_id"):
        pipeline.process_frame(make_frame(0.0))
