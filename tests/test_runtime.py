import numpy as np
from pathlib import Path

from src.detection.video_source import VideoFrame, VideoSource
from src.event.schemas import Event
from src.pipeline.runtime import EndToEndRunner


class FakeSource(VideoSource):
    def __init__(self):
        self.frames = [
            VideoFrame(
                frame=np.zeros((40, 60, 3), dtype=np.uint8),
                frame_id=index,
                timestamp=timestamp,
                width=60,
                height=40,
            )
            for index, timestamp in enumerate([0.0, 2.0, 3.0, 6.0, 9.0])
        ]

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def release(self):
        pass


class CameraSource(VideoSource):
    def __init__(self):
        self.frames = [
            VideoFrame(
                frame=np.full((48, 64, 3), index % 255, dtype=np.uint8),
                frame_id=index,
                timestamp=index / 10,
                width=64,
                height=48,
            )
            for index in range(120)
        ]

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def release(self):
        pass


def fake_detector(frame):
    if frame.timestamp >= 9.0:
        return []
    return [
        {"class_name": "person", "confidence": 0.95, "bbox": [0, 0, 30, 40]},
        {"class_name": "chair", "confidence": 0.9, "bbox": [1, 1, 31, 39]},
        {"class_name": "book", "confidence": 0.8, "bbox": [10, 10, 20, 20]},
    ]


def fake_tracker(detections, timestamp):
    return [
        {**item, "track_id": index + 1, "timestamp": timestamp}
        for index, item in enumerate(detections)
    ]


def fake_vlm(event, paths):
    event.description = f"VLM确认：{event.event_type}"
    return event, {
        "event_confirmed": True,
        "description": event.description,
        "is_phone_usage": False,
        "is_studying": True,
    }


def test_full_runtime_reaches_memory_and_retrieval(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(),
        detector=fake_detector,
        tracker=fake_tracker,
        event_analyzer=fake_vlm,
        clips_dir=tmp_path,
        buffer_fps=10,
        analysis_fps=10,
    )

    result = runner.run()
    search_results = runner.search("什么时候阅读了？")

    assert result["memories_saved"] == 3
    assert not result["errors"]
    assert search_results
    assert search_results[0]["event_type"] == "reading"


def test_realtime_runner_records_and_creates_event_replay(tmp_path):
    runner = EndToEndRunner(
        source=CameraSource(),
        detector=fake_detector,
        tracker=fake_tracker,
        event_analyzer=fake_vlm,
        clips_dir=tmp_path / "clips",
        recordings_dir=tmp_path / "recordings",
        replays_dir=tmp_path / "replays",
        buffer_fps=5,
        analysis_fps=2,
        recording_segment_seconds=3,
        replay_max_seconds=4,
    )

    result = runner.run()
    records = runner.memory_store.list_all()

    assert result["memories_saved"] >= 1
    assert runner.recorder is not None
    assert len(runner.recorder.segments) >= 3
    assert any(Path(record.video_path).is_file() for record in records)


def test_runtime_reclassifies_rejected_activity_from_vlm_description(tmp_path):
    def rejected_reading(event, paths):
        return None, {
            "event_confirmed": False,
            "description": "女孩坐在书桌前，用笔在笔记本上写字。",
            "is_phone_usage": False,
            "is_studying": True,
        }

    runner = EndToEndRunner(
        source=FakeSource(),
        detector=fake_detector,
        tracker=fake_tracker,
        event_analyzer=rejected_reading,
        clips_dir=tmp_path,
    )
    runner._extract_event_frames = lambda event: ["one.jpg", "two.jpg", "three.jpg"]
    runner._create_replay = lambda event: ""

    runner._handle_event(Event("reading", 10.0, 20.0, 1, 0.9, "学生正在阅读"))

    records = runner.memory_store.list_all()
    assert len(records) == 1
    assert records[0].event_type == "writing"
    assert records[0].metadata["vlm"]["reclassified_from"] == "reading"


def test_runtime_keeps_lifecycle_event_without_sending_it_to_vlm(tmp_path):
    def rejected_sit_event(event, paths):
        raise AssertionError("生命周期事件不应发送给 VLM")

    runner = EndToEndRunner(
        source=FakeSource(),
        detector=fake_detector,
        tracker=fake_tracker,
        event_analyzer=rejected_sit_event,
        clips_dir=tmp_path,
    )
    runner._extract_event_frames = lambda event: ["one.jpg", "two.jpg", "three.jpg"]
    runner._create_replay = lambda event: ""

    runner._handle_event(
        Event("sit_at_study_position", 0.5, 2.5, 1, 0.9, "学生坐到学习位置")
    )

    records = runner.memory_store.list_all()
    assert len(records) == 1
    assert records[0].event_type == "sit_at_study_position"


def test_runtime_reclassifies_generic_candidate_from_structured_vlm_events(tmp_path):
    def rejected_preparation(event, paths):
        return None, {
            "event_confirmed": False,
            "description": "女孩坐在桌前，手里拿着书，然后拿起手机。",
            "is_phone_usage": True,
            "is_studying": True,
            "events": {"reading": True},
        }

    runner = EndToEndRunner(
        source=FakeSource(),
        detector=fake_detector,
        tracker=fake_tracker,
        event_analyzer=rejected_preparation,
        clips_dir=tmp_path,
    )
    runner._extract_event_frames = lambda event: ["one.jpg", "two.jpg", "three.jpg"]
    runner._create_replay = lambda event: ""

    runner._handle_event(
        Event("other_study_behavior", 2.5, 10.0, 1, 0.9, "待 VLM 识别")
    )

    records = runner.memory_store.list_all()
    assert [record.event_type for record in records] == [
        "reading",
        "phone_distraction",
    ]
    assert records[0].start_time == 2.5
    assert records[0].metadata["vlm"]["reclassified_from"] == "other_study_behavior"


def test_runtime_splits_multiple_vlm_activities_in_temporal_order(tmp_path):
    def multiple_activities(event, paths):
        return None, {
            "event_confirmed": False,
            "description": "女孩先看书，随后拿起手机查看内容。",
            "is_phone_usage": True,
            "is_studying": True,
            "events": {"reading": True, "phone_distraction": True},
            "observed_activities": ["reading", "phone_distraction"],
            "activity_segments": [
                {"event_type": "reading", "start_frame": 1, "end_frame": 1},
                {
                    "event_type": "phone_distraction",
                    "start_frame": 2,
                    "end_frame": 3,
                },
            ],
        }

    runner = EndToEndRunner(
        source=FakeSource(),
        detector=fake_detector,
        tracker=fake_tracker,
        event_analyzer=multiple_activities,
        clips_dir=tmp_path,
    )
    runner._extract_event_frames = lambda event: ["one.jpg", "two.jpg", "three.jpg"]
    runner._create_replay = lambda event: ""

    runner._handle_event(
        Event("other_study_behavior", 2.0, 10.0, 1, 0.9, "待 VLM 识别")
    )

    records = runner.memory_store.list_all()
    assert [record.event_type for record in records] == [
        "reading",
        "phone_distraction",
    ]
    assert [(record.start_time, record.end_time) for record in records] == [
        (2.0, 6.0),
        (6.0, 10.0),
    ]
