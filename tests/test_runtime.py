import numpy as np
import pytest
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

    assert result["memories_saved"] == 2
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


def test_runtime_recovers_explicit_behavior_from_vlm_description(tmp_path):
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
    assert [record.event_type for record in records] == ["writing"]
    assert records[0].metadata["vlm"]["timeline_fallback"] == "vlm_description"
    assert runner.rejected_events == []


def test_runtime_recovers_specific_event_engine_candidate_before_other(tmp_path):
    def unclassified(event, paths):
        return None, {
            "event_confirmed": False,
            "description": "人物低头看向桌面。",
            "events": {},
            "contract_warnings": [],
        }

    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=unclassified, clips_dir=tmp_path,
    )
    runner._extract_event_frames = lambda event: ["one.jpg", "two.jpg", "three.jpg"]
    runner._create_replay = lambda event: ""

    runner._handle_event(Event("reading", 10.0, 18.0, 1, 0.9, "待识别"))

    records = runner.memory_store.list_all()
    assert [record.event_type for record in records] == ["reading"]
    assert records[0].metadata["vlm"]["timeline_fallback"] == "event_engine_candidate"


def test_runtime_uses_other_only_when_no_specific_fallback_evidence(tmp_path):
    def unclassified(event, paths):
        return None, {
            "event_confirmed": False,
            "description": "人物坐在桌前，动作无法明确分类。",
            "events": {},
            "contract_warnings": [],
        }

    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=unclassified, clips_dir=tmp_path,
    )
    runner._extract_event_frames = lambda event: ["one.jpg", "two.jpg", "three.jpg"]
    runner._create_replay = lambda event: ""

    runner._handle_event(
        Event("other_behavior", 18.0, 26.0, 1, 0.9, "待识别")
    )

    records = runner.memory_store.list_all()
    assert [record.event_type for record in records] == ["other_behavior"]
    assert (
        records[0].metadata["vlm"]["timeline_fallback"]
        == "unclassified_seated_window"
    )


def test_compact_window_log_links_keyframe_time_to_nearest_yolo_frame(
    tmp_path, capsys
):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=fake_vlm, clips_dir=tmp_path,
    )
    frame = VideoFrame(
        frame=np.zeros((40, 60, 3), dtype=np.uint8),
        frame_id=120,
        timestamp=4.0,
        width=60,
        height=40,
    )
    runner.latest_result = {
        "detections": [
            {"class_name": "person", "confidence": 0.95},
            {"class_name": "book", "confidence": 0.81},
        ]
    }
    runner._record_analysis_observation(frame)

    runner._print_window_trace(
        Event("reading", 4.0, 12.0, 1, 0.9, "待识别"),
        ["one.jpg", "two.jpg"],
    )

    output = capsys.readouterr().out
    assert "[分析窗口] 4.00s - 12.00s" in output
    assert "Event候选=reading" in output
    assert "[关键帧 1/2] 4.00s" in output
    assert "最近YOLO源帧#120 @4.00s" in output
    assert "person(0.95), book(0.81)" in output


def test_runtime_allows_vlm_to_reclassify_lifecycle_event(tmp_path):
    calls = []

    def reclassified_sit_event(event, paths):
        calls.append(event.event_type)
        return None, {
            "event_confirmed": True,
            "description": "人物坐在桌前阅读书本，阅读。",
            "events": {"sit_at_study_position": False, "reading": True},
            "primary_event": "reading",
            "observed_activities": ["reading"],
            "activity_segments": [
                {"event_type": "reading", "start_frame": 1, "end_frame": 3}
            ],
        }

    runner = EndToEndRunner(
        source=FakeSource(),
        detector=fake_detector,
        tracker=fake_tracker,
        event_analyzer=reclassified_sit_event,
        clips_dir=tmp_path,
    )
    runner._extract_event_frames = lambda event: ["one.jpg", "two.jpg", "three.jpg"]
    runner._create_replay = lambda event: ""

    runner._handle_event(
        Event("sit_at_study_position", 0.5, 2.5, 1, 0.9, "学生坐到学习位置")
    )

    records = runner.memory_store.list_all()
    assert calls == ["sit_at_study_position"]
    assert len(records) == 1
    assert records[0].event_type == "reading"


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
        Event("other_behavior", 2.5, 10.0, 1, 0.9, "待 VLM 识别")
    )

    records = runner.memory_store.list_all()
    assert [record.event_type for record in records] == ["reading"]
    assert records[0].start_time == 2.5
    assert records[0].metadata["vlm"]["reclassified_from"] == "other_behavior"


def test_runtime_splits_multiple_vlm_activities_in_temporal_order(tmp_path):
    def multiple_activities(event, paths):
        return None, {
            "event_confirmed": False,
            "description": "女孩先看书，随后拿起手机查看内容。",
            "is_phone_usage": True,
            "is_studying": True,
            "events": {"reading": True, "phone_usage": True},
            "observed_activities": ["reading", "phone_usage"],
            "activity_segments": [
                {"event_type": "reading", "start_frame": 1, "end_frame": 1},
                {
                    "event_type": "phone_usage",
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
        Event("other_behavior", 2.0, 10.0, 1, 0.9, "待 VLM 识别")
    )

    records = runner.memory_store.list_all()
    assert [record.event_type for record in records] == ["reading", "phone_usage"]
    assert records[0].start_time == pytest.approx(2.0)
    assert records[0].end_time == pytest.approx(2.0 + 8.0 / 3.0)
    assert records[1].start_time == pytest.approx(2.0 + 8.0 / 3.0)
    assert records[1].end_time == pytest.approx(10.0)


def test_runtime_preserves_repeated_activity_segments(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=fake_vlm, clips_dir=tmp_path,
    )
    events = runner._split_activity_event(
        Event("other_behavior", 0.0, 9.0, 1, 0.9, "待识别"),
        ["phone_usage", "reading"],
        "先看手机，再看书，最后继续看手机。",
        activity_segments=[
            {"event_type": "phone_usage", "start_frame": 1, "end_frame": 2},
            {"event_type": "reading", "start_frame": 3, "end_frame": 6},
            {"event_type": "phone_usage", "start_frame": 7, "end_frame": 9},
        ],
        frame_count=9,
        primary_activity="reading",
    )

    assert [event.event_type for event in events] == [
        "phone_usage", "reading", "phone_usage"
    ]
    assert [(event.start_time, event.end_time) for event in events] == [
        (0.0, 2.0), (2.0, 6.0), (6.0, 9.0)
    ]


def _temporal_meta(event_type, objective_description):
    return {
        "event_confirmed": True,
        "objective_description": objective_description,
        "final_description": objective_description,
        "description": objective_description,
        "events": {event_type: True},
        "primary_event": event_type,
        "observed_activities": [event_type],
        "activity_segments": [
            {"event_type": event_type, "start_frame": 1, "end_frame": 9}
        ],
    }


def _temporal_runner(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=fake_vlm, clips_dir=tmp_path,
    )
    runner._create_replay = lambda event: ""
    return runner


def test_temporal_smoothing_repairs_computer_reading_computer_glitch(tmp_path):
    runner = _temporal_runner(tmp_path)
    paths = ["one.jpg"]
    runner._stage_or_remember_event(
        Event("computer_usage", 0.0, 8.0, 1, 0.9, "使用电脑"),
        paths,
        _temporal_meta("computer_usage", "人物坐在电脑前。"),
    )
    runner._stage_or_remember_event(
        Event("reading", 8.0, 16.0, 1, 0.9, "阅读"),
        paths,
        _temporal_meta("reading", "人物坐在桌前，桌上放着书和电脑。"),
    )
    assert [record.event_type for record in runner.memory_store.list_all()] == [
        "computer_usage"
    ]

    runner._stage_or_remember_event(
        Event("computer_usage", 16.0, 24.0, 1, 0.9, "使用电脑"),
        paths,
        _temporal_meta("computer_usage", "人物继续操作电脑键盘。"),
    )

    records = runner.memory_store.list_all()
    assert [record.event_type for record in records] == [
        "computer_usage", "computer_usage", "computer_usage"
    ]
    assert records[1].metadata["vlm"]["temporal_correction"]["from"] == "reading"


def test_temporal_smoothing_uses_logical_subject_across_tracker_id_changes(tmp_path):
    runner = _temporal_runner(tmp_path)
    paths = ["one.jpg"]
    runner._stage_or_remember_event(
        Event("computer_usage", 0.0, 8.0, 3, 0.9, "使用电脑"),
        paths,
        _temporal_meta("computer_usage", "人物操作电脑。"),
    )
    runner._stage_or_remember_event(
        Event("reading", 8.0, 16.0, 19, 0.9, "阅读"),
        paths,
        _temporal_meta("reading", "人物低头看向桌面材料。"),
    )
    runner._stage_or_remember_event(
        Event("computer_usage", 16.0, 24.0, 27, 0.9, "使用电脑"),
        paths,
        _temporal_meta("computer_usage", "人物继续操作电脑。"),
    )

    assert [record.event_type for record in runner.memory_store.list_all()] == [
        "computer_usage", "computer_usage", "computer_usage"
    ]


def test_temporal_smoothing_keeps_explicit_reading_switch(tmp_path):
    runner = _temporal_runner(tmp_path)
    paths = ["one.jpg"]
    runner._stage_or_remember_event(
        Event("computer_usage", 0.0, 8.0, 1, 0.9, "使用电脑"),
        paths,
        _temporal_meta("computer_usage", "人物操作电脑键盘。"),
    )
    runner._stage_or_remember_event(
        Event("reading", 8.0, 16.0, 1, 0.9, "阅读"),
        paths,
        _temporal_meta("reading", "人物拿起书持续看书并翻页。"),
    )

    assert [record.event_type for record in runner.memory_store.list_all()] == [
        "computer_usage", "reading"
    ]


def test_temporal_smoothing_does_not_confirm_two_labels_without_action_evidence(tmp_path):
    runner = _temporal_runner(tmp_path)
    paths = ["one.jpg"]
    runner._stage_or_remember_event(
        Event("computer_usage", 0.0, 8.0, 1, 0.9, "使用电脑"),
        paths,
        _temporal_meta("computer_usage", "人物坐在电脑前。"),
    )
    for start in (8.0, 16.0):
        runner._stage_or_remember_event(
            Event("reading", start, start + 8.0, 1, 0.9, "阅读"),
            paths,
            _temporal_meta("reading", "人物低头看向桌面材料。"),
        )

    assert [record.event_type for record in runner.memory_store.list_all()] == [
        "computer_usage", "computer_usage"
    ]
    assert runner._pending_behavior_by_track[0].event.event_type == "reading"


def test_short_unresolved_tail_becomes_other(tmp_path):
    runner = _temporal_runner(tmp_path)
    paths = ["one.jpg"]
    runner._stage_or_remember_event(
        Event("writing", 132.0, 148.0, 1, 0.9, "写字"),
        paths,
        _temporal_meta("writing", "人物持续写字。"),
    )
    runner._stage_or_remember_event(
        Event("reading", 148.0, 149.37, 1, 0.9, "阅读"),
        paths,
        _temporal_meta("reading", "人物坐在桌边，书本仍在画面中。"),
    )

    runner._flush_temporal_pending(end_of_stream=True)

    records = runner.memory_store.list_all()
    assert [record.event_type for record in records] == ["writing", "other_behavior"]


def test_transition_other_breaks_previous_computer_immediately(tmp_path):
    runner = _temporal_runner(tmp_path)
    paths = ["one.jpg"]
    runner._stage_or_remember_event(
        Event("computer_usage", 76.0, 108.0, 1, 0.9, "使用电脑"),
        paths,
        _temporal_meta("computer_usage", "人物持续操作电脑键盘。"),
    )
    runner._stage_or_remember_event(
        Event("other_behavior", 108.0, 116.0, 1, 0.9, "其他"),
        paths,
        _temporal_meta("other_behavior", "人物收好电脑并拿出笔袋。"),
    )

    assert [record.event_type for record in runner.memory_store.list_all()] == [
        "computer_usage", "other_behavior"
    ]


def test_runtime_only_falls_back_to_primary_when_segments_are_missing(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=fake_vlm, clips_dir=tmp_path,
    )

    events = runner._split_activity_event(
        Event("other_behavior", 2.0, 8.0, 1, 0.9, "待识别"),
        ["reading", "writing"],
        "人物先阅读后书写，阅读，书写。",
        primary_activity="reading",
    )

    assert [event.event_type for event in events] == ["reading"]
    assert [(event.start_time, event.end_time) for event in events] == [
        (2.0, 8.0)
    ]


def test_runtime_does_not_fallback_to_unsegmented_transition_reclassification(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=fake_vlm, clips_dir=tmp_path,
    )

    events = runner._split_activity_event(
        Event("other_behavior", 2.0, 8.0, 1, 0.9, "待识别"),
        ["start_study", "end_study", "writing"],
        "人物正在写字。",
        primary_activity="writing",
    )

    assert [event.event_type for event in events] == ["writing"]


def test_runtime_keeps_matching_transition_candidate_without_segment(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=fake_vlm, clips_dir=tmp_path,
    )

    events = runner._split_activity_event(
        Event("sit_at_study_position", 2.0, 4.0, 1, 0.9, "人物入座"),
        ["sit_at_study_position", "start_study"],
        "人物走到桌前并坐下。",
        primary_activity="sit_at_study_position",
    )

    assert [event.event_type for event in events] == ["sit_at_study_position"]


def test_runtime_does_not_duplicate_sit_with_unsegmented_preparation(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=fake_vlm, clips_dir=tmp_path,
    )

    events = runner._split_activity_event(
        Event("sit_at_study_position", 1.6, 3.73, 1, 0.9, "人物入座"),
        ["study_preparation", "sit_at_study_position"],
        "人物坐下并拿取本子。",
        primary_activity="study_preparation",
    )

    assert [event.event_type for event in events] == ["sit_at_study_position"]


def test_runtime_does_not_map_removed_cleanup_event(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=fake_vlm, clips_dir=tmp_path,
    )

    events = runner._split_activity_event(
        Event("other_behavior", 10.0, 12.0, 1, 0.9, "待识别"),
        ["study_end_cleanup"],
        "人物收拾书本。",
        primary_activity="study_end_cleanup",
    )

    assert events == []
