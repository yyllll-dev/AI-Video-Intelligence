import numpy as np
import pytest
from pathlib import Path

from src.detection.video_source import VideoFrame, VideoSource
from src.event.schemas import Event
from src.pipeline.runtime import EndToEndRunner
from src.vlm.inference import frame_labels_to_segments


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
        "objective_description": event.description,
        "description": event.description,
        "is_phone_usage": False,
        "is_studying": True,
        "events": {event.event_type: True},
        "activity_segments": [],
    }


def test_full_runtime_reaches_memory_and_retrieval(tmp_path):
    summarized_records = []

    def summarize(records):
        summarized_records.extend(records)
        return "人物先进入学习位置，随后完成阅读，最后离开座位。"

    runner = EndToEndRunner(
        source=FakeSource(),
        detector=fake_detector,
        tracker=fake_tracker,
        event_analyzer=fake_vlm,
        clips_dir=tmp_path,
        buffer_fps=10,
        analysis_fps=10,
        summary_analyzer=summarize,
    )

    result = runner.run()
    search_results = runner.search("什么时候阅读了？")

    assert result["memories_saved"] == 3
    assert not result["errors"]
    assert search_results
    assert search_results[0]["event_type"] == "reading"
    assert result["video_summary"] == "人物先进入学习位置，随后完成阅读，最后离开座位。"
    assert len(summarized_records) == result["memories_saved"]


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


def test_runtime_does_not_recover_behavior_from_failed_vlm_description(tmp_path):
    def rejected_reading(event, paths):
        return None, {
            "event_confirmed": False,
            "description": "女孩坐在书桌前，用笔在笔记本上写字。",
            "is_phone_usage": False,
            "is_studying": True,
            "parse_error": "broken json",
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
    assert [record.event_type for record in records] == ["other_behavior"]
    assert records[0].metadata["vlm"]["timeline_fallback"] == "parse_error_boundary_other"
    assert runner.rejected_events == []


def test_runtime_does_not_promote_event_engine_object_candidate(tmp_path):
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
    assert [record.event_type for record in records] == ["other_behavior"]


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


def test_static_lifecycle_candidate_is_rejected_not_reclassified_to_reading(tmp_path):
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
    assert records == ()
    assert len(runner.rejected_events) == 1


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
    assert records[0].end_time == pytest.approx(6.0)
    assert records[1].start_time == pytest.approx(6.0)
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
        (0.0, 2.25), (2.25, 6.75), (6.75, 9.0)
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


def test_primary_cannot_fill_multi_event_window_without_segments(tmp_path):
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

    assert events == []


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


def _boundary_result(labels):
    return {
        "valid": True,
        "frame_labels": labels,
        "activity_segments": frame_labels_to_segments(labels, len(labels)),
    }


def test_opening_frame_labels_find_sit_then_preparation_boundary(tmp_path):
    labels = ["sit_at_study_position"] * 4 + ["other_behavior"] * 5
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=lambda event, paths: (None, {
            "event_confirmed": False, "events": {},
            "description": "人物动作处于变化中。",
        }),
        boundary_analyzer=lambda paths: _boundary_result(labels),
        clips_dir=tmp_path,
    )
    runner._create_replay = lambda event: ""
    paths = [f"{index}.jpg" for index in range(9)]
    runner._handle_event(Event("other_behavior", 4.0, 12.0, 1, 0.9), paths)

    records = runner.memory_store.list_all()
    assert [record.event_type for record in records] == [
        "sit_at_study_position", "other_behavior"
    ]
    assert records[0].end_time == pytest.approx(8.0)
    assert records[1].start_time == pytest.approx(8.0)


def test_frame_boundary_can_locate_writing_at_126_seconds(tmp_path):
    runner = _temporal_runner(tmp_path)
    labels = ["other_behavior", "other_behavior"] + ["writing"] * 7
    events = runner._split_activity_event(
        Event("other_behavior", 124.0, 132.0, 1, 0.9),
        ["other_behavior", "writing"],
        "人物先整理用品，随后写字。",
        activity_segments=frame_labels_to_segments(labels, 9),
        frame_count=9,
    )
    assert [(item.event_type, item.start_time, item.end_time) for item in events] == [
        ("other_behavior", 124.0, 126.0),
        ("writing", 126.0, 132.0),
    ]


def test_boundary_window_does_not_extend_previous_stable_on_parse_error(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=lambda event, paths: (None, {
            "event_confirmed": False,
            "events": {},
            "description": "人物收起电脑并拿出书本。",
            "objective_description": "人物收起电脑并拿出书本。",
            "parse_error": "broken json",
        }),
        boundary_analyzer=lambda paths: _boundary_result(["other_behavior"] * 9),
        clips_dir=tmp_path,
    )
    runner._create_replay = lambda event: ""
    runner._stable_behavior_by_track[0] = "computer_usage"
    runner._handle_event(
        Event("computer_usage", 108.0, 112.0, 1, 0.9),
        [f"{index}.jpg" for index in range(9)],
    )
    assert [record.event_type for record in runner.memory_store.list_all()] == [
        "other_behavior"
    ]


def test_final_short_window_continues_matching_stable_event(tmp_path):
    runner = _temporal_runner(tmp_path)
    paths = ["one.jpg"]
    runner._stage_or_remember_event(
        Event("writing", 132.0, 148.03, 1, 0.9, "持续写字"),
        paths,
        _temporal_meta("writing", "人物持续写字。"),
    )
    runner._stage_or_remember_event(
        Event(
            "writing", 148.03, 149.37, 1, 0.9, "收尾",
            is_final_window=True,
        ),
        paths,
        _temporal_meta("writing", "人物仍坐在桌前。"),
    )
    assert [record.event_type for record in runner.memory_store.list_all()] == [
        "writing", "writing"
    ]


def test_stable_reading_with_sparse_yolo_does_not_trigger_boundary_review(tmp_path):
    runner = _temporal_runner(tmp_path)
    runner._stable_behavior_by_track[0] = "reading"
    event = Event(
        "other_behavior", 20.0, 28.0, 1, 0.9,
        candidate_scores={"reading": 0.1}, unclassified_ratio=0.9,
    )
    meta = {
        "events": {"reading": True},
        "objective_description": "人物持续阅读书本。",
        "activity_segments": [],
    }
    assert runner._should_refine_window(event, ["x"] * 9, meta) is False


def test_clean_single_reading_is_not_refined_only_because_stable_is_other(tmp_path):
    runner = _temporal_runner(tmp_path)
    runner._stable_behavior_by_track[0] = "other_behavior"
    event = Event(
        "reading", 20.0, 28.0, 1, 0.9,
        candidate_scores={"reading": 0.1}, unclassified_ratio=0.9,
    )
    meta = {
        "events": {"reading": True},
        "objective_description": "人物持续阅读书本。",
        "activity_segments": [],
    }
    assert runner._should_refine_window(event, ["x"] * 9, meta) is False


def test_strong_computer_candidate_forces_review_of_main_reading_at_68s(tmp_path):
    runner = _temporal_runner(tmp_path)
    runner._stable_behavior_by_track[0] = "reading"
    event = Event(
        "computer_usage", 68.0, 76.0, 1, 0.9,
        candidate_scores={"computer_usage": 0.3, "reading": 0.07},
        unclassified_ratio=0.7,
    )
    meta = {
        "events": {"reading": True},
        "objective_description": "人物坐着看书。",
        "activity_segments": [],
    }
    assert runner._should_refine_window(event, ["x"] * 9, meta) is True


def test_valid_boundary_other_immediately_breaks_previous_reading(tmp_path):
    runner = _temporal_runner(tmp_path)
    runner._stable_behavior_by_track[0] = "reading"
    meta = _temporal_meta("other_behavior", "人物收起书本并拿出电脑。")
    meta["boundary_labels_valid"] = True
    runner._stage_or_remember_event(
        Event("other_behavior", 68.0, 76.0, 1, 0.9, "切换用品"),
        ["one.jpg"],
        meta,
    )
    assert [item.event_type for item in runner.memory_store.list_all()] == [
        "other_behavior"
    ]
    assert runner._stable_behavior_by_track[0] == "other_behavior"


def test_pathological_boundary_cycle_cannot_restore_conflicting_main_reading(tmp_path):
    labels = [
        "reading", "writing", "computer_usage", "phone_usage",
        "other_behavior", "other_behavior", "other_behavior",
        "other_behavior", "other_behavior",
    ]
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=lambda event, paths: (event, {
            "event_confirmed": True,
            "events": {"reading": True},
            "objective_description": "人物持续阅读书本。",
            "description": "人物持续阅读书本。",
            "activity_segments": [],
        }),
        boundary_analyzer=lambda paths: _boundary_result(labels),
        clips_dir=tmp_path,
    )
    runner._create_replay = lambda event: ""
    runner._stable_behavior_by_track[0] = "other_behavior"
    runner._handle_event(
        Event(
            "reading", 20.0, 28.0, 1, 0.9,
            candidate_scores={"computer_usage": 0.3},
        ),
        [f"{index}.jpg" for index in range(9)],
    )
    assert [item.event_type for item in runner.memory_store.list_all()] == [
        "other_behavior"
    ]


def test_parse_error_description_can_veto_but_not_create_event(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=lambda event, paths: (None, {
            "event_confirmed": False,
            "events": {},
            "objective_description": "人物持续看书。",
            "description": "人物持续看书。",
            "parse_error": "broken json",
            "activity_segments": [],
        }),
        clips_dir=tmp_path,
    )
    runner._create_replay = lambda event: ""
    runner._stable_behavior_by_track[0] = "computer_usage"
    runner._handle_event(
        Event(
            "reading", 116.0, 124.0, 1, 0.9,
            candidate_scores={"reading": 0.5},
        ),
        [f"{index}.jpg" for index in range(9)],
    )
    assert [item.event_type for item in runner.memory_store.list_all()] == [
        "other_behavior"
    ]


def test_short_final_tail_can_continue_verified_stable_fallback(tmp_path):
    runner = _temporal_runner(tmp_path)
    paths = ["one.jpg"]
    runner._stage_or_remember_event(
        Event("writing", 132.0, 148.0, 1, 0.9, "持续写字"),
        paths,
        _temporal_meta("writing", "人物持续写字。"),
    )
    meta = _temporal_meta("writing", "人物仍在写字。")
    meta["timeline_fallback"] = "previous_stable_event"
    runner._stage_or_remember_event(
        Event(
            "writing", 148.0, 149.37, 1, 0.9, "仍在写字",
            is_final_window=True,
        ),
        paths,
        meta,
    )
    assert [item.event_type for item in runner.memory_store.list_all()] == [
        "writing", "writing"
    ]


def test_dedicated_position_review_recovers_real_sitting_transition(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=lambda event, paths: (None, {
            "event_confirmed": False,
            "events": {},
            "objective_description": "人物坐着休息。",
            "description": "人物坐着休息。",
            "activity_segments": [],
        }),
        position_analyzer=lambda paths, event_type: {
            "valid": True,
            "confirmed": True,
            "position_change": "standing_to_sitting",
        },
        clips_dir=tmp_path,
    )
    runner._create_replay = lambda event: ""
    runner._handle_event(
        Event("sit_at_study_position", 0.5, 4.3, 1, 0.9),
        [f"{index}.jpg" for index in range(9)],
    )
    assert [item.event_type for item in runner.memory_store.list_all()] == [
        "sit_at_study_position"
    ]
    assert runner.vlm_position_calls == 1


@pytest.mark.parametrize(
    ("description", "label"),
    [
        ("人物用笔记本电脑工作。", "computer_usage"),
        ("人物持续落笔写字。", "writing"),
    ],
)
def test_parse_error_uses_restricted_visual_review_to_recover_concrete_event(
    tmp_path, description, label
):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=lambda event, paths: (None, {
            "event_confirmed": False,
            "events": {},
            "objective_description": description,
            "description": description,
            "parse_error": "broken json",
            "activity_segments": [],
        }),
        boundary_analyzer=lambda paths: _boundary_result([label] * 9),
        window_label_analyzer=lambda paths: {
            "valid": True,
            "label": label,
            "parse_error": "",
        },
        clips_dir=tmp_path,
    )
    runner._create_replay = lambda event: ""
    runner._stable_behavior_by_track[0] = "other_behavior"
    runner._handle_event(
        Event("other_behavior", 100.0, 108.0, 1, 0.9),
        [f"{index}.jpg" for index in range(9)],
    )
    assert [item.event_type for item in runner.memory_store.list_all()] == [label]
    assert runner.vlm_boundary_calls == 0
    assert runner.vlm_window_label_calls == 1


def test_stable_windows_use_one_main_vlm_call_and_no_boundary_call(tmp_path):
    def stable_reading(event, paths):
        return event, {
            "event_confirmed": True,
            "events": {"reading": True},
            "objective_description": "人物持续阅读书本。",
            "description": "人物持续阅读书本。",
            "activity_segments": [],
        }
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=stable_reading,
        boundary_analyzer=lambda paths: pytest.fail("稳定窗口不应边界复核"),
        clips_dir=tmp_path,
    )
    runner._create_replay = lambda event: ""
    for start in (20.0, 28.0):
        runner._handle_event(
            Event("reading", start, start + 8.0, 1, 0.9),
            [f"{index}.jpg" for index in range(9)],
        )
    assert runner.vlm_main_calls == 2
    assert runner.vlm_boundary_calls == 0


def test_invalid_boundary_review_immediately_breaks_sticky_reading(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=lambda event, paths: (event, {
            "event_confirmed": True,
            "events": {"reading": True},
            "objective_description": "人物坐着看书。",
            "description": "人物坐着看书。",
            "activity_segments": [],
        }),
        boundary_analyzer=lambda paths: _boundary_result([
            "other_behavior", "reading", "computer_usage", "reading",
            "reading", "reading", "reading", "reading", "reading",
        ]),
        clips_dir=tmp_path,
    )
    runner._create_replay = lambda event: ""
    runner._stable_behavior_by_track[0] = "reading"
    runner._handle_event(
        Event(
            "computer_usage", 68.0, 76.0, 1, 0.9,
            candidate_scores={"computer_usage": 0.3, "reading": 0.07},
        ),
        [f"{index}.jpg" for index in range(9)],
    )
    assert [item.event_type for item in runner.memory_store.list_all()] == [
        "other_behavior"
    ]
    assert runner._stable_behavior_by_track[0] == "other_behavior"


def test_whole_window_visual_label_can_correct_wrong_salvaged_description(tmp_path):
    runner = EndToEndRunner(
        source=FakeSource(), detector=fake_detector, tracker=fake_tracker,
        event_analyzer=lambda event, paths: (None, {
            "event_confirmed": False,
            "events": {},
            "objective_description": "人物看书。",
            "description": "人物看书。",
            "parse_error": "broken json",
            "activity_segments": [],
        }),
        window_label_analyzer=lambda paths: {
            "valid": True, "label": "writing", "parse_error": "",
        },
        clips_dir=tmp_path,
    )
    runner._create_replay = lambda event: ""
    runner._stable_behavior_by_track[0] = "reading"
    runner._handle_event(
        Event("reading", 132.0, 140.0, 1, 0.9),
        [f"{index}.jpg" for index in range(9)],
    )
    assert [item.event_type for item in runner.memory_store.list_all()] == ["writing"]
    assert runner._stable_behavior_by_track[0] == "writing"


def test_putting_book_away_is_transition_evidence_not_reading():
    from src.vlm.inference import description_has_transition_evidence

    assert description_has_transition_evidence("人物拿起书，随后将书放在桌子下面。")
