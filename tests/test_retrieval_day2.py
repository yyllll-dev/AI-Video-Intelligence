"""高嘉沐 Day 2：真实 Event、14 类事件和可替换 Embedding 的测试。"""

from __future__ import annotations

import unittest

from src.event.event_types import ALL_EVENTS
from src.event.schemas import Event
from src.retrieval import (
    HashingEmbedder,
    InMemoryStore,
    VideoMemoryService,
    merge_events_for_display,
)


EVENT_CASES = (
    ("sit_at_study_position", "学生坐到学习座位", "什么时候坐到座位了？"),
    ("leave_study_position", "学生起身离开座位", "什么时候离开座位？"),
    ("study_preparation", "学生摆放书本并准备文具", "学习准备从什么时候开始？"),
    ("start_study", "学生正式开始学习", "什么时候正式开始学习？"),
    ("end_study", "学生结束本次学习", "这次什么时候结束学习？"),
    ("reading", "学生正在阅读书籍", "什么时候在看书？"),
    ("writing", "学生正在书写作业", "什么时候写作业了？"),
    ("phone_learning", "学生使用手机查询学习资料", "什么时候用手机查学习资料？"),
    ("computer_learning", "学生使用电脑上网课", "什么时候在用电脑学习？"),
    ("other_study_behavior", "学生进行其他学习行为", "有没有其他学习行为？"),
    ("phone_distraction", "学生暂停学习并玩手机", "刚才什么时候玩手机了？"),
    ("computer_distraction", "学生玩电脑游戏导致分心", "什么时候玩电脑游戏了？"),
    ("communication_distraction", "学生与他人聊天导致分心", "什么时候和别人聊天？"),
    ("study_end_cleanup", "学生结束学习后收拾书本", "什么时候开始收拾书本？"),
)


def test_display_merges_continuous_reading_without_changing_raw_events():
    raw = [
        {"event_type": "reading", "start_time": 13.0, "end_time": 25.0,
         "track_id": 1, "confidence": 0.8, "caption": "阅读", "video_path": "a.mp4"},
        {"event_type": "start_study", "start_time": 13.0, "end_time": 16.0,
         "track_id": 1, "confidence": 0.9, "caption": "开始", "video_path": "b.mp4"},
        {"event_type": "reading", "start_time": 25.0, "end_time": 37.0,
         "track_id": 1, "confidence": 0.9, "caption": "阅读", "video_path": "c.mp4"},
        {"event_type": "reading", "start_time": 37.0, "end_time": 61.0,
         "track_id": 1, "confidence": 0.85, "caption": "阅读", "video_path": "d.mp4"},
    ]

    displayed = merge_events_for_display(raw)

    reading = next(item for item in displayed if item["event_type"] == "reading")
    assert len(raw) == 4
    assert reading["start_time"] == 13.0
    assert reading["end_time"] == 61.0
    assert reading["merged_event_count"] == 3
    assert [item["event_type"] for item in displayed] == ["reading"]


def test_display_keeps_only_one_action_for_each_time_interval():
    raw = [
        {"event_type": "sit_at_study_position", "start_time": 0.25,
         "end_time": 2.25, "track_id": 1, "confidence": 0.9},
        {"event_type": "study_preparation", "start_time": 2.25,
         "end_time": 3.25, "track_id": 1, "confidence": 0.8},
        {"event_type": "reading", "start_time": 2.25,
         "end_time": 10.0, "track_id": 1, "confidence": 0.95},
        {"event_type": "start_study", "start_time": 3.25,
         "end_time": 6.25, "track_id": 1, "confidence": 0.8},
        {"event_type": "end_study", "start_time": 3.25,
         "end_time": 10.0, "track_id": 1, "confidence": 0.8},
    ]

    displayed = merge_events_for_display(raw)

    assert [item["event_type"] for item in displayed] == [
        "sit_at_study_position",
        "reading",
    ]
    assert [(item["start_time"], item["end_time"]) for item in displayed] == [
        (0.25, 2.25),
        (2.25, 10.0),
    ]


class FakeProductionEmbedder:
    """模拟后续真实模型，验证替换实现时主流程和调用方式不变。"""

    @property
    def model_name(self) -> str:
        return "fake-production-embedding-v1"

    @property
    def dimension(self) -> int:
        return len(ALL_EVENTS) + 1

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for index, event_type in enumerate(ALL_EVENTS):
            if event_type in text:
                vector[index] = 1.0
                return vector
        vector[-1] = 1.0
        return vector


class WrongDimensionEmbedder:
    @property
    def model_name(self) -> str:
        return "broken-embedding"

    @property
    def dimension(self) -> int:
        return 4

    def encode(self, text: str) -> list[float]:
        return [1.0, 0.0]


class RetrievalDay2Test(unittest.TestCase):
    def test_event_catalog_matches_all_14_official_event_types(self) -> None:
        self.assertEqual(len(ALL_EVENTS), 14)
        self.assertEqual(
            {event_type for event_type, _, _ in EVENT_CASES},
            set(ALL_EVENTS),
        )

    def test_real_event_converts_to_memory_record_without_interface_change(self) -> None:
        store = InMemoryStore()
        service = VideoMemoryService(store, HashingEmbedder(256))
        event = Event(
            event_type="phone_distraction",
            start_time=632.5,
            end_time=640.0,
            track_id=1,
            confidence=0.91,
            description="学生暂停学习并玩手机",
        )

        record = service.remember_event(
            event,
            screenshot_path="data/screenshots/phone.jpg",
            video_path="data/clips/phone.mp4",
        )

        self.assertEqual(record.event_type, event.event_type)
        self.assertEqual(record.timestamp, event.start_time)
        self.assertEqual(record.start_time, event.start_time)
        self.assertEqual(record.end_time, event.end_time)
        self.assertEqual(record.track_id, event.track_id)
        self.assertEqual(record.confidence, event.confidence)
        self.assertEqual(record.caption, event.description)
        self.assertEqual(record.screenshot_path, "data/screenshots/phone.jpg")
        self.assertEqual(record.video_path, "data/clips/phone.mp4")
        self.assertIs(service.get_event(record.event_id), record)

    def test_all_14_events_can_be_saved_and_found_by_natural_language(self) -> None:
        store = InMemoryStore()
        service = VideoMemoryService(store, HashingEmbedder(1024))

        for index, (event_type, caption, _) in enumerate(EVENT_CASES):
            start_time = float(index * 10)
            service.remember_event(
                Event(
                    event_type=event_type,
                    start_time=start_time,
                    end_time=start_time + 5.0,
                    track_id=1,
                    confidence=0.9,
                    description=caption,
                )
            )

        self.assertEqual(len(store), 14)
        for expected_event_type, _, query in EVENT_CASES:
            with self.subTest(query=query):
                result = service.search(query, top_k=1)[0]
                self.assertEqual(result.record.event_type, expected_event_type)

    def test_real_embedder_can_replace_placeholder_without_changing_service_flow(self) -> None:
        store = InMemoryStore()
        service = VideoMemoryService(store, FakeProductionEmbedder())
        service.remember_event(
            Event("reading", 10.0, 20.0, 1, 0.92, "学生正在看书")
        )
        service.remember_event(
            Event(
                "phone_distraction",
                30.0,
                35.0,
                1,
                0.95,
                "学生暂停学习并玩手机",
            )
        )

        result = service.search("刚才什么时候玩手机了？", top_k=1)[0]

        self.assertEqual(result.record.event_type, "phone_distraction")
        self.assertEqual(result.record.embedding_model, "fake-production-embedding-v1")

    def test_invalid_embedder_output_is_rejected_before_storage(self) -> None:
        store = InMemoryStore()
        service = VideoMemoryService(store, WrongDimensionEmbedder())

        with self.assertRaisesRegex(ValueError, "向量维度错误"):
            service.remember_event(
                Event("reading", 1.0, 2.0, 1, 0.9, "学生正在阅读")
            )
        self.assertEqual(len(store), 0)


if __name__ == "__main__":
    unittest.main()
