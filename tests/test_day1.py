"""高嘉沐 Day 1 框架的自动化测试。"""

from dataclasses import dataclass
import unittest

from src.retrieval import HashingEmbedder, InMemoryStore, VideoMemoryService


@dataclass
class TeamEvent:
    """模拟队长提供的 Event，验证接口可以直接对接。"""

    event_type: str
    start_time: float
    end_time: float
    track_id: int | None
    confidence: float
    description: str = ""


class Day1FrameworkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryStore()
        self.service = VideoMemoryService(self.store, HashingEmbedder(256))

    def test_team_event_can_be_saved(self) -> None:
        event = TeamEvent(
            event_type="phone_distraction",
            start_time=632.5,
            end_time=640.0,
            track_id=1,
            confidence=0.91,
            description="学生暂停学习并使用手机",
        )
        record = self.service.remember_event(
            event,
            screenshot_path="data/screenshots/phone.jpg",
            video_path="data/clips/phone.mp4",
        )

        self.assertEqual(len(self.store), 1)
        self.assertEqual(record.event_type, "phone_distraction")
        self.assertEqual(record.timestamp, 632.5)
        self.assertEqual(record.caption, event.description)
        self.assertEqual(record.video_path, "data/clips/phone.mp4")
        self.assertEqual(len(record.embedding), 256)

    def test_query_finds_phone_distraction(self) -> None:
        events = [
            TeamEvent("start_study", 12.0, 18.0, 1, 0.94, "学生坐下并开始学习"),
            TeamEvent("writing", 120.5, 180.0, 1, 0.89, "学生拿起笔书写"),
            TeamEvent("phone_distraction", 632.5, 640.0, 1, 0.91, "学生暂停学习并使用手机"),
        ]
        for event in events:
            self.service.remember_event(event)

        result = self.service.search("什么时候使用了手机？", top_k=1)[0]
        self.assertEqual(result.record.event_type, "phone_distraction")
        self.assertEqual(result.record.timestamp, 632.5)

    def test_invalid_event_time_is_rejected(self) -> None:
        event = TeamEvent("phone_distraction", 20.0, 10.0, 1, 0.91, "学生使用手机")
        with self.assertRaisesRegex(ValueError, "end_time"):
            self.service.remember_event(event)

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "query"):
            self.service.search("   ")


if __name__ == "__main__":
    unittest.main()
