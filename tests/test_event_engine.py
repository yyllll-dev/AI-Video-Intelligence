from src.event.engine import EventEngine
from src.event.schemas import TrackingResult


def make_result(
    timestamp,
    track_id,
    class_name,
    confidence=0.95,
):
    return TrackingResult(
        frame_id=int(timestamp * 30),
        timestamp=timestamp,
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        bbox=[100, 100, 300, 500],
    )


def run_frame(engine, timestamp, objects):

    results = [
        make_result(
            timestamp,
            track_id,
            class_name,
        )
        for track_id, class_name in objects
    ]

    events = engine.update(results)

    for event in events:

        print(
            f"[{event.start_time:5.1f}s"
            f" -> {event.end_time:5.1f}s] "
            f"{event.event_type:22s} "
            f"{event.description}"
        )


def main():

    engine = EventEngine()

    print("=" * 70)
    print("Event Engine Test")
    print("=" * 70)

    # =====================================================
    # 1. 学生进入学习区域
    # =====================================================

    run_frame(
        engine,
        0,
        [
            (1, "person"),
        ],
    )

    # =====================================================
    # 2. 学生开始阅读
    # =====================================================

    run_frame(
        engine,
        1,
        [
            (1, "person"),
            (2, "book"),
        ],
    )

    run_frame(
        engine,
        2,
        [
            (1, "person"),
            (2, "book"),
        ],
    )

    run_frame(
        engine,
        3,
        [
            (1, "person"),
            (2, "book"),
        ],
    )

    # =====================================================
    # 3. 学生拿起手机
    # =====================================================

    run_frame(
        engine,
        5,
        [
            (1, "person"),
            (3, "cell phone"),
        ],
    )

    # =====================================================
    # 4. 持续使用手机
    # =====================================================

    run_frame(
        engine,
        6,
        [
            (1, "person"),
            (3, "cell phone"),
        ],
    )

    run_frame(
        engine,
        7,
        [
            (1, "person"),
            (3, "cell phone"),
        ],
    )

    run_frame(
        engine,
        8,
        [
            (1, "person"),
            (3, "cell phone"),
        ],
    )

    run_frame(
        engine,
        9,
        [
            (1, "person"),
            (3, "cell phone"),
        ],
    )

    run_frame(
        engine,
        10,
        [
            (1, "person"),
            (3, "cell phone"),
        ],
    )

    # =====================================================
    # 5. 学生放下手机
    # =====================================================

    run_frame(
        engine,
        12,
        [
            (1, "person"),
        ],
    )

    # =====================================================
    # 6. 回到学习
    # =====================================================

    run_frame(
        engine,
        13,
        [
            (1, "person"),
            (2, "book"),
        ],
    )

    run_frame(
        engine,
        14,
        [
            (1, "person"),
            (2, "book"),
        ],
    )

    # =====================================================
    # 7. 学生开始使用电脑学习
    # =====================================================

    run_frame(
        engine,
        15,
        [
            (1, "person"),
            (4, "laptop"),
        ],
    )

    run_frame(
        engine,
        16,
        [
            (1, "person"),
            (4, "laptop"),
        ],
    )

    # =====================================================
    # 8. 学生离开学习区域
    # =====================================================

    # 17 秒：最后看到学生
    run_frame(
        engine,
        17,
        [
            (1, "person"),
        ],
    )

    # 这里用一个非 person 目标提供时间戳
    # 模拟学生已经离开
    run_frame(
        engine,
        22,
        [
            (99, "chair"),
        ],
    )

    print("=" * 70)
    print("Event Engine Test Finished")
    print("=" * 70)


if __name__ == "__main__":
    main()