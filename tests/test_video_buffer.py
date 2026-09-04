"""C/潇睿 模块 —— VideoBuffer 环形缓冲最小测试

分工：杨潇睿 —— VideoBuffer（实时视频 → 历史画面 → VLM 的桥梁）
本文件：tests/test_video_buffer.py

运行方式（在仓库根目录）：
    - python -m pytest tests/test_video_buffer.py -v   # pytest 方式
    - python tests/test_video_buffer.py                 # 直接运行全部断言

覆盖点：
    1. add_frame 只保留最近 buffer_duration 秒（旧帧自动淘汰）。
    2. get_frames 在事件时间窗口内均匀抽取关键帧并保存为 jpg。
    3. 返回图片路径按时间顺序排列、文件真实存在。
    4. 窗口内帧不足时有多少返回多少；窗口为空时返回空列表。
    5. 非法输入抛出清晰错误。
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# 保证从仓库根目录可直接导入 src 包（项目尚未安装为包）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.pipeline.video_buffer import (  # noqa: E402
    VideoBuffer,
    create_video_buffer,
)


# ============ 工具：生成测试帧 ============

def _make_frame(width=32, height=24, brightness=100):
    """生成一帧纯色 BGR 图像。"""
    return np.full((height, width, 3), brightness, dtype=np.uint8)


def _count_images(directory):
    """统计目录下保存的 jpg 数量。"""
    return len(list(Path(directory).glob("*.jpg")))


# ============ 环形缓冲：只保留最近 N 秒 ============

def test_add_frame_keeps_only_recent_duration():
    """add_frame 后只保留最近 buffer_duration 秒内的帧。"""
    buffer = VideoBuffer(buffer_duration=5.0)

    # 1 秒 1 帧，从 0 秒喂到 10 秒，共 11 帧
    for timestamp in range(11):
        buffer.add_frame(
            frame=_make_frame(),
            timestamp=float(timestamp),
        )

    # 最新时间 10.0，保留窗口 [10 - 5, 10] = [5, 10]
    # 即保留 5.0 ~ 10.0 共 6 帧，更早的帧全部被淘汰
    assert len(buffer) == 6
    assert buffer.earliest_timestamp == 5.0
    assert buffer.latest_timestamp == 10.0


def test_add_frame_accepts_single_frame():
    """单帧写入后缓冲可用。"""
    buffer = VideoBuffer(buffer_duration=10.0)
    buffer.add_frame(
        frame=_make_frame(),
        timestamp=3.3,
    )
    assert len(buffer) == 1
    assert buffer.earliest_timestamp == 3.3


# ============ get_frames：时间窗口内均匀抽帧 ============

def test_get_frames_returns_uniform_keyframes_in_window():
    """事件窗口内均匀抽取 num_frames 帧并保存为 jpg。"""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = os.path.join(tmp, "event_001")
        buffer = VideoBuffer(buffer_duration=20.0)

        # 0 ~ 10 秒，每 0.1 秒一帧，共 101 帧
        for index in range(101):
            timestamp = index * 0.1
            buffer.add_frame(
                frame=_make_frame(brightness=int(timestamp * 20)),
                timestamp=timestamp,
            )

        paths = buffer.get_frames(
            start_time=2.0,
            end_time=8.0,
            num_frames=5,
            output_dir=output_dir,
        )

        # 恰好 5 张、文件存在、编号从 frame_001 开始
        assert len(paths) == 5
        assert _count_images(output_dir) == 5
        assert Path(output_dir, "frame_001.jpg").exists()
        assert Path(output_dir, "frame_005.jpg").exists()

        # 图片路径指向 output_dir 目录，文件名 frame_001 ~ frame_005
        # （output_dir 在项目根之外时返回绝对路径）
        for path in paths:
            assert Path(path).exists()
            assert Path(path).parent == Path(output_dir)
        names = [Path(path).name for path in paths]
        assert names == [
            f"frame_{index:03d}.jpg"
            for index in range(1, 6)
        ]


def test_get_frames_returns_all_when_window_has_few_frames():
    """窗口内帧不足 num_frames 时，有多少返回多少。"""
    with tempfile.TemporaryDirectory() as tmp:
        buffer = VideoBuffer(buffer_duration=20.0)
        for timestamp in (2.0, 2.5, 3.0):
            buffer.add_frame(
                frame=_make_frame(),
                timestamp=timestamp,
            )

        paths = buffer.get_frames(
            start_time=2.0,
            end_time=3.0,
            num_frames=5,
            output_dir=os.path.join(tmp, "event_short"),
        )

        # 窗口内只有 3 帧，应全部返回
        assert len(paths) == 3
        assert len(set(paths)) == 3


def test_get_frames_returns_empty_when_no_frame_in_window():
    """窗口内没有帧时返回空列表（不抛错）。"""
    buffer = VideoBuffer(buffer_duration=10.0)
    buffer.add_frame(
        frame=_make_frame(),
        timestamp=2.0,
    )

    paths = buffer.get_frames(
        start_time=5.0,
        end_time=6.0,
        num_frames=3,
    )

    assert paths == []


def test_get_frames_never_returns_duplicate_frames():
    """同一帧不会被重复选中。"""
    with tempfile.TemporaryDirectory() as tmp:
        buffer = VideoBuffer(buffer_duration=20.0)
        for index in range(101):
            timestamp = index * 0.1
            buffer.add_frame(
                frame=_make_frame(brightness=int(timestamp)),
                timestamp=timestamp,
            )

        paths = buffer.get_frames(
            start_time=0.0,
            end_time=10.0,
            num_frames=5,
            output_dir=os.path.join(tmp, "event_no_dup"),
        )

        assert len(paths) == 5
        assert len(set(paths)) == 5


# ============ 非法输入报错 ============

def test_add_frame_rejects_non_array_frame():
    """frame 不是 numpy 数组时应抛出 TypeError。"""
    buffer = VideoBuffer(buffer_duration=10.0)
    try:
        buffer.add_frame(
            frame="not an image",
            timestamp=1.0,
        )
        assert False, "应当抛出 TypeError"
    except TypeError:
        pass


def test_add_frame_rejects_invalid_timestamp():
    """timestamp 非法时应抛出 TypeError / ValueError。"""
    buffer = VideoBuffer(buffer_duration=10.0)

    # 非数字
    try:
        buffer.add_frame(
            frame=_make_frame(),
            timestamp="abc",
        )
        assert False, "应当抛出 TypeError"
    except TypeError:
        pass

    # 负数
    try:
        buffer.add_frame(
            frame=_make_frame(),
            timestamp=-1.0,
        )
        assert False, "应当抛出 ValueError"
    except ValueError:
        pass


def test_get_frames_rejects_invalid_arguments():
    """get_frames 参数非法时应抛出清晰错误。"""
    buffer = VideoBuffer(buffer_duration=10.0)

    # 时间窗口颠倒
    try:
        buffer.get_frames(
            start_time=10.0,
            end_time=5.0,
        )
        assert False, "应当抛出 ValueError"
    except ValueError:
        pass

    # num_frames 小于 1
    try:
        buffer.get_frames(
            start_time=0.0,
            end_time=1.0,
            num_frames=0,
        )
        assert False, "应当抛出 ValueError"
    except ValueError:
        pass


# ============ 统一创建入口 ============

def test_create_video_buffer_factory():
    """create_video_buffer 返回可用实例。"""
    buffer = create_video_buffer(
        buffer_duration=3.0,
    )
    assert isinstance(buffer, VideoBuffer)
    assert buffer.buffer_duration == 3.0


# ============ 健全性：并发与边界 ============

def test_video_buffer_thread_safe_concurrent_write_and_query():
    """采集线程写入 + 查询线程读取同时进行，不应崩溃或竞态。"""
    import threading

    with tempfile.TemporaryDirectory() as tmp:
        buffer = VideoBuffer(
            buffer_duration=2.0,
            clips_dir=os.path.join(tmp, "clips"),
        )
        errors = []

        def writer():
            try:
                for index in range(600):
                    buffer.add_frame(
                        frame=_make_frame(),
                        timestamp=index * 0.01,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"writer: {exc!r}")

        def reader():
            try:
                for index in range(200):
                    _ = len(buffer)
                    _ = buffer.earliest_timestamp
                    _ = buffer.latest_timestamp
                    # 每隔若干次发起一次真实窗口查询
                    if index % 20 == 0:
                        buffer.get_frames(
                            start_time=0.0,
                            end_time=100.0,
                            num_frames=3,
                            output_dir=os.path.join(tmp, "concurrent"),
                        )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"reader: {exc!r}")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"并发访问出现异常: {errors}"
        assert len(buffer) > 0


def test_get_frames_single_point_window_and_one_frame():
    """单点窗口(start==end)与 num_frames=1 都应正常工作。"""
    with tempfile.TemporaryDirectory() as tmp:
        buffer = VideoBuffer(buffer_duration=20.0)
        for index in range(101):
            buffer.add_frame(
                frame=_make_frame(brightness=index),
                timestamp=index * 0.1,
            )

        point_paths = buffer.get_frames(
            start_time=5.0,
            end_time=5.0,
            num_frames=1,
            output_dir=os.path.join(tmp, "point"),
        )
        assert len(point_paths) == 1

        single_paths = buffer.get_frames(
            start_time=1.0,
            end_time=9.0,
            num_frames=1,
            output_dir=os.path.join(tmp, "single"),
        )
        assert len(single_paths) == 1


def test_get_frames_partial_overlap_returns_available():
    """事件窗口只有部分与缓冲重叠时，返回重叠部分的帧。"""
    buffer = VideoBuffer(buffer_duration=2.0)
    # 0 ~ 4 秒，每 0.5 秒一帧；最新 4.0，缓冲保留 [2.0, 4.0]
    for index in range(9):
        buffer.add_frame(
            frame=_make_frame(),
            timestamp=index * 0.5,
        )

    with tempfile.TemporaryDirectory() as tmp:
        paths = buffer.get_frames(
            start_time=3.0,
            end_time=100.0,   # 大部分超出缓冲
            num_frames=5,
            output_dir=os.path.join(tmp, "partial"),
        )

    # 窗口内实际可用帧为 3.0 / 3.5 / 4.0 共 3 帧
    assert len(paths) == 3


def test_get_frames_rejects_nan():
    """NaN 时间戳应抛出清晰错误。"""
    buffer = VideoBuffer(buffer_duration=10.0)
    buffer.add_frame(
        frame=_make_frame(),
        timestamp=1.0,
    )

    try:
        buffer.get_frames(
            start_time=float("nan"),
            end_time=2.0,
        )
        assert False, "应当抛出 ValueError"
    except ValueError:
        pass

    try:
        buffer.get_frames(
            start_time=0.0,
            end_time=float("nan"),
        )
        assert False, "应当抛出 ValueError"
    except ValueError:
        pass


def test_saved_frame_pixels_match_input():
    """保存的关键帧像素内容应与输入一致（允许 jpg 有损）。"""
    with tempfile.TemporaryDirectory() as tmp:
        buffer = VideoBuffer(buffer_duration=10.0)
        gray = np.full((24, 32, 3), 123, dtype=np.uint8)
        buffer.add_frame(frame=gray, timestamp=0.0)

        path = buffer.get_frames(
            start_time=0.0,
            end_time=0.5,
            num_frames=1,
            output_dir=os.path.join(tmp, "pixels"),
        )[0]

        full = path if os.path.isabs(path) else os.path.join(
            os.getcwd(), path
        )
        image = cv2.imdecode(
            np.fromfile(full, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert image is not None
        assert image.shape == (24, 32, 3)
        # jpg 是有损压缩，允许小幅偏差
        assert abs(int(image.mean()) - 123) < 3


def test_black_frame_can_be_saved():
    """全黑帧也应能正常保存。"""
    with tempfile.TemporaryDirectory() as tmp:
        buffer = VideoBuffer(buffer_duration=10.0)
        buffer.add_frame(
            frame=np.zeros((16, 16, 3), dtype=np.uint8),
            timestamp=0.0,
        )

        paths = buffer.get_frames(
            start_time=0.0,
            end_time=0.5,
            num_frames=1,
            output_dir=os.path.join(tmp, "black"),
        )
        assert len(paths) == 1


def test_out_of_order_timestamps_do_not_crash():
    """乱序时间戳输入不应崩溃，仍可按窗口查询。"""
    buffer = VideoBuffer(buffer_duration=5.0)
    for timestamp in (5.0, 3.0, 4.0, 1.0, 6.0):
        buffer.add_frame(
            frame=_make_frame(),
            timestamp=timestamp,
        )

    with tempfile.TemporaryDirectory() as tmp:
        paths = buffer.get_frames(
            start_time=3.0,
            end_time=5.0,
            num_frames=3,
            output_dir=os.path.join(tmp, "ooo"),
        )

    # 窗口 [3,5] 内帧为 3.0 / 4.0 / 5.0 共 3 帧
    assert len(paths) == 3


def test_frame_exactly_at_cutoff_is_kept():
    """时间恰好在淘汰边界上的帧应被保留。"""
    buffer = VideoBuffer(buffer_duration=5.0)
    buffer.add_frame(
        frame=_make_frame(),
        timestamp=0.0,
    )
    buffer.add_frame(
        frame=_make_frame(),
        timestamp=5.0,   # 恰在 cutoff = 5.0 - 5.0 = 0.0 边界之上
    )

    assert len(buffer) == 2


def test_auto_output_dirs_increment_without_overwrite():
    """同一实例多次自动命名目录应递增(event_001, event_002)，不互相覆盖。"""
    with tempfile.TemporaryDirectory() as tmp:
        buffer = VideoBuffer(
            buffer_duration=20.0,
            clips_dir=os.path.join(tmp, "clips"),
        )
        for index in range(21):
            buffer.add_frame(
                frame=_make_frame(brightness=index * 10),
                timestamp=index * 0.5,
            )

        first = buffer.get_frames(
            start_time=0.0,
            end_time=10.0,
            num_frames=2,
        )
        second = buffer.get_frames(
            start_time=0.0,
            end_time=10.0,
            num_frames=2,
        )

        assert len(first) == 2 and len(second) == 2
        assert "event_001" in first[0]
        assert "event_002" in second[0]
        assert first[0] != second[0]


def test_add_frame_keeps_an_immutable_pixel_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        buffer = VideoBuffer(buffer_duration=10, clips_dir=tmp)
        frame = _make_frame(brightness=25)
        buffer.add_frame(frame, timestamp=0.0)
        frame[:] = 200

        paths = buffer.get_frames(
            0.0,
            0.0,
            output_dir=Path(tmp) / "snapshot",
        )
        saved = cv2.imdecode(
            np.fromfile(paths[0], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        assert int(saved[0, 0, 0]) < 50


# ============ 直接运行 ============

if __name__ == "__main__":
    test_add_frame_keeps_only_recent_duration()
    test_add_frame_accepts_single_frame()
    test_get_frames_returns_uniform_keyframes_in_window()
    test_get_frames_returns_all_when_window_has_few_frames()
    test_get_frames_returns_empty_when_no_frame_in_window()
    test_get_frames_never_returns_duplicate_frames()
    test_add_frame_rejects_non_array_frame()
    test_add_frame_rejects_invalid_timestamp()
    test_get_frames_rejects_invalid_arguments()
    test_create_video_buffer_factory()
    test_video_buffer_thread_safe_concurrent_write_and_query()
    test_get_frames_single_point_window_and_one_frame()
    test_get_frames_partial_overlap_returns_available()
    test_get_frames_rejects_nan()
    test_saved_frame_pixels_match_input()
    test_black_frame_can_be_saved()
    test_out_of_order_timestamps_do_not_crash()
    test_frame_exactly_at_cutoff_is_kept()
    test_auto_output_dirs_increment_without_overwrite()
    print("ALL TESTS PASSED (VideoBuffer)")
