"""杨潇睿 模块 —— FrameStream 逐帧操作循环最小测试

分工：杨潇睿 —— VideoBuffer / FrameStream（实时视频 → 历史画面 → VLM 的桥梁）
本文件：tests/test_frame_stream.py

运行方式（在仓库根目录）：
    - python -m pytest tests/test_frame_stream.py -v   # pytest 方式
    - python tests/test_frame_stream.py                 # 直接运行全部断言

覆盖点：
    1. 长视频文件：逐帧全部喂入 VideoBuffer，帧数正确、自动释放。
    2. 视频文件帧的时间戳递增，VideoFrame 结构完整。
    3. on_frame 回调对每一帧都被调用。
    4. 视频结束后 step() / run() 正确返回。
    5. 流式喂入后，能按事件时间窗口取回关键帧（回看最近画面）。
    6. 摄像头模式：本机无摄像头时跳过（有摄像头时才跑）。
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

from src.detection.video_source import VideoFrame  # noqa: E402
from src.pipeline.frame_stream import FrameStream  # noqa: E402
from src.pipeline.video_buffer import VideoBuffer  # noqa: E402


# ============ 工具：生成临时长视频 ============

def _make_test_video(path, frames=60, width=320, height=240, fps=10.0):
    """用 OpenCV 生成一段纯色渐变视频，模拟"长视频文件"。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建测试视频: {path}")
    for i in range(frames):
        value = int(255 * i / max(frames - 1, 1))
        frame = np.full((height, width, 3), value, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _has_camera(index=0):
    """探测本机是否有可用摄像头。"""
    cap = cv2.VideoCapture(index)
    ok = cap.isOpened()
    cap.release()
    return ok


# ============ 长视频：逐帧全部喂入 ============

def test_frame_stream_feeds_all_video_frames_into_buffer():
    """长视频文件逐帧全部喂入 VideoBuffer。"""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "long.mp4")
        _make_test_video(video_path, frames=60, fps=10.0)  # 6 秒

        buffer = VideoBuffer(
            buffer_duration=5.0,
            clips_dir=os.path.join(tmp, "clips"),
        )

        stream = FrameStream(
            source=video_path,
            buffer=buffer,
        )

        stats = stream.run()

        # 60 帧全部处理
        assert int(stats["frames"]) == 60
        assert stream.processed == 60

        # 时间从 0.0 到约 5.9 秒
        assert abs(stats["start_time"] - 0.0) < 1e-6
        assert abs(stats["end_time"] - 5.9) < 1e-6

        # 缓冲只保留最近 5 秒：约保留 0.9s ~ 5.9s 的帧
        assert len(buffer) > 0
        assert buffer.latest_timestamp > 5.89
        assert buffer.earliest_timestamp > 0.8

        # 视频结束后缓冲里能取回最近画面关键帧
        paths = buffer.get_frames(
            start_time=3.0,
            end_time=5.9,
            num_frames=5,
        )
        assert len(paths) == 5


def test_frame_stream_video_frame_has_valid_fields():
    """逐帧取出的 VideoFrame 结构完整、时间戳递增。"""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "short.mp4")
        _make_test_video(video_path, frames=8, fps=10.0)

        timestamps = []

        def on_frame(video_frame):
            assert isinstance(video_frame, VideoFrame)
            assert video_frame.frame.shape[:2] == (240, 320)
            timestamps.append(video_frame.timestamp)

        stream = FrameStream(
            source=video_path,
            on_frame=on_frame,
        )

        stream.run()

        assert len(timestamps) == 8
        # 时间戳单调递增
        assert timestamps == sorted(timestamps)
        assert abs(timestamps[0] - 0.0) < 1e-6
        assert abs(timestamps[-1] - 0.7) < 1e-6


def test_frame_stream_calls_on_frame_for_each_frame():
    """on_frame 回调对每一帧都会被调用一次。"""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "cb.mp4")
        _make_test_video(video_path, frames=20, fps=10.0)

        counter = {"frames": 0}

        def on_frame(video_frame):
            counter["frames"] += 1

        stream = FrameStream(
            source=video_path,
            on_frame=on_frame,
        )

        stream.run()

        assert counter["frames"] == 20


def test_frame_stream_stops_at_video_end():
    """视频结束后 step() / run() 正确返回，不再出帧。"""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "end.mp4")
        _make_test_video(video_path, frames=5, fps=10.0)

        stream = FrameStream(source=video_path)

        # 手动逐帧 step，读到结尾
        count = 0
        while stream.step() is not None:
            count += 1

        assert count == 5

        # 视频结束后 step 返回 None
        assert stream.step() is None

        # run() 在已结束后不再处理新帧
        stats = stream.run()
        assert int(stats["frames"]) == 5


def test_frame_stream_streaming_then_lookback_keyframes():
    """流式喂入整段视频后，能按窗口取回关键帧（回看最近画面）。"""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "lookback.mp4")
        _make_test_video(video_path, frames=100, fps=10.0)  # 10 秒

        buffer = VideoBuffer(
            buffer_duration=10.0,
            clips_dir=os.path.join(tmp, "clips"),
        )

        stream = FrameStream(
            source=video_path,
            buffer=buffer,
        )

        stream.run()

        # 把"最后一小段"当作刚发生的事件窗口
        latest = buffer.latest_timestamp
        assert latest is not None

        paths = buffer.get_frames(
            start_time=max(0.0, latest - 2.0),
            end_time=latest,
            num_frames=5,
        )

        # 5 张关键帧全部真实存在
        assert len(paths) == 5
        for path in paths:
            assert Path(path).exists()


# ============ 实时摄像头（无摄像头则跳过） ============

def test_frame_stream_camera_starts_and_stops():
    """实时摄像头：能打开则逐帧处理指定帧数；无摄像头跳过。"""
    if not _has_camera():
        print("SKIP: 本机无可用摄像头，跳过摄像头测试")
        return

    buffer = VideoBuffer(buffer_duration=3.0)
    stream = FrameStream(
        source=0,
        buffer=buffer,
    )

    stats = stream.run(max_frames=10)

    assert int(stats["frames"]) >= 1
    assert buffer.latest_timestamp is not None


# ============ 健全性：资源与参数边界 ============

def test_frame_stream_step_after_close_returns_none():
    """close() 之后 step() 不再出帧。"""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "closed.mp4")
        _make_test_video(video_path, frames=5, fps=10.0)

        stream = FrameStream(source=video_path)
        stream.run()

        # run() 已释放自己打开的源，之后 step 应为 None
        assert stream.step() is None


def test_frame_stream_rejects_invalid_source_type():
    """非法 source 类型应抛出清晰错误。"""
    try:
        FrameStream(source=3.14)   # 非 int(摄像头)、非 str(视频)、非 VideoSource
        assert False, "应当抛出 TypeError"
    except TypeError:
        pass


def test_frame_stream_video_with_target_fps_still_processes_all_frames():
    """视频文件设置 target_fps 后仍应逐帧处理完全部帧。"""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "fps.mp4")
        _make_test_video(video_path, frames=8, fps=10.0)

        stream = FrameStream(
            source=video_path,
            target_fps=50.0,   # 帧率较高，总等待时间很短
        )

        stats = stream.run()

        assert int(stats["frames"]) == 8


# ============ 直接运行 ============

if __name__ == "__main__":
    test_frame_stream_feeds_all_video_frames_into_buffer()
    test_frame_stream_video_frame_has_valid_fields()
    test_frame_stream_calls_on_frame_for_each_frame()
    test_frame_stream_stops_at_video_end()
    test_frame_stream_streaming_then_lookback_keyframes()
    test_frame_stream_camera_starts_and_stops()
    test_frame_stream_step_after_close_returns_none()
    test_frame_stream_rejects_invalid_source_type()
    test_frame_stream_video_with_target_fps_still_processes_all_frames()
    print("ALL TESTS PASSED (FrameStream)")
