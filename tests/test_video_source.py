"""C 模块 —— 视频输入接口最小测试

分工：C 杨潇睿 —— YOLO + Tracking
本文件：tests/test_video_source.py —— Day 1：验证摄像头 / 视频文件 / 非法输入。

运行方式（在仓库根目录）：
    - python -m pytest tests/test_video_source.py      # pytest 方式
    - python tests/test_video_source.py                # 全部最小验证
    - python tests/test_video_source.py --video 路径   # 用指定视频文件验证
    - python tests/test_video_source.py --camera       # 用真实摄像头验证
"""

import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

# 保证从仓库根目录可直接导入 src 包（项目尚未安装为包）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.detection.video_source import (  # noqa: E402
    CameraSource,
    VideoFileSource,
    VideoFrame,
    open_source,
)


# ============ 工具：生成临时测试视频 ============

def _make_test_video(path, frames=8, width=320, height=240, fps=10.0):
    """用 OpenCV 生成一段纯色渐变测试视频，用于模拟上传视频文件。"""
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


# ============ 视频文件输入测试（模拟网络上传视频） ============

def test_video_file_source_outputs_unified_frames():
    """上传视频文件输出统一 VideoFrame：frame / frame_id / timestamp / width / height。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.mp4")
        _make_test_video(path, frames=8, fps=10.0)
        with VideoFileSource(path) as src:
            frames = []
            while True:
                vf = src.read()
                if vf is None:
                    break
                frames.append(vf)
        assert len(frames) == 8
        first = frames[0]
        assert isinstance(first, VideoFrame)
        assert first.frame_id == 0
        assert first.timestamp == 0.0
        assert first.width == 320
        assert first.height == 240
        assert first.frame.shape == (240, 320, 3)
        # timestamp 按帧率推进
        assert abs(frames[1].timestamp - 0.1) < 1e-6
        # frame_id 连续递增
        assert [f.frame_id for f in frames] == list(range(8))


def test_video_file_source_ends_with_none():
    """视频读完后 read() 返回 None，且再次读取仍为 None。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.mp4")
        _make_test_video(path, frames=3)
        with VideoFileSource(path) as src:
            count = 0
            while src.read() is not None:
                count += 1
            assert count == 3
            assert src.read() is None


# ============ 摄像头输入测试（无摄像头则跳过） ============

def test_camera_source_outputs_unified_frames():
    """摄像头输出统一 VideoFrame；本机无摄像头时跳过。"""
    if not _has_camera():
        print("SKIP: 本机无可用摄像头，跳过摄像头测试")
        return
    with CameraSource(0) as src:
        vf = src.read()
        assert vf is not None
        assert isinstance(vf, VideoFrame)
        assert vf.frame_id == 0
        assert vf.width > 0 and vf.height > 0
        assert vf.frame.shape[:2] == (vf.height, vf.width)
        assert vf.timestamp >= 0.0


# ============ 非法输入测试 ============

def test_invalid_video_path_raises():
    """不存在的视频文件路径应抛出 ValueError。"""
    bad_path = os.path.join(tempfile.gettempdir(), "not_exist_video_xxx.mp4")
    try:
        VideoFileSource(bad_path)
        assert False, "应当抛出 ValueError"
    except ValueError:
        pass


def test_invalid_source_type_raises():
    """open_source 收到非法类型（float/None/object）应抛出 TypeError。"""
    for bad in (3.14, None, object()):
        try:
            open_source(bad)
            assert False, f"应当抛出 TypeError: {bad!r}"
        except TypeError:
            pass


# ============ 统一入口测试：摄像头与文件进入同一链路 ============

def test_open_source_unified_entry():
    """open_source 是统一入口：int 返回摄像头源，str 返回文件源。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.mp4")
        _make_test_video(path, frames=2)
        assert isinstance(open_source(0), CameraSource)
        assert isinstance(open_source(path), VideoFileSource)


def test_two_sources_output_same_frame_object():
    """摄像头与视频文件输出同一种 VideoFrame 对象（共用后续流程的前提）。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.mp4")
        _make_test_video(path, frames=2)
        with VideoFileSource(path) as file_src:
            file_frame = file_src.read()
        assert isinstance(file_frame, VideoFrame)
        if not _has_camera():
            print("SKIP: 本机无可用摄像头，仅验证文件路")
            return
        with CameraSource(0) as cam_src:
            cam_frame = cam_src.read()
        assert isinstance(cam_frame, VideoFrame)
        # 两路输出的字段完全一致
        assert set(file_frame.__dataclass_fields__) == set(cam_frame.__dataclass_fields__)


# ============ 可执行验证入口 ============

def _run_all():
    """无 pytest 时直接以脚本方式运行全部最小验证。"""
    failed = 0
    tests = [
        test_video_file_source_outputs_unified_frames,
        test_video_file_source_ends_with_none,
        test_camera_source_outputs_unified_frames,
        test_invalid_video_path_raises,
        test_invalid_source_type_raises,
        test_open_source_unified_entry,
        test_two_sources_output_same_frame_object,
    ]
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {t.__name__}: {e!r}")
    print(f"\n结果: {len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频输入接口最小验证")
    parser.add_argument("--video", type=str, default=None, help="用指定视频文件验证")
    parser.add_argument("--camera", action="store_true", help="用真实摄像头验证")
    args = parser.parse_args()

    if args.video:
        with VideoFileSource(args.video) as src:
            count = 0
            while True:
                vf = src.read()
                if vf is None:
                    break
                count += 1
                if count <= 3 or vf.frame_id % 30 == 0:
                    print(f"frame_id={vf.frame_id} timestamp={vf.timestamp:.2f}s "
                          f"size={vf.width}x{vf.height}")
            print(f"视频读取完成，共 {count} 帧: {args.video}")
    elif args.camera:
        with CameraSource(0) as src:
            print("摄像头已打开，读取前 10 帧...")
            for _ in range(10):
                vf = src.read()
                if vf is None:
                    print("摄像头读取中断")
                    break
                print(f"frame_id={vf.frame_id} timestamp={vf.timestamp:.2f}s "
                      f"size={vf.width}x{vf.height}")
    else:
        sys.exit(_run_all())
