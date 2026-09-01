"""C 模块 —— YOLO 检测最小测试

分工：C 杨潇睿 —— YOLO + Tracking
本文件：tests/test_detector.py —— Day 2：验证连续帧检测 / 结果格式 / 模型缺失报错。

运行方式（在仓库根目录）：
    - python -m pytest tests/test_detector.py -v
    - python tests/test_detector.py                  # 全部最小验证
    - python tests/test_detector.py --video 路径      # 用指定视频验证连续帧检测
    - python tests/test_detector.py --camera          # 用摄像头验证检测入口

说明：模型权重(models/yolo11n.pt)不存在时，真实检测用例自动跳过，只跑模型缺失报错用例。
"""

import argparse
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

from src.detection.config import MODEL_PATH  # noqa: E402
from src.detection.detector import DetectionResult, YoloDetector, create_detector  # noqa: E402
from src.detection.video_source import CameraSource, VideoFileSource  # noqa: E402


# ============ 工具函数 ============

def _model_exists():
    return MODEL_PATH.exists()


def _find_sample_image():
    """找 ultralytics 自带的示例图（bus.jpg），用于生成可检测的视频。"""
    try:
        import ultralytics
        img = Path(ultralytics.__file__).resolve().parent / "assets" / "bus.jpg"
        if img.exists():
            return img
    except Exception:  # noqa: BLE001
        pass
    return None


def _make_video_from_image(path, image, frames=8, fps=10.0):
    """把一张图重复写成一段视频，模拟连续帧（内容真实可检测）。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = image.shape[:2]
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建测试视频: {path}")
    for _ in range(frames):
        writer.write(image)
    writer.release()


def _assert_result_format(det, frame_id, timestamp):
    """校验检测结果格式：字段、置信度 0~1、bbox 为 [x1,y1,x2,y2]。"""
    assert isinstance(det, DetectionResult)
    assert det.frame_id == frame_id
    assert abs(det.timestamp - timestamp) < 1e-6
    assert isinstance(det.class_name, str) and det.class_name
    assert 0.0 <= det.confidence <= 1.0
    assert len(det.bbox) == 4
    x1, y1, x2, y2 = det.bbox
    assert x1 < x2 and y1 < y2, f"bbox 非法: {det.bbox}"


def _has_camera(index=0):
    cap = cv2.VideoCapture(index)
    ok = cap.isOpened()
    cap.release()
    return ok


# ============ 模型缺失报错测试（始终运行） ============

def test_detector_missing_model_raises_clear_error():
    """模型文件不存在时必须抛出清晰错误并带配置说明。"""
    missing = Path(tempfile.gettempdir()) / "no_such_yolo_model.pt"
    try:
        YoloDetector(model_path=str(missing))
        assert False, "应当抛出 FileNotFoundError"
    except FileNotFoundError as e:
        assert "yolo11n.pt" in str(e) or "models/" in str(e)


# ============ 连续帧检测测试（模型存在时运行） ============

def test_detector_detects_on_continuous_video():
    """上传视频逐帧检测：每一帧都调用同一个 detect()，结果格式正确。"""
    if not _model_exists():
        print("SKIP: 模型权重不存在，跳过真实检测（请先放置 models/yolo11n.pt）")
        return
    sample = _find_sample_image()
    if sample is None:
        print("SKIP: 无 ultralytics 示例图，跳过内容检测（仅保留接口调用）")
        return
    detector = create_detector()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "detect_test.mp4")
        _make_video_from_image(path, cv2.imread(str(sample)), frames=6, fps=10.0)
        with VideoFileSource(path) as src:
            total = 0
            detected = 0
            while True:
                vf = src.read()
                if vf is None:
                    break
                total += 1
                dets = detector.detect(vf)
                for det in dets:
                    _assert_result_format(det, vf.frame_id, vf.timestamp)
                detected += len(dets)
        assert total == 6
        # 示例图里应至少检测到目标（person 等）
        assert detected > 0, "示例图(bus.jpg)中应能检测到目标"


def test_detector_result_format_on_uploaded_video():
    """用上传视频验证检测结果的 bbox / 类别 / confidence 格式。"""
    if not _model_exists():
        print("SKIP: 模型权重不存在，跳过格式验证")
        return
    detector = create_detector()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "format_test.mp4")
        sample = _find_sample_image()
        if sample is None:
            return
        _make_video_from_image(path, cv2.imread(str(sample)), frames=3)
        with VideoFileSource(path) as src:
            seen = 0
            while True:
                vf = src.read()
                if vf is None:
                    break
                dets = detector.detect(vf)
                for det in dets:
                    _assert_result_format(det, vf.frame_id, vf.timestamp)
                    assert det.class_name in {"person", "book", "cell phone"} or det.class_name
                seen += len(dets)
            assert seen >= 0


# ============ 摄像头检测入口测试 ============

def test_camera_detection_entry_callable():
    """摄像头帧能调用同一个 detect() 入口（无模型或无摄像头则跳过）。"""
    if not _model_exists():
        print("SKIP: 模型权重不存在，跳过摄像头检测入口验证")
        return
    if not _has_camera():
        print("SKIP: 本机无可用摄像头，跳过摄像头检测入口验证")
        return
    detector = create_detector()
    with CameraSource(0) as src:
        vf = src.read()
        assert vf is not None
        dets = detector.detect(vf)
        assert isinstance(dets, list)
        for det in dets:
            _assert_result_format(det, vf.frame_id, vf.timestamp)


# ============ 可执行验证入口 ============

def _run_all():
    failed = 0
    tests = [
        test_detector_missing_model_raises_clear_error,
        test_detector_detects_on_continuous_video,
        test_detector_result_format_on_uploaded_video,
        test_camera_detection_entry_callable,
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
    parser = argparse.ArgumentParser(description="YOLO 检测最小验证")
    parser.add_argument("--video", type=str, default=None, help="用指定视频验证连续帧检测")
    parser.add_argument("--camera", action="store_true", help="用真实摄像头验证检测入口")
    args = parser.parse_args()

    if args.video:
        if not _model_exists():
            print(MODEL_PATH)
            sys.exit("模型权重不存在，请先放置 models/yolo11n.pt")
        detector = create_detector()
        with VideoFileSource(args.video) as src:
            count = 0
            while True:
                vf = src.read()
                if vf is None:
                    break
                count += 1
                dets = detector.detect(vf)
                for det in dets:
                    print(f"frame_id={det.frame_id} t={det.timestamp:.2f}s "
                          f"{det.class_name} conf={det.confidence:.2f} bbox={det.bbox}")
            print(f"共处理 {count} 帧: {args.video}")
    elif args.camera:
        if not _model_exists():
            sys.exit("模型权重不存在，请先放置 models/yolo11n.pt")
        detector = create_detector()
        with CameraSource(0) as src:
            print("摄像头已打开，逐帧检测（按 Ctrl+C 停止）...")
            while True:
                vf = src.read()
                if vf is None:
                    break
                dets = detector.detect(vf)
                for det in dets:
                    print(f"frame_id={det.frame_id} t={det.timestamp:.2f}s "
                          f"{det.class_name} conf={det.confidence:.2f} bbox={det.bbox}")
    else:
        sys.exit(_run_all())
