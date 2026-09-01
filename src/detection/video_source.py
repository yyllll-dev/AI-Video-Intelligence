"""视频输入层 —— 统一摄像头与视频文件的连续帧源

本文件：src/detection/video_source.py —— 统一视频输入接口。

设计原则：
    - 本地摄像头与上传视频文件实现同一个 VideoSource 接口，输出统一的 VideoFrame。
    - 后续 YOLO / Tracking / Pipeline 只认 open_source() 返回的帧源，不区分输入来源。
    - 每个 VideoFrame 至少包含 frame / frame_id / timestamp / width / height。
"""

import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np


# ============ 读取健壮性参数 ============

_READ_RETRIES = 5            # 瞬时读取失败重试次数（USB 摄像头偶发丢帧 / 视频坏帧）
_READ_RETRY_DELAY = 0.02     # 每次重试间隔（秒）


# ============ 统一连续视频帧对象 ============

@dataclass
class VideoFrame:
    """一帧连续视频的统一表示，摄像头与视频文件输出完全一致。"""
    frame: np.ndarray      # BGR 图像数组
    frame_id: int          # 帧序号，从 0 开始递增
    timestamp: float       # 秒；摄像头=相对启动时刻，视频文件=视频内部时间
    width: int             # 画面宽度（像素）
    height: int            # 画面高度（像素）


# ============ 统一帧源接口 ============

class VideoSource:
    """统一视频输入接口：摄像头与上传视频都实现本接口。"""

    def read(self) -> Optional[VideoFrame]:
        """读取下一帧；视频结束或持续读取失败返回 None。"""
        raise NotImplementedError

    def _read_frame(self):
        """读取一帧原始图像；瞬时失败自动重试，持续失败（含视频结束）返回 None。"""
        for _ in range(_READ_RETRIES):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                return frame
            time.sleep(_READ_RETRY_DELAY)
        return None

    def release(self) -> None:
        """释放底层资源。"""
        raise NotImplementedError

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False


# ============ 本地摄像头输入 ============

class CameraSource(VideoSource):
    """本地摄像头输入，逐帧输出 VideoFrame。"""

    def __init__(self, source: int = 0, width: int = 1280, height: int = 720):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap.release()
            raise ValueError(f"无法打开摄像头，source={source}")
        # 尽力设置分辨率，部分设备可能忽略
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._frame_id = 0
        self._start = time.monotonic()

    def read(self) -> Optional[VideoFrame]:
        if self._cap is None:
            return None
        frame = self._read_frame()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        video_frame = VideoFrame(
            frame=frame,
            frame_id=self._frame_id,
            timestamp=time.monotonic() - self._start,
            width=w,
            height=h,
        )
        self._frame_id += 1
        return video_frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ============ 本地 / 上传视频文件输入 ============

class VideoFileSource(VideoSource):
    """视频文件输入（含网络上传后落盘的本地文件），输出与摄像头一致的 VideoFrame。"""

    def __init__(self, path: str):
        if not isinstance(path, str):
            raise TypeError(f"视频路径必须是字符串，收到 {type(path)}")
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._cap.release()
            raise ValueError(f"无法打开视频文件: {path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0
        self._frame_id = 0

    def read(self) -> Optional[VideoFrame]:
        if self._cap is None:
            return None
        frame = self._read_frame()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        # 优先用视频真实播放位置（兼容可变帧率视频），取不到时退化为 帧号/帧率
        pos_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = pos_msec / 1000.0 if pos_msec and pos_msec > 0 else self._frame_id / self._fps
        video_frame = VideoFrame(
            frame=frame,
            frame_id=self._frame_id,
            timestamp=timestamp,
            width=w,
            height=h,
        )
        self._frame_id += 1
        return video_frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ============ 统一入口：摄像头与上传视频进入同一链路 ============

def open_source(source: Union[int, str]) -> VideoSource:
    """统一输入入口：int=本地摄像头，str=视频文件路径，返回统一 VideoSource。"""
    if isinstance(source, int):
        return CameraSource(source)
    if isinstance(source, str):
        return VideoFileSource(source)
    raise TypeError(f"不支持的视频输入类型: {type(source)}，需要 int(摄像头) 或 str(文件路径)")
