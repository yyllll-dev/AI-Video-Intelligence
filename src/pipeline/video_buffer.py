"""视频环形缓冲 —— 为实时视频保留"最近 N 秒"的历史画面

本文件：src/pipeline/video_buffer.py

职责（对齐任务书讲解）：
    VideoBuffer 位于实时视频与 VLM 之间的"桥梁"位置：

    Camera Frame
        ↓
    VideoBuffer
        ↓  保存最近 N 秒（buffer_duration）
        ↓  输入 start_time / end_time
        ↓  找到对应历史帧
        ↓  均匀抽取 3~5 帧
        ↓  保存为 jpg
        ↓
    返回图片路径 list[str]  →  交给 B 的 Qwen-VL 分析

    本模块不负责：
        - 目标检测 / 跟踪 / 事件判断（A、C 已有模块负责）
        - Qwen-VL 调用（B 的 src/vlm 负责）
        - Embedding / Retrieval（D 的 src/retrieval 负责）

接口（与任务书讲解一致）：
    buffer = VideoBuffer(buffer_duration=10)
    buffer.add_frame(frame=image, timestamp=timestamp)
    paths = buffer.get_frames(start_time=..., end_time=..., num_frames=5)

设计说明：
    1. 只保留最近 buffer_duration 秒内的帧，旧帧自动淘汰，内存不会无限增长。
    2. 时间戳必须是单调递增的视频时间（对应 src/detection/video_source.py
       输出的 VideoFrame.timestamp 口径）。
    3. 关键帧落盘目录默认在 data/clips/event_xxx/ 下（与 D 模块的
       data/clips 约定一致），图片命名为 frame_001.jpg、frame_002.jpg ...
    4. 本模块不依赖 YOLO / EventEngine / Qwen-VL，可独立运行与测试。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Union

import cv2
import numpy as np


# ============================================================
# 路径与默认配置
# ============================================================

# 项目根目录：本文件位于 src/pipeline/ 下
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认关键帧落盘根目录
# 对齐 D 模块 README 中的约定：data/clips/event_xxx/
DEFAULT_CLIPS_DIR = _PROJECT_ROOT / "data" / "clips"

# 事件关键帧目录名前缀（event_001、event_002 ...）
_EVENT_DIR_PREFIX = "event_"

# 图片文件名编号位数（frame_001.jpg）
_FILENAME_DIGITS = 3

# 时间戳比较容差（秒）
_TIME_EPS = 1e-6


# ============================================================
# VideoBuffer 内部保存的一帧
# ============================================================

@dataclass
class BufferedFrame:
    """VideoBuffer 内部保存的一帧历史画面。

    字段：
        frame：
            一帧 BGR 图像（与 VideoFrame.frame 同口径）。

        timestamp：
            该帧的视频时间（秒），
            必须与 Event.start_time / end_time 使用同一时间轴。
    """

    frame: np.ndarray
    timestamp: float


# ============================================================
# 视频环形缓冲
# ============================================================

class VideoBuffer:
    """保存最近一段时间视频画面的环形缓冲。

    输入：
        add_frame(frame, timestamp)：
            逐帧写入，自动只保留最近 buffer_duration 秒。

    输出：
        get_frames(start_time, end_time, num_frames)：
            从历史画面中取出事件时间窗口内的若干关键帧，
            均匀抽取后保存为 jpg，
            返回图片路径列表 list[str]。

    典型用法：

        buffer = VideoBuffer(buffer_duration=10)
        buffer.add_frame(frame=image, timestamp=632.0)
        ...
        paths = buffer.get_frames(
            start_time=632.5,
            end_time=640.0,
            num_frames=5,
        )
    """

    def __init__(
        self,
        buffer_duration: float = 10.0,
        clips_dir: Optional[Union[str, Path]] = None,
    ):
        """创建视频环形缓冲。

        参数：
            buffer_duration：
                最多保留多少秒的历史画面，超过的部分自动淘汰。

            clips_dir：
                关键帧 jpg 的落盘根目录。
                默认使用项目 data/clips/ 目录。
        """

        if not isinstance(buffer_duration, (int, float)):
            raise TypeError(
                f"buffer_duration 必须是数字，收到 {type(buffer_duration)}"
            )

        if buffer_duration <= 0:
            raise ValueError(
                f"buffer_duration 必须大于 0，收到 {buffer_duration}"
            )

        self.buffer_duration = float(buffer_duration)

        # 历史帧队列：队首最旧、队尾最新
        self._frames: Deque[BufferedFrame] = deque()

        # 线程安全锁（可重入）：
        # 采集线程持续 add_frame，
        # 事件/查询线程可能同时访问缓冲，
        # 因此所有读写缓冲状态的操作都必须在锁内进行。
        self._lock = threading.RLock()

        # 关键帧落盘根目录
        self._clips_dir = (
            Path(clips_dir)
            if clips_dir is not None
            else DEFAULT_CLIPS_DIR
        )

        # 自动生成事件目录时使用的自增序号
        self._event_counter = 0

    # ========================================================
    # 只读辅助
    # ========================================================

    def __len__(self) -> int:
        """当前缓存的帧数量（便于测试与调试）。"""

        with self._lock:

            return len(self._frames)

    @property
    def earliest_timestamp(self) -> Optional[float]:
        """当前缓存中最旧一帧的时间戳；空缓冲返回 None。"""

        with self._lock:

            if not self._frames:
                return None

            return self._frames[0].timestamp

    @property
    def latest_timestamp(self) -> Optional[float]:
        """当前缓存中最新一帧的时间戳；空缓冲返回 None。"""

        with self._lock:

            if not self._frames:
                return None

            return self._frames[-1].timestamp

    # ========================================================
    # 写入接口
    # ========================================================

    def add_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> None:
        """向缓冲写入一帧，并自动淘汰超过 buffer_duration 的旧帧。

        参数：
            frame：
                一帧 BGR 图像（numpy 数组）。

            timestamp：
                该帧的视频时间（秒）。
                约定与 VideoFrame.timestamp 同口径，
                并应随视频播放单调递增。
        """

        # ----------------------------------------------------
        # 1. 参数校验
        # ----------------------------------------------------

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"frame 必须是 numpy 数组（一帧 BGR 图像），"
                f"收到 {type(frame)}"
            )

        if not isinstance(timestamp, (int, float)):
            raise TypeError(
                f"timestamp 必须是数字（秒），收到 {type(timestamp)}"
            )

        timestamp = float(timestamp)

        if timestamp != timestamp:  # NaN 校验
            raise ValueError("timestamp 不能是 NaN")

        if timestamp < 0:
            raise ValueError(
                f"timestamp 不能为负数，收到 {timestamp}"
            )

        # ----------------------------------------------------
        # 2. 写入当前帧并淘汰过期帧（全程持锁，保证线程安全）
        # ----------------------------------------------------

        with self._lock:

            self._frames.append(
                BufferedFrame(
                    frame=frame.copy(),
                    timestamp=timestamp,
                )
            )

            # ------------------------------------------------
            # 3. 淘汰超过 buffer_duration 的旧帧
            #
            # 保留窗口为 [最新时间 - buffer_duration, 最新时间]。
            # ------------------------------------------------

            cutoff = (
                self._frames[-1].timestamp
                - self.buffer_duration
            )

            while (
                self._frames
                and self._frames[0].timestamp < cutoff
            ):
                self._frames.popleft()

    # ========================================================
    # 查询接口
    # ========================================================

    def get_frames(
        self,
        start_time: float,
        end_time: float,
        num_frames: int = 5,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """从历史画面中取出一段事件的关键帧并保存为 jpg。

        参数：
            start_time：
                事件开始时间（秒），对应 Event.start_time。

            end_time：
                事件结束时间（秒），对应 Event.end_time。

            num_frames：
                需要抽取的关键帧数量（推荐 3~5）。

            output_dir：
                本次关键帧的保存目录。
                默认自动创建 data/clips/event_xxx/ 目录。

        返回：
            图片路径列表 list[str]，按时间顺序排列。
            例如：
                ["data/clips/event_001/frame_001.jpg",
                 "data/clips/event_001/frame_002.jpg", ...]

        说明：
            1. 只处理落在 [start_time, end_time] 内的历史帧。
            2. 在事件时间轴上均匀取 num_frames 个目标时刻，
               每个目标时刻取最接近的一帧，避免图片重复。
            3. 若窗口内可用的帧不足 num_frames，
               有多少返回多少（含窗口为空时返回空列表）。
        """

        # ----------------------------------------------------
        # 1. 参数校验
        # ----------------------------------------------------

        if not isinstance(start_time, (int, float)):
            raise TypeError(
                f"start_time 必须是数字（秒），收到 {type(start_time)}"
            )

        if not isinstance(end_time, (int, float)):
            raise TypeError(
                f"end_time 必须是数字（秒），收到 {type(end_time)}"
            )

        if not isinstance(num_frames, int):
            raise TypeError(
                f"num_frames 必须是整数，收到 {type(num_frames)}"
            )

        start_time = float(start_time)
        end_time = float(end_time)

        if start_time != start_time or end_time != end_time:
            raise ValueError(
                "start_time / end_time 不能是 NaN"
            )

        if end_time < start_time:
            raise ValueError(
                f"end_time({end_time}) 不能早于 start_time({start_time})"
            )

        if num_frames < 1:
            raise ValueError(
                f"num_frames 必须至少为 1，收到 {num_frames}"
            )

        # ----------------------------------------------------
        # 2. 取出事件时间窗口内的历史帧
        #    并确定本次关键帧的落盘目录
        #
        # 锁内完成：
        #   - 快照时间窗口内的帧（返回后可安全在锁外采样/存图）；
        #   - 分配自动事件目录序号（event_001 ...），避免多线程重号。
        # ----------------------------------------------------

        with self._lock:

            window = [
                buffered
                for buffered in self._frames
                if (
                    buffered.timestamp >= start_time - _TIME_EPS
                    and buffered.timestamp <= end_time + _TIME_EPS
                )
            ]

            if output_dir is not None:

                output_directory = Path(output_dir)

            else:

                # 自动生成 data/clips/event_001、event_002 ...
                self._event_counter += 1

                output_directory = (
                    self._clips_dir
                    / f"{_EVENT_DIR_PREFIX}{self._event_counter:03d}"
                )

        if not window:
            print(
                "[VideoBuffer] 警告：事件时间窗口内没有可用帧，"
                f"[{start_time}, {end_time}]，返回空列表。"
            )

            return []

        # ----------------------------------------------------
        # 3. 均匀抽取关键帧
        # ----------------------------------------------------

        selected = self._sample_uniform(
            window,
            start_time,
            end_time,
            num_frames,
        )

        # ----------------------------------------------------
        # 4. 保存关键帧为 jpg
        # ----------------------------------------------------

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved_paths: List[str] = []

        for index, buffered in enumerate(selected, start=1):

            image_path = output_directory / (
                f"frame_{index:0{_FILENAME_DIGITS}d}.jpg"
            )

            # 注意：
            # Windows 下 cv2.imwrite 无法写入含中文的路径
            # （本项目根目录含"国创赛"等中文字符）。
            # 因此先 cv2.imencode 编码为 jpg 字节，
            # 再用 Python 原生 tofile 写盘，兼容中文路径。
            success, encoded = cv2.imencode(
                ".jpg",
                buffered.frame,
            )

            if not success:
                raise RuntimeError(
                    f"关键帧编码失败: {image_path}"
                )

            encoded.tofile(str(image_path))

            saved_paths.append(
                self._to_reported_path(image_path)
            )

        return saved_paths

    # ========================================================
    # 内部工具
    # ========================================================

    @staticmethod
    def _sample_uniform(
        window: List[BufferedFrame],
        start_time: float,
        end_time: float,
        num_frames: int,
    ) -> List[BufferedFrame]:
        """在事件时间轴上均匀抽取关键帧。

        思路：
            1. 计算 num_frames 个均匀目标时刻
               （含事件开始与事件结束两个端点）；
            2. 为每个目标时刻，在窗口内挑选时间上最接近的一帧；
            3. 同一帧不会被重复选中；
            4. 结果按时间顺序返回。

        说明：
            若窗口内帧数不足 num_frames，则直接返回全部帧，
            保证不会因为候选耗尽而出错。
        """

        if len(window) <= num_frames:
            return sorted(
                window,
                key=lambda buffered: buffered.timestamp,
            )

        # 已按时间排序的候选帧（add_frame 顺序通常已有序，这里再排序一次）
        candidates = sorted(
            window,
            key=lambda buffered: buffered.timestamp,
        )

        selected: List[BufferedFrame] = []

        # num_frames >= 2 时：起点 = start_time，终点 = end_time
        denominator = max(num_frames - 1, 1)

        for index in range(num_frames):

            target = (
                start_time
                + (end_time - start_time)
                * index
                / denominator
            )

            # 按索引找最接近目标时刻的一帧，
            # 避免使用 list.remove（dataclass 含 numpy 数组，
            # 直接做相等比较会产生歧义错误）。
            best_index = min(
                range(len(candidates)),
                key=lambda index: abs(
                    candidates[index].timestamp - target
                ),
            )

            best = candidates[best_index]

            selected.append(best)

            # 删除已选中的帧，保证同一帧不会被重复抽取
            del candidates[best_index]

        selected.sort(
            key=lambda buffered: buffered.timestamp,
        )

        return selected

    def _to_reported_path(
        self,
        image_path: Path,
    ) -> str:
        """把图片绝对路径转换成对外返回的路径字符串。

        项目根目录内的文件返回相对路径
        （例如 data/clips/event_001/frame_001.jpg），
        与任务书讲解中的示例保持一致；
        项目根目录之外的文件返回绝对路径。
        """

        try:

            return image_path.relative_to(
                _PROJECT_ROOT
            ).as_posix()

        except ValueError:

            return str(image_path)


# ============================================================
# 统一创建入口（对齐项目其它模块的 create_xxx 风格）
# ============================================================

def create_video_buffer(
    buffer_duration: float = 10.0,
    clips_dir: Optional[Union[str, Path]] = None,
) -> VideoBuffer:
    """统一创建 VideoBuffer。"""

    return VideoBuffer(
        buffer_duration=buffer_duration,
        clips_dir=clips_dir,
    )
