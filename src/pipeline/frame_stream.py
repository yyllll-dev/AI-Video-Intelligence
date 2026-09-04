"""统一逐帧操作循环 —— 让实时摄像头与长视频文件"逐帧"驱动 VideoBuffer

本文件：src/pipeline/frame_stream.py

背景：
    VideoBuffer 需要"每一帧"持续写入才能保留最近 N 秒历史画面。
    真实输入有两种：
        1. 实时摄像头：无限持续输出，需要随时可以停止；
        2. 长视频文件：有结束（read() 返回 None 后自然停止）。
    本项目 src/detection/video_source.py 已经用统一接口
    open_source() / VideoSource.read() -> VideoFrame
    屏蔽了这两种来源的差异。

    本模块在这个统一接口之上，
    提供"逐帧取出 + 逐帧操作 + 逐帧喂入 VideoBuffer"的循环：

        FrameStream(source)
            ↓  read() 逐帧取出 VideoFrame
            ↓  1. 自动写入 VideoBuffer（保留最近 N 秒）
            ↓  2. 调用 on_frame(video_frame)（对每一帧做自定义操作）
            ↓
        实时摄像头：持续运行，直到 stop() / Ctrl+C / max_frames；
        视频文件：读到结尾自动结束。

典型用法（实时闭环）：
    stream = FrameStream(
        source=0,                  # 0 = 本机摄像头；也可以是视频文件路径
        buffer=buffer,             # VideoBuffer：自动保留最近 N 秒
        on_frame=on_frame,         # 每帧回调：可做检测 / 事件判断 ...
    )
    stats = stream.run()           # 摄像头：Ctrl+C 停止

    在 on_frame 里，一旦 EventEngine 报出事件，
    可以立刻通过 stream.buffer.get_frames(...) 取回该事件的历史关键帧。

设计说明：
    1. 本模块只负责"逐帧取出 + 逐帧回调"，不做检测 / 跟踪 / 事件判断，
       也不直接调用 Qwen-VL，保持单一职责。
    2. 视频文件默认全速逐帧处理（适合长视频快速回放）；
       若希望按真实帧率节奏播放，可设置 target_fps。
    3. 摄像头读取是阻塞式的，stop() 会在读到下一帧时生效。
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Union

from ..detection.video_source import (
    VideoFrame,
    VideoSource,
    open_source,
)


# ============================================================
# 统一逐帧操作循环
# ============================================================

class FrameStream:
    """从实时摄像头或视频文件中逐帧取出画面并逐帧操作。

    输入：
        source：
            - int：本地摄像头编号（0 = 第一个摄像头）
            - str：视频文件路径
            - VideoSource：已打开的帧源（CameraSource / VideoFileSource）

    每帧处理顺序：
        1. 若指定了 buffer，先把该帧写入 VideoBuffer；
        2. 若指定了 on_frame，再调用 on_frame(video_frame)，
           让回调可以立即基于刚写入的画面做判断。

    输出：
        run() 返回处理统计 dict：
            {"frames", "start_time", "end_time", "duration"}

    说明：
        1. 本模块负责释放自己创建的帧源；
           外部传入的 VideoSource 仍由调用方负责释放。
        2. 摄像头模式没有"结束"概念，
           通过 stop() / Ctrl+C / max_frames 停止。
    """

    def __init__(
        self,
        source: Union[int, str, VideoSource],
        buffer=None,
        on_frame: Optional[Callable[[VideoFrame], None]] = None,
        target_fps: Optional[float] = None,
    ):
        """创建逐帧操作循环。

        参数：
            source：
                视频输入来源：
                int=摄像头编号、str=视频文件路径、
                VideoSource=已打开的帧源。

            buffer：
                可选的 VideoBuffer。
                指定后每一帧都会自动写入缓冲，
                使缓冲始终保留最近 N 秒画面。

            on_frame：
                可选的逐帧回调，签名 on_frame(video_frame)。
                每一帧被取出后都会调用一次，
                可用于检测、事件判断或其它逐帧处理。

            target_fps：
                可选。仅对视频文件生效：
                设置后按指定帧率节奏逐帧（模拟实时播放）；
                不设置（默认）则全速逐帧，适合长视频快速处理。
                摄像头来源会自动忽略此参数（按其真实帧率运行）。
        """

        # ----------------------------------------------------
        # 1. 打开或接收帧源
        # ----------------------------------------------------

        if isinstance(source, VideoSource):

            self._source = source

            # 外部传入的源由调用方负责释放
            self._owns_source = False

        else:

            # int(摄像头) / str(视频文件) 统一由 open_source 打开
            self._source = open_source(source)

            # 本模块自己打开的源由本模块负责释放
            self._owns_source = True

        # ----------------------------------------------------
        # 2. 每帧操作配置
        # ----------------------------------------------------

        self.buffer = buffer

        self.on_frame = on_frame

        self._target_fps = (
            float(target_fps)
            if target_fps is not None and target_fps > 0
            else None
        )

        # 是否为摄像头来源（摄像头忽略 target_fps）
        self._is_camera = isinstance(source, int) or (
            type(self._source).__name__ == "CameraSource"
        )

        # ----------------------------------------------------
        # 3. 运行状态
        # ----------------------------------------------------

        # 已处理帧数
        self.processed = 0

        # 停止标志：stop() 或 Ctrl+C 时置 True
        self._stopped = False

        # 是否已释放
        self._closed = False

    # ========================================================
    # 运行控制
    # ========================================================

    def stop(self) -> None:
        """请求停止循环。

        摄像头读取是阻塞式的，
        该标志会在读到下一帧后生效；
        也可以在另一个线程中调用。
        """

        self._stopped = True

    @property
    def stopped(self) -> bool:
        """当前是否已请求停止。"""

        return self._stopped

    # ========================================================
    # 逐帧处理
    # ========================================================

    def step(self) -> Optional[VideoFrame]:
        """取出并处理一帧。

        返回：
            处理完成的 VideoFrame；
            视频结束 / 已请求停止 / 已释放时返回 None。

        处理顺序：
            1. 先写入 buffer（若配置了 buffer）；
            2. 再调用 on_frame（若配置了回调）。
        """

        if self._closed:
            return None

        if self._stopped:
            return None

        video_frame = self._source.read()

        if video_frame is None:
            return None

        # ----------------------------------------------------
        # 1. 先写入 VideoBuffer
        # ----------------------------------------------------

        if self.buffer is not None:

            self.buffer.add_frame(
                frame=video_frame.frame,
                timestamp=video_frame.timestamp,
            )

        self.processed += 1

        # ----------------------------------------------------
        # 2. 再调用逐帧回调
        # ----------------------------------------------------

        if self.on_frame is not None:

            self.on_frame(video_frame)

        return video_frame

    def run(
        self,
        max_frames: Optional[int] = None,
        progress_every: int = 0,
    ) -> Dict[str, float]:
        """逐帧运行，直到结束 / 停止 / 达到 max_frames。

        参数：
            max_frames：
                最多处理多少帧。
                摄像头模式建议设置（例如录制 10 秒），
                也可以不设置，用 Ctrl+C 停止。

            progress_every：
                每隔多少帧打印一次进度；
                0 表示不打印。

        返回：
            统计 dict：
                frames     已处理帧数
                start_time 第一帧时间戳（秒）
                end_time   最后一帧时间戳（秒）
                duration   首末帧时间跨度（秒）

        行为：
            - 视频文件：读到结尾（read() 返回 None）自动结束；
            - 实时摄像头：持续运行，
              直到 max_frames 达到、stop() 被调用或 Ctrl+C。
        """

        self._stopped = False

        start_wall = time.monotonic()

        first_timestamp: Optional[float] = None
        last_timestamp: Optional[float] = None

        try:

            while True:

                # 达到最大帧数则停止
                if (
                    max_frames is not None
                    and self.processed >= max_frames
                ):
                    break

                video_frame = self.step()

                # 视频结束 / 已停止
                if video_frame is None:
                    break

                if first_timestamp is None:
                    first_timestamp = video_frame.timestamp

                last_timestamp = video_frame.timestamp

                # 进度打印
                if (
                    progress_every > 0
                    and self.processed % progress_every == 0
                ):
                    print(
                        f"[FrameStream] 已处理 {self.processed} 帧，"
                        f"当前时间 {video_frame.timestamp:.2f}s"
                    )

                # 视频文件按 target_fps 节奏播放（模拟实时）
                if (
                    self._target_fps is not None
                    and not self._is_camera
                ):
                    time.sleep(1.0 / self._target_fps)

        except KeyboardInterrupt:

            print(
                "\n[FrameStream] 收到 Ctrl+C，停止运行。"
            )

        finally:

            # 释放本模块自己打开的帧源
            self.close()

        return {
            "frames": float(self.processed),
            "start_time": (
                first_timestamp if first_timestamp is not None else 0.0
            ),
            "end_time": (
                last_timestamp if last_timestamp is not None else 0.0
            ),
            "duration": (
                last_timestamp - first_timestamp
                if (
                    first_timestamp is not None
                    and last_timestamp is not None
                )
                else 0.0
            ),
        }

    # ========================================================
    # 资源管理
    # ========================================================

    def close(self) -> None:
        """释放帧源资源。

        只释放本模块自己打开的帧源；
        外部传入的 VideoSource 仍由调用方负责释放。
        """

        if self._closed:
            return

        self._closed = True

        if self._owns_source:

            self._source.release()

    def __enter__(self) -> "FrameStream":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
