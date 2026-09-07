# AI-Video-Intelligence
Real-time local video intelligence system for Intel AI PC

## Current pipeline

`VideoPipeline` now connects detector output, optional tracking, and the event
engine. A detector returns dictionaries with `class_name`, `confidence`, and
`bbox`; it may also include `track_id`. If a tracker is supplied, it receives
the normalized detections and the frame timestamp. Every call to
`process_frame(frame, timestamp)` returns the detections, `TrackingResult`
objects, and newly emitted events.

Run the complete entry point from the repository root:

```powershell
python -m src.main --source path\to\video.mp4 --qwen-model-path path\to\Qwen2-VL-2B-Instruct --yolo-device 0
```

排查全链路时增加 `--trace`：

```powershell
python -m src.main --source path\to\video.mp4 --qwen-model-path path\to\Qwen2-VL-2B-Instruct --yolo-device 0 --trace
```

日志按 `[TRACE][FRAME]`、`[TRACE][YOLO_OUTPUT]`、
`[TRACE][TRACKER_OUTPUT]`、`[TRACE][EVENT_OUTPUT]`、
`[TRACE][EVENT_TO_VLM]`、`[TRACE][VLM_KEYFRAMES]`、
`[TRACE][VLM_NORMALIZED_OUTPUT]`、`[TRACE][EVENT_AFTER_VLM]` 和
`[TRACE][MEMORY_OUTPUT]` 标记。注意，YOLO 只处理由 `--analysis-fps`
选中的分析帧；这里的“逐帧”指每一张实际送入 YOLO 的帧。

网页流程默认输出紧凑的窗口日志：`[分析窗口]` 给出起止秒数和 Event
候选；随后逐项列出关键帧编号、对应秒数、最近的 YOLO 源帧及目标；
`[VLM判断]` 只显示原始/校验后的 true 事件与分段，`[VLM最终]` 显示
最终事件、判定依据和短描述。VLM 的完整原始文本不再打印到终端，解析失败
时仅显示输出长度和抢救出的去重描述。

Use `--source 0` for the first camera. Use `--no-vlm` only when checking the
YOLO/Tracking/Event half of the pipeline without loading Qwen2-VL. Keyframes
are written to `data/clips/` and the final JSON report is written to
`data/outputs/`.

To use the visual demo, set `QWEN_VL_MODEL_PATH` and run:

```powershell
python demo/app.py
```

Upload a video under "本地视频文件", then click "开始分析". The event cards and
natural-language search use the real runtime after analysis completes.

## Long videos and replay

- Uploaded files are sampled directly from the original video by timestamp, so
  long events do not depend on the 30-second in-memory buffer.
- Camera input is compressed into one-minute MP4 chunks under `data/recordings/`.
- Camera capture/recording runs independently from YOLO and Qwen inference, so
  a slow semantic analysis does not pause the archived video.
- Confirmed short events get a complete replay clip under `data/replays/`.
- Events longer than 60 seconds get a three-part highlight containing their
  beginning, middle, and end instead of duplicating the entire long recording.

The visual demo can analyze either an uploaded video or the computer's first
camera. Camera analysis continues until the Stop button is clicked; after the
remaining queued analysis finishes, the page explicitly shows that analysis
has ended.

At the end of each run, the already loaded Qwen model combines the confirmed
event sequence and every event caption into one factual whole-video summary.
The presentation UI only exposes the input controls, this summary, the event
timeline, natural-language search, and replay. Selecting a row in either the
timeline or the search results immediately opens that event's replay clip.

The event engine accepts a timestamp even for an empty frame, so absence-based
events continue to advance when the detector finds no objects.

## Event / VLM 判定原则

- 桌椅只用于建立学习位置；会话开始后，桌椅漏检视为“证据未知”，不会清空状态。
- 人物完全漏检持续 3 秒才确认离开；人物仍可见但明确远离桌椅时需持续 6 秒。
- Event Engine 每个语义窗口只提供弱候选；VLM 对八个正式事件逐项判断，结构化 `events` 是最终分类依据。
- VLM 先客观描述，再判断八类，最后在不改变画面事实的前提下自然融入已确认事件名。
- 正式事件只有入座、离座、阅读、书写、使用手机、使用电脑、交流分心和其他；没有旧名称映射。
- 学习开始和结束只作为 EventEngine 内部会话状态，不进入 VLM JSON、Memory 或页面时间轴。
- 未检测到具体物体的帧不再给“其他”投票；物体候选按全部分析帧计算出现比例，只作为平等弱线索，不向 VLM 预设事件结论。
- `other_behavior` 同时表示无法分类的兜底，以及拿出、收起、整理、摆放学习用品等明确过渡行为。它可以和具体行为出现在同一窗口的不重叠 `activity_segments` 中。
- 正常阅读过程中的短暂翻页仍归阅读；从非阅读状态拿书、寻找页码再开始阅读时，前面的准备阶段归其他。电脑、手机和书写采用相同的“持续使用”和“拿取整理”区分。
- `activity_segments` 保留真实起止帧和重复动作。物体线索较弱且与时序上下文冲突的 8 秒窗口会被拆成两个短窗口复核；不足 2 秒且没有后续确认的尾部弱切换归其他。
- VLM 或关键帧处理失败的候选会进入 `rejected_events`，不会无记录地消失。
- 摄像头模式将 YOLO/Event 与 VLM 放在两个串行阶段中；事件产生时立即固化关键帧，慢速 VLM 不会阻塞前端状态机。
- 页面只合并同视频、同类型且真正相邻的窗口；Tracker 临时换 ID 不会切断展示，中间出现其他事件或离座边界时不会跨越合并。
