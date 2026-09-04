"""AI 学习视频智能记忆系统 —— 界面

分工：E 王彦如 —— UI + Replay + README + Demo
本文件：demo/app.py —— 界面主文件，调用 demo/core/ 里的事件映射与模拟数据。

视觉设计：白底 + 蓝色点缀，内联 SVG 图标（Lucide 风格），卡片化布局。

展示链路（对齐团队「明日任务安排」PDF 的验收标准）：
    视频/画面 → 当前检测到的人 → 当前事件 → 事件时间 → VLM 分析结果
    → 历史事件记录列表 → 简单自然语言检索

当前状态：UI 结构就位，数据来自 demo/core/mock_data.py（假数据）。
后续接入：把 mock_data 里的函数替换成对 src/ 真实模块的调用即可，UI 无需改动。
"""

import os
import sys
from pathlib import Path

# 确保项目根目录和 demo/ 都能被直接运行的脚本导入
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEMO_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DEMO_DIR)

import huggingface_hub

if not hasattr(huggingface_hub, "HfFolder"):
    class HfFolder:
        get_token = staticmethod(huggingface_hub.get_token)

    huggingface_hub.HfFolder = HfFolder

import gradio as gr

from src.pipeline.runtime import EndToEndRunner
from src.pipeline.media_archive import prepare_browser_video
from src.retrieval import merge_events_for_display

from core import (
    event_label,
    class_label,
)

# ============ SVG 图标（Lucide 风格，内联 stroke 图标） ============

ICONS = {
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',
    "video": '<path d="m22 8-6 4 6 4V8Z"/><rect x="2" y="6" width="14" height="12" rx="2"/>',
    "camera": '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
}


def icon(name: str, size: int = 18, color: str = "currentColor") -> str:
    body = ICONS[name]
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )


# ============ 全局样式（白底 + 蓝色点缀） ============

CSS = """
:root {
  --brand: #2563eb;
  --brand-strong: #1d4ed8;
  --brand-soft: #eff6ff;
  --ink: #0f172a;
  --muted: #64748b;
  --border: #e2e8f0;
  --page: #f7f9fc;
  --card: #ffffff;
}

.gradio-container {
  background: var(--page) !important;
  max-width: 1240px !important;
  padding: 28px 32px !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
               "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}

/* 顶部标题 */
.app-header {
  display: flex; align-items: center; gap: 14px;
  padding-bottom: 20px; margin-bottom: 6px;
  border-bottom: 1px solid var(--border);
}
.app-logo {
  width: 44px; height: 44px; border-radius: 12px;
  background: linear-gradient(135deg, #2563eb, #60a5fa);
  color: #fff; display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 14px rgba(37, 99, 235, .28);
}
.app-title { font-size: 22px; font-weight: 700; color: var(--ink); letter-spacing: .2px; }
.app-subtitle { font-size: 13px; color: var(--muted); margin-top: 2px; }

/* 区标题 */
.section-title {
  display: flex; align-items: center; gap: 9px;
  font-size: 15px; font-weight: 600; color: var(--ink);
  margin: 22px 0 10px;
}
.section-title .bar { width: 3px; height: 16px; border-radius: 2px; background: var(--brand); }
.section-title svg { color: var(--brand); flex-shrink: 0; }

/* 卡片 */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}

/* 当前事件卡片 */
.event-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--brand);
  border-radius: 10px;
  padding: 14px;
}
.event-name { font-size: 16px; font-weight: 700; color: var(--ink); }
.event-meta { color: var(--muted); font-size: 13px; margin-top: 4px; }
.event-time { color: var(--ink); font-size: 13px; margin-top: 8px;
              font-variant-numeric: tabular-nums; }

/* 历史事件记录列表 */
.timeline { padding: 2px 0; }
.timeline-item { display: flex; align-items: flex-start; gap: 9px;
                 padding: 6px 0; font-size: 13px; color: var(--ink); }
.timeline-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.timeline-dot.current { background: var(--brand); box-shadow: 0 0 0 3px var(--brand-soft); }
.timeline-dot.past { background: #cbd5e1; }
.timeline-time { font-variant-numeric: tabular-nums; color: var(--muted); }
.timeline-label { color: var(--ink); }
.timeline-tag { font-size: 11px; color: var(--brand); margin-left: 4px; }

/* 状态 */
.status-line { display: flex; align-items: center; gap: 8px;
               font-size: 13px; color: var(--muted); margin: 10px 2px 2px; }
.status-dot { width: 9px; height: 9px; border-radius: 50%; }
.status-dot.idle { background: #cbd5e1; }
.status-dot.run { background: #22c55e; box-shadow: 0 0 0 3px rgba(34, 197, 94, .2); }
.status-dot.stop { background: #94a3b8; }

/* AI Analysis 卡片内部 */
.analysis-section { font-size: 12px; font-weight: 600; color: var(--muted);
                    letter-spacing: .4px; margin-top: 10px; }
.analysis-section:first-child { margin-top: 0; }
.analysis-detect { font-size: 13px; color: var(--ink); margin-top: 6px;
                   font-variant-numeric: tabular-nums; }
.analysis-caption { font-size: 14px; color: var(--ink); margin-top: 6px; line-height: 1.6; }
.badge { display: inline-block; font-size: 10px; font-weight: 600; border-radius: 4px;
         padding: 1px 6px; margin-left: 6px; vertical-align: 1px; }
.badge.live { background: var(--brand-soft); color: var(--brand); }
.badge.trigger { background: #f1f5f9; color: var(--muted); }

/* 按钮 */
button.btn-primary {
  background: var(--brand) !important;
  border: 1px solid var(--brand) !important;
  color: #fff !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}
button.btn-primary:hover { background: var(--brand-strong) !important; }
button.btn-secondary {
  background: #fff !important;
  border: 1px solid var(--brand) !important;
  color: var(--brand) !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}
button.btn-secondary:hover { background: var(--brand-soft) !important; }

/* 结果表格表头 */
.table-wrap th { background: var(--brand-soft) !important; color: var(--brand-strong) !important; }

/* 提示文案 */
.hint { font-size: 12px; color: var(--muted); margin: -2px 0 10px 2px; }

.panel-card {
  background: #fff; border: 1px solid var(--border); border-radius: 16px;
  padding: 18px !important; box-shadow: 0 8px 28px rgba(15, 23, 42, .06);
}
.step-kicker {
  display: inline-flex; align-items: center; gap: 7px; color: var(--brand);
  font-size: 12px; font-weight: 700; letter-spacing: .5px; margin-bottom: 4px;
}
.step-title { color: var(--ink); font-size: 18px; font-weight: 700; margin-bottom: 4px; }
.step-desc { color: var(--muted); font-size: 13px; line-height: 1.6; margin-bottom: 14px; }
.memory-guide {
  padding: 13px 15px; border-radius: 10px; background: var(--brand-soft);
  color: #1e40af; font-size: 13px; line-height: 1.6; margin-bottom: 12px;
}
.empty-note { color: var(--muted); font-size: 13px; padding: 14px 0; }
"""


# ============ HTML 片段 ============

def app_header() -> str:
    return (
        '<div class="app-header">'
        f'<div class="app-logo">{icon("sparkles", 22, "#fff")}</div>'
        '<div>'
        '<div class="app-title">AI 学习记忆</div>'
        '<div class="app-subtitle">AI Learning Memory · 实时视频智能分析</div>'
        '</div>'
        '</div>'
    )


def section_title(icon_name: str, text: str) -> str:
    return (
        f'<div class="section-title"><span class="bar"></span>'
        f'{icon(icon_name)}<span>{text}</span></div>'
    )


def status_html(state: str) -> str:
    label = {"idle": "待机", "run": "运行中", "stop": "已停止"}[state]
    return (
        '<div class="status-line">'
        f'<span class="status-dot {state}"></span><span>{label}</span>'
        '</div>'
    )


def detection_summary() -> str:
    return "尚未开始分析"


def current_event_html() -> str:
    return (
        '<div class="event-card">'
        '<div class="event-name">暂无事件</div>'
        '<div class="event-meta">上传视频并开始分析后显示</div>'
        '</div>'
    )


def timeline_html() -> str:
    return '<div class="timeline"><span class="event-meta">暂无历史事件</span></div>'


def analysis_html() -> str:
    return (
        '<div class="card">'
        '<div class="analysis-section">目标检测<span class="badge live">等待中</span></div>'
        f'<div class="analysis-detect">{detection_summary()}</div>'
        '<div class="analysis-section">语义理解<span class="badge trigger">事件触发</span></div>'
        '<div class="analysis-caption">分析完成后显示 Qwen2-VL 描述</div>'
        '</div>'
    )


# 整块结果列表（列：时间戳 / 事件 / 描述 / 视频片段）
RESULT_HEADERS = ["时间戳", "事件", "描述", "回放视频"]

_runtime_runner = None


# ============ 回调（后续接 A/D 真实模块） ============

def on_search(query: str):
    if _runtime_runner is None:
        gr.Warning("请先上传视频并完成分析")
        return gr.update(value=[], visible=False), gr.update(visible=False)
    found = merge_events_for_display(
        _runtime_runner.search(query or "刚才发生了什么？", top_k=50)
    )
    for item in found:
        if item["merged_event_count"] > 1:
            item["video_path"] = _runtime_runner.create_replay(
                item["start_time"], item["end_time"]
            )
    rows = [
        [
            f"{item['start_time']:.2f}s - {item['end_time']:.2f}s",
            event_label(item["event_type"]),
            item["caption"],
            item["video_path"],
        ]
        for item in found
    ]
    return gr.update(value=rows, visible=True), gr.update(visible=bool(rows))


def on_select(evt: gr.SelectData):
    """点击结果列表某行时，记录选中行的视频片段路径。"""
    if evt.row_value is not None:
        return evt.row_value[-1]
    return None


def on_replay(seg_path):
    if not seg_path:
        gr.Warning("请先在结果列表中点击选中一条记录")
        return gr.update(visible=False)
    return gr.update(value=seg_path, visible=True)


def on_video_upload(video_path):
    if not video_path:
        return None
    try:
        return prepare_browser_video(
            video_path,
            Path(PROJECT_ROOT) / "data" / "previews",
        )
    except Exception as exc:
        gr.Warning(f"视频预览转换失败：{exc}")
        return video_path


def on_start(input_mode, video_path, camera_seconds):
    global _runtime_runner
    if input_mode == "上传视频" and not video_path:
        gr.Warning("请先上传一个本地视频文件")
        return (
            status_html("idle"), analysis_html(), current_event_html(), timeline_html(),
            gr.update(interactive=False), gr.update(interactive=False),
        )

    gr.Info("正在运行完整分析，首次加载 Qwen2-VL 需要一些时间")
    source = 0 if input_mode == "本机摄像头" else str(video_path)
    _runtime_runner = EndToEndRunner(
        source=source,
        qwen_model_path=os.getenv("QWEN_VL_MODEL_PATH"),
        yolo_device=os.getenv("YOLO_DEVICE", "cpu"),
        yolo_confidence=float(os.getenv("YOLO_CONFIDENCE", "0.35")),
    )
    result = _runtime_runner.run(
        max_duration=float(camera_seconds) if input_mode == "本机摄像头" else None
    )
    records = merge_events_for_display(_runtime_runner.memory_store.list_all())
    for record in records:
        if record["merged_event_count"] > 1:
            record["video_path"] = _runtime_runner.create_replay(
                record["start_time"], record["end_time"]
            )

    detections = _runtime_runner.latest_result.get("detections", [])
    counts = {}
    for detection in detections:
        name = class_label(detection.class_name)
        counts[name] = counts.get(name, 0) + 1
    detected_text = " · ".join(f"{name} ×{count}" for name, count in counts.items()) or "无"

    if records:
        latest = records[-1]
        current = (
            '<div class="event-card">'
            f'<div class="event-name">{event_label(latest["event_type"])}</div>'
            f'<div class="event-meta">{latest["event_type"]} · #{latest["track_id"]} · '
            f'置信度 {latest["confidence"]:.2f}</div>'
            f'<div class="event-time">{latest["start_time"]:.2f}s → {latest["end_time"]:.2f}s</div>'
            '</div>'
        )
        rows = [
            '<div class="timeline-item">'
            '<span class="timeline-dot past"></span>'
            f'<span class="timeline-time">{record["start_time"]:.2f}s - {record["end_time"]:.2f}s</span>'
            f'<span class="timeline-label">{event_label(record["event_type"])}</span>'
            '</div>'
            for record in records
        ]
        timeline = '<div class="timeline">' + "".join(rows) + "</div>"
        caption = latest["caption"]
    else:
        current = '<div class="event-card"><div class="event-name">未确认到事件</div></div>'
        timeline = '<div class="timeline">暂无正式记忆</div>'
        caption = "VLM 未确认候选事件，请查看终端日志和分析报告。"

    analysis = (
        '<div class="card">'
        '<div class="analysis-section">实时检测<span class="badge live">DONE</span></div>'
        f'<div class="analysis-detect">{detected_text}</div>'
        '<div class="analysis-section">语义理解<span class="badge trigger">Qwen2-VL</span></div>'
        f'<div class="analysis-caption">{caption}</div>'
        f'<div class="analysis-caption">处理 {int(result["stream"]["frames"])} 帧，'
        f'写入 {result["memories_saved"]} 条记忆，错误 {len(result["errors"])} 个。</div>'
        '</div>'
    )
    ready = bool(records)
    return (
        status_html("stop"), analysis, current, timeline,
        gr.update(
            interactive=ready,
            placeholder=(
                "例如：刚才什么时候阅读了？"
                if ready else "本次分析没有产生可检索的确认事件"
            ),
        ),
        gr.update(interactive=ready),
    )


def on_stop() -> str:
    if _runtime_runner is not None:
        _runtime_runner.stop()
    return status_html("stop")


def on_input_mode_change(mode: str):
    use_upload = mode == "上传视频"
    return (
        gr.update(visible=use_upload),
        gr.update(visible=not use_upload),
        gr.update(visible=not use_upload),
    )


# ============ Gradio 界面 ============

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI 学习记忆") as demo:
        gr.HTML("<style>" + CSS + "</style>")
        gr.HTML(app_header())

        with gr.Row():
            with gr.Column(scale=3, elem_classes=["panel-card"]):
                gr.HTML(
                    '<div class="step-kicker">步骤 1</div>'
                    '<div class="step-title">选择分析来源</div>'
                    '<div class="step-desc">上传已有视频，或使用本机摄像头进行限时实时分析。</div>'
                )
                input_mode = gr.Radio(
                    ["上传视频", "本机摄像头"], value="上传视频", label=None
                )
                video_input = gr.Video(
                    sources=["upload"], label="上传视频文件", visible=True
                )
                camera_hint = gr.HTML(
                    '<div class="memory-guide">系统将打开本机摄像头0。实时录像会分段保存，'
                    '确认后的事件会生成可播放回放。</div>',
                    visible=False,
                )
                camera_seconds = gr.Slider(
                    minimum=10, maximum=600, value=60, step=10,
                    label="分析时长（秒）", visible=False,
                )
                with gr.Row():
                    start_btn = gr.Button(
                        "开始分析", variant="primary", elem_classes=["btn-primary"]
                    )
                    stop_btn = gr.Button("停止", elem_classes=["btn-secondary"])
                status = gr.HTML(status_html("idle"))

            with gr.Column(scale=2, elem_classes=["panel-card"]):
                gr.HTML(
                    '<div class="step-kicker">步骤 2</div>'
                    '<div class="step-title">AI 分析摘要</div>'
                    '<div class="step-desc">完成后显示目标检测数量、Qwen描述和记忆条数。</div>'
                )
                analysis_panel = gr.HTML(analysis_html())

        with gr.Row():
            with gr.Column(scale=2, elem_classes=["panel-card"]):
                gr.HTML(
                    '<div class="step-kicker">步骤 3</div>'
                    '<div class="step-title">最近确认事件</div>'
                    '<div class="step-desc">只展示经过 Qwen2-VL 确认并写入记忆的事件。</div>'
                )
                current_event_panel = gr.HTML(current_event_html())
            with gr.Column(scale=3, elem_classes=["panel-card"]):
                gr.HTML(
                    '<div class="step-kicker">事件记录</div>'
                    '<div class="step-title">本次分析时间线</div>'
                    '<div class="step-desc">按视频时间排列本次分析确认的全部事件。</div>'
                )
                timeline_panel = gr.HTML(timeline_html())

        with gr.Column(elem_classes=["panel-card"]):
            gr.HTML(
                '<div class="step-kicker">步骤 4</div>'
                '<div class="step-title">搜索学习记忆</div>'
                '<div class="memory-guide">先完成视频分析，系统会把确认事件写入记忆。'
                '之后可以提问“什么时候阅读了？”或“什么时候离开座位？”。'
                '搜索结果只在点击搜索后显示。</div>'
            )
            with gr.Row():
                query = gr.Textbox(
                    placeholder="请先完成视频分析",
                    label=None,
                    show_label=False,
                    interactive=False,
                    scale=5,
                )
                search_btn = gr.Button(
                    "搜索记忆", scale=1, variant="primary",
                    interactive=False, elem_classes=["btn-primary"],
                )

            results = gr.Dataframe(
                headers=RESULT_HEADERS, value=[], interactive=False,
                wrap=True, visible=False,
            )
            selected_row = gr.State(None)
            replay_btn = gr.Button(
                "播放选中事件", visible=False, elem_classes=["btn-primary"]
            )
            replay_video = gr.Video(
                label="事件回放", interactive=False, visible=False
            )

        # ---- 事件绑定 ----
        input_mode.change(
            on_input_mode_change,
            inputs=input_mode,
            outputs=[video_input, camera_hint, camera_seconds],
        )
        video_input.upload(on_video_upload, inputs=video_input, outputs=video_input)
        start_btn.click(
            on_start,
            inputs=[input_mode, video_input, camera_seconds],
            outputs=[
                status, analysis_panel, current_event_panel, timeline_panel,
                query, search_btn,
            ],
        )
        stop_btn.click(on_stop, outputs=status)
        search_btn.click(on_search, inputs=query, outputs=[results, replay_btn])
        query.submit(on_search, inputs=query, outputs=[results, replay_btn])
        results.select(on_select, outputs=selected_row)
        replay_btn.click(on_replay, inputs=selected_row, outputs=replay_video)

    return demo


if __name__ == "__main__":
    build_ui().launch(
        allowed_paths=[str(Path(PROJECT_ROOT) / "data")],
    )
