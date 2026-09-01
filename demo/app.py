"""AI 学习视频智能记忆系统 —— 界面

分工：E 王彦如 —— UI + Replay + README + Demo
本文件：demo/app.py —— Day 1：搭 UI 框架，使用假数据，不接后端。

视觉设计：白底 + 蓝色点缀，内联 SVG 图标（Lucide 风格），卡片化布局。
运行语义（与团队对齐）：
    - 视频源（摄像头 / 本地文件）统一作为「帧流」处理
    - 「检测到」（YOLO）逐帧实时变 → AI Analysis 上段
    - 「语义理解」（Qwen-VL）事件触发才变 → AI Analysis 下段
    - Current Event / 时间线由事件引擎随播放进度逐步生成

后续接线点（Day 2~7）：
    - LIVE VIDEO  ← C(检测追踪)：frame + bbox
    - Current Event ← A(事件引擎)：event_type / track_id / time / confidence
    - 搜索结果 / 回放 ← D(记忆检索)：event / timestamp / caption / video 路径
"""

import gradio as gr

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

/* 时间线 */
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
"""


# ============ 事件类型中文映射 ============

EVENT_LABELS = {
    "sit_at_study_position": "坐到学习位置",
    "leave_study_position": "离开学习位置",
    "study_preparation": "学习准备",
    "start_study": "开始学习",
    "end_study": "结束学习",
    "reading": "阅读",
    "writing": "书写",
    "phone_learning": "手机学习",
    "computer_learning": "电脑学习",
    "other_study_behavior": "其他学习行为",
    "phone_distraction": "手机分心",
    "computer_distraction": "电脑分心",
    "communication_distraction": "交流分心",
    "study_end_cleanup": "学习结束整理",
}


def event_label(event_type: str) -> str:
    return EVENT_LABELS.get(event_type, event_type)


# ============ HTML 片段（假数据） ============

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


def current_event_html() -> str:
    return (
        '<div class="event-card">'
        '<div class="event-name">手机分心</div>'
        '<div class="event-meta">phone_distraction · #1 · 置信度 0.91</div>'
        '<div class="event-time">10:32:15 → 10:32:23</div>'
        '</div>'
    )


def timeline_html() -> str:
    items = [
        ("10:28:05", "writing", False),
        ("10:30:11", "reading", False),
        ("10:32:15", "phone_distraction", True),
    ]
    rows = []
    for t, etype, is_current in items:
        cls = "current" if is_current else "past"
        tag = '<span class="timeline-tag">当前</span>' if is_current else ""
        rows.append(
            '<div class="timeline-item">'
            f'<span class="timeline-dot {cls}"></span>'
            f'<span class="timeline-time">{t}</span>'
            f'<span class="timeline-label">{event_label(etype)}{tag}</span>'
            '</div>'
        )
    return '<div class="timeline">' + "".join(rows) + "</div>"


def analysis_html() -> str:
    """AI Analysis：上段「实时检测」（逐帧变）+ 下段「语义理解」（事件触发）。"""
    return (
        '<div class="card">'
        '<div class="analysis-section">实时检测<span class="badge live">LIVE</span></div>'
        '<div class="analysis-detect">person ×1 · cell phone ×1 · book ×1</div>'
        '<div class="analysis-section">语义理解<span class="badge trigger">事件触发</span></div>'
        '<div class="analysis-caption">画面中一名学生正低头查看手机，桌面上有学习资料。</div>'
        '</div>'
    )


# 整块结果列表（列：时间戳 / 事件 / 描述 / 视频片段）
FAKE_RESULTS = [
    ["10:32:15", "手机分心", "学生拿起手机并查看手机内容", "segment_632.mp4"],
    ["09:14:02", "书写", "学生正在书写", "segment_140.mp4"],
    ["09:05:30", "开始学习", "学生开始学习", "segment_055.mp4"],
]

RESULT_HEADERS = ["时间戳", "事件", "描述", "视频片段"]


# ============ 回调（Day 1 假逻辑，后续接 A/D） ============

def on_search(query: str):
    """Day 1 返回固定假结果；Day 5 替换为调用 D 的检索接口。"""
    return FAKE_RESULTS


def on_select(evt: gr.SelectData):
    """点击结果列表某行时，记录选中行的视频片段路径。"""
    if evt.row_value is not None:
        return evt.row_value[-1]
    return None


def on_replay(seg_path):
    """回放选中片段（Day 6 接 D 返回的视频片段路径，用播放器播）。"""
    if not seg_path:
        gr.Warning("请先在结果列表中点击选中一条记录")
    else:
        gr.Info(f"回放功能将在 Day 6 接入，选中片段：{seg_path}")


def on_start() -> str:
    return status_html("run")


def on_stop() -> str:
    return status_html("stop")


# ============ Gradio 界面 ============

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI 学习记忆") as demo:
        gr.HTML("<style>" + CSS + "</style>")
        gr.HTML(app_header())

        with gr.Row():
            # ---- 左列（宽）：实时画面 + 控制 + 分析 ----
            with gr.Column(scale=3):
                gr.HTML(section_title("video", "LIVE VIDEO 实时画面"))

                with gr.Tabs():
                    with gr.Tab("实时摄像头"):
                        gr.Image(sources=["webcam"], streaming=True, label=None)
                    with gr.Tab("本地视频文件"):
                        gr.Video(sources=["upload"], label=None)

                with gr.Row():
                    start_btn = gr.Button("开始分析", variant="primary",
                                          elem_classes=["btn-primary"])
                    stop_btn = gr.Button("停止分析",
                                         elem_classes=["btn-secondary"])
                status = gr.HTML(status_html("idle"))

                gr.HTML(section_title("activity", "AI Analysis 分析"))
                gr.HTML(analysis_html())

            # ---- 右列（窄）：当前事件 + 时间线 ----
            with gr.Column(scale=1):
                gr.HTML(section_title("zap", "当前事件 Current Event"))
                gr.HTML(current_event_html())

                gr.HTML(section_title("clock", "近期事件时间线"))
                gr.HTML(timeline_html())

        # ---- 搜索区 ----
        gr.HTML(section_title("search", "问问你的学习记忆"))
        with gr.Row():
            query = gr.Textbox(
                placeholder="例如：什么时候有人玩手机？",
                label=None,
                scale=4,
            )
            search_btn = gr.Button("搜索", scale=1, variant="primary",
                                   elem_classes=["btn-primary"])

        # ---- 结果区（整块列表）----
        gr.HTML(section_title("play", "搜索结果"))
        gr.HTML(
            '<div class="hint">点击表格中任意一行选中该条结果，'
            '再点击「回放选中片段」播放对应的视频片段。</div>'
        )
        results = gr.Dataframe(
            headers=RESULT_HEADERS,
            value=FAKE_RESULTS,
            interactive=True,
            wrap=True,
        )
        selected_row = gr.State(None)
        replay_btn = gr.Button("回放选中片段", elem_classes=["btn-primary"])

        # ---- 事件绑定 ----
        start_btn.click(on_start, outputs=status)
        stop_btn.click(on_stop, outputs=status)
        search_btn.click(on_search, inputs=query, outputs=results)
        query.submit(on_search, inputs=query, outputs=results)
        results.select(on_select, outputs=selected_row)
        replay_btn.click(on_replay, inputs=selected_row)

    return demo


if __name__ == "__main__":
    build_ui().launch()
