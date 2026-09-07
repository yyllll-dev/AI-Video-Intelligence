"""面向评委展示的 AI 学习视频记忆界面。"""

import os
import sys
import html
import threading
import time
from datetime import datetime
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
from src.retrieval import match_event_type, merge_events_for_display

from core import (
    event_label,
)
from core.ui_state import load_ui_state, save_ui_state
from core.utf8_tee import configure_utf8_tee

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
  --alm-brand: #2563eb;
  --alm-brand-strong: #1d4ed8;
  --alm-brand-soft: #eff6ff;
  --alm-ink: #0f172a;
  --alm-muted: #475569;
  --alm-border: #cbd5e1;
  --alm-page: #f7f9fc;
  --alm-card: #ffffff;
}

.gradio-container {
  background: var(--alm-page) !important;
  color: var(--alm-ink) !important;
  max-width: 1240px !important;
  padding: 28px 32px !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
               "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
}

/* 顶部标题 */
.app-header {
  display: flex; align-items: center; gap: 14px;
  padding-bottom: 20px; margin-bottom: 6px;
  border-bottom: 1px solid var(--alm-border);
}
.app-logo {
  width: 44px; height: 44px; border-radius: 12px;
  background: linear-gradient(135deg, #2563eb, #60a5fa);
  color: #fff; display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 14px rgba(37, 99, 235, .28);
}
.app-title { font-size: 22px; font-weight: 700; color: var(--alm-ink) !important; letter-spacing: .2px; }
.app-subtitle { font-size: 13px; color: var(--alm-muted) !important; margin-top: 2px; }

/* 区标题 */
.section-title {
  display: flex; align-items: center; gap: 9px;
  font-size: 15px; font-weight: 600; color: var(--alm-ink) !important;
  margin: 22px 0 10px;
}
.section-title .bar { width: 3px; height: 16px; border-radius: 2px; background: var(--alm-brand); }
.section-title svg { color: var(--alm-brand); flex-shrink: 0; }

/* 卡片 */
.card {
  background: var(--alm-card);
  border: 1px solid var(--alm-border);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}

/* 当前事件卡片 */
.event-card {
  background: var(--alm-card);
  border: 1px solid var(--alm-border);
  border-left: 3px solid var(--alm-brand);
  border-radius: 10px;
  padding: 14px;
}
.event-name { font-size: 16px; font-weight: 700; color: var(--alm-ink) !important; }
.event-meta { color: var(--alm-muted) !important; font-size: 13px; margin-top: 4px; }
.event-time { color: var(--alm-ink) !important; font-size: 13px; margin-top: 8px;
              font-variant-numeric: tabular-nums; }

/* 历史事件记录列表 */
.timeline { padding: 2px 0; }
.timeline-item { display: flex; align-items: flex-start; gap: 9px;
                 padding: 6px 0; font-size: 13px; color: var(--alm-ink) !important; }
.timeline-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.timeline-dot.current { background: var(--alm-brand); box-shadow: 0 0 0 3px var(--alm-brand-soft); }
.timeline-dot.past { background: #cbd5e1; }
.timeline-time { font-variant-numeric: tabular-nums; color: var(--alm-muted) !important; }
.timeline-label { color: var(--alm-ink) !important; }
.timeline-tag { font-size: 11px; color: var(--alm-brand); margin-left: 4px; }
.timeline-content { display: flex; flex-direction: column; gap: 3px; }
.timeline-caption { color: var(--alm-muted) !important; font-size: 12px; line-height: 1.5; }

/* 状态 */
.status-line { display: flex; align-items: center; gap: 8px;
               font-size: 13px; color: var(--alm-muted) !important; margin: 10px 2px 2px; }
.status-dot { width: 9px; height: 9px; border-radius: 50%; }
.status-dot.idle { background: #cbd5e1; }
.status-dot.run { background: #22c55e; box-shadow: 0 0 0 3px rgba(34, 197, 94, .2); }
.status-dot.stop { background: #94a3b8; }
.status-dot.ending { background: #f59e0b; }
.status-dot.done { background: #22c55e; }

/* AI Analysis 卡片内部 */
.analysis-section { font-size: 12px; font-weight: 600; color: var(--alm-muted) !important;
                    letter-spacing: .4px; margin-top: 10px; }
.analysis-section:first-child { margin-top: 0; }
.analysis-detect { font-size: 13px; color: var(--alm-ink) !important; margin-top: 6px;
                   font-variant-numeric: tabular-nums; }
.analysis-caption { font-size: 14px; color: var(--alm-ink) !important; margin-top: 6px; line-height: 1.6; }
.badge { display: inline-block; font-size: 10px; font-weight: 600; border-radius: 4px;
         padding: 1px 6px; margin-left: 6px; vertical-align: 1px; }
.badge.live { background: var(--alm-brand-soft); color: var(--alm-brand); }
.badge.trigger { background: #f1f5f9; color: var(--alm-muted); }

/* 按钮 */
button.btn-primary {
  background: var(--alm-brand) !important;
  border: 1px solid var(--alm-brand) !important;
  color: #fff !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}
button.btn-primary:hover { background: var(--alm-brand-strong) !important; }
button.btn-secondary {
  background: #fff !important;
  border: 1px solid var(--alm-brand) !important;
  color: var(--alm-brand) !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}
button.btn-secondary:hover { background: var(--alm-brand-soft) !important; }

/* 结果表格表头 */
.table-wrap th { background: var(--alm-brand-soft) !important; color: var(--alm-brand-strong) !important; }

/* 提示文案 */
.hint { font-size: 12px; color: var(--alm-muted) !important; margin: -2px 0 10px 2px; }

.panel-card {
  background: #fff; border: 1px solid var(--alm-border); border-radius: 16px;
  padding: 18px !important; box-shadow: 0 8px 28px rgba(15, 23, 42, .06);
}
.step-kicker {
  display: inline-flex; align-items: center; gap: 7px; color: var(--alm-brand);
  font-size: 12px; font-weight: 700; letter-spacing: .5px; margin-bottom: 4px;
}
.step-title { color: var(--alm-ink) !important; font-size: 18px; font-weight: 700; margin-bottom: 4px; }
.step-desc { color: var(--alm-muted) !important; font-size: 13px; line-height: 1.6; margin-bottom: 14px; }
.memory-guide {
  padding: 13px 15px; border-radius: 10px; background: var(--alm-brand-soft);
  color: #1e40af; font-size: 13px; line-height: 1.6; margin-bottom: 12px;
}
.empty-note { color: var(--alm-muted) !important; font-size: 13px; padding: 14px 0; }
"""


PRESENTATION_CSS = """
.gradio-container { max-width: 1180px !important; padding: 30px 28px 64px !important;
  background:radial-gradient(circle at 8% 0%,rgba(124,58,237,.16),transparent 31%),
  radial-gradient(circle at 94% 9%,rgba(14,165,233,.15),transparent 29%),
  linear-gradient(145deg,#f8faff 0%,#f5f3ff 48%,#f0f9ff 100%) !important; }
.app-header { justify-content:space-between !important; padding:18px 4px 26px !important;
  border-bottom:1px solid transparent !important;
  border-image:linear-gradient(90deg,rgba(99,102,241,.38),rgba(14,165,233,.18),transparent) 1 !important; }
.brand-lockup { display:flex; align-items:center; gap:14px; }
.app-logo { width:46px !important; height:46px !important; border-radius:15px !important;
  background:linear-gradient(135deg,#7c3aed 0%,#4f46e5 48%,#0ea5e9 100%) !important;
  box-shadow:0 13px 32px rgba(79,70,229,.30); }
.app-title { font-size:24px !important; font-weight:800 !important; letter-spacing:-.5px;
  background:linear-gradient(90deg,#312e81,#5b21b6 55%,#0369a1); color:transparent !important;
  -webkit-background-clip:text; background-clip:text; }
.judge-pill { padding:8px 12px; border:1px solid rgba(79,70,229,.18); border-radius:999px;
  background:linear-gradient(135deg,rgba(238,242,255,.96),rgba(240,249,255,.92));
  color:#4338ca; font-size:12px; font-weight:700; box-shadow:0 7px 20px rgba(79,70,229,.09); }
.panel-card { border:1px solid rgba(199,210,254,.72) !important; border-radius:20px !important;
  padding:22px !important; background:linear-gradient(145deg,rgba(255,255,255,.97),rgba(248,250,255,.93)) !important;
  box-shadow:0 18px 48px rgba(49,46,129,.08),inset 0 1px 0 rgba(255,255,255,.9) !important;
  transition:box-shadow .22s ease,border-color .22s ease !important; }
.panel-card:hover { border-color:rgba(165,180,252,.9) !important;
  box-shadow:0 21px 54px rgba(49,46,129,.11),inset 0 1px 0 #fff !important; }
.section-head { margin-bottom:15px; }
.section-eyebrow { display:inline-block; padding:4px 8px; border-radius:999px;
  background:linear-gradient(90deg,#ede9fe,#e0f2fe); color:#4f46e5; font-size:10px; font-weight:800; letter-spacing:1.4px; }
.section-title-new { margin-top:4px; color:#0f172a; font-size:19px; font-weight:800; }
.section-desc-new { margin-top:5px; color:#64748b; font-size:13px; line-height:1.6; }
.summary-box { min-height:128px; padding:20px 22px; border-radius:16px; border:1px solid #dbeafe;
  background:linear-gradient(135deg,#eef2ff 0%,#f5f3ff 48%,#ecfeff 100%);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.85); color:#1e293b; font-size:15px; line-height:1.9; }
.summary-empty { color:#94a3b8; }
.tab-nav { padding:4px !important; border-radius:13px !important; background:linear-gradient(90deg,#eef2ff,#f0f9ff) !important; }
.tab-nav button { border-radius:10px !important; color:#64748b !important; font-weight:700 !important; }
.tab-nav button.selected { color:#4338ca !important; background:linear-gradient(135deg,#fff,#f5f3ff) !important;
  box-shadow:0 5px 15px rgba(79,70,229,.12) !important; }
button.primary, button.btn-primary { border:0 !important; color:#fff !important;
  background:linear-gradient(100deg,#7c3aed,#4f46e5 52%,#0284c7) !important;
  box-shadow:0 9px 22px rgba(79,70,229,.22) !important; transition:transform .18s ease,box-shadow .18s ease !important; }
button.primary:hover, button.btn-primary:hover { transform:translateY(-1px); box-shadow:0 12px 28px rgba(79,70,229,.29) !important; }
button.secondary { border-color:#c7d2fe !important; color:#4338ca !important; background:linear-gradient(135deg,#fff,#f5f3ff) !important; }
button.secondary:hover { border-color:#818cf8 !important; background:linear-gradient(135deg,#f5f3ff,#eff6ff) !important; }
textarea:focus, input:focus { border-color:#818cf8 !important; box-shadow:0 0 0 3px rgba(99,102,241,.12) !important; }
.compact-status .status-line { display:inline-flex; padding:7px 11px; border:1px solid #e0e7ff;
  border-radius:999px; background:linear-gradient(90deg,rgba(238,242,255,.9),rgba(240,249,255,.9)); }
.panel-card .prose { color:#475569 !important; }
.media-frame { width:min(100%,800px) !important; height:450px !important; margin:0 auto !important;
  aspect-ratio:16/9 !important; border:1px solid #c7d2fe !important;
  border-radius:14px !important; overflow:hidden !important; background:#fff !important;
  box-shadow:0 12px 32px rgba(30,64,175,.10) !important; }
.media-frame > div { height:100% !important; background:#fff !important; }
.media-frame video,
.media-frame img { display:block !important; width:100% !important; height:100% !important;
  object-fit:cover !important; vertical-align:top !important; }
.media-frame button[aria-label*="Trim"],
.media-frame button[aria-label*="trim"],
.media-frame button[aria-label*="剪辑"],
.media-frame button[aria-label*="Reset"],
.media-frame button[aria-label*="reset"],
.media-frame button[aria-label*="重新"],
.media-frame button[title*="Trim"],
.media-frame button[title*="Reset"] { display:none !important; }
.browser-camera-shell { position:relative; width:min(100%,800px); aspect-ratio:16/9; margin:0 auto;
  overflow:hidden; border:1px solid #c7d2fe; border-radius:14px; background:#0f172a;
  box-shadow:0 12px 32px rgba(30,64,175,.10); }
.browser-camera-shell video { width:100%; height:100%; display:block; object-fit:cover; background:#0f172a; }
.browser-camera-empty { position:absolute; inset:0; display:grid; place-items:center; color:#94a3b8;
  font-size:14px; pointer-events:none; }
.compact-status { min-height:0 !important; padding:0 !important; }
.event-list { border:0 !important; background:transparent !important; }
.event-list fieldset { display:grid !important; grid-template-columns:minmax(0,1fr) !important; gap:10px !important; width:100% !important; }
.event-list label { display:block !important; margin:0 0 10px !important; padding:14px 16px !important;
  border:1px solid #dbeafe !important; border-radius:14px !important;
  background:linear-gradient(135deg,#fff,#f8faff) !important;
  box-shadow:0 5px 16px rgba(49,46,129,.05); cursor:pointer !important;
  width:100% !important; min-width:0 !important; box-sizing:border-box !important; }
.event-list label:hover { border-color:#a5b4fc !important; background:linear-gradient(135deg,#f5f3ff,#eff6ff) !important;
  transform:translateY(-1px); }
.event-list label:has(input:checked) { border-color:#6366f1 !important;
  background:linear-gradient(135deg,#ede9fe,#e0f2fe) !important; box-shadow:0 7px 20px rgba(79,70,229,.13); }
.event-list input,
.event-list input[type="radio"] { display:none !important; appearance:none !important; width:0 !important; height:0 !important; margin:0 !important; }
.event-list label span { display:block !important; white-space:pre-line !important; color:#64748b !important;
  font-size:12px !important; line-height:1.65 !important; }
.event-list label span::first-line { color:#172554 !important; font-size:17px !important; font-weight:800 !important; line-height:1.9 !important; }
.event-list label::before,
.event-list label::after,
.event-list label span::before,
.event-list label span::after,
.event-list label .radio,
.event-list label .check,
.event-list label .checkmark { display:none !important; content:none !important; }
.toast-title { display:none !important; }
#replay-modal { position:fixed !important; inset:0 !important; z-index:9999 !important;
  padding:0 !important; background:rgba(15,23,42,.62) !important; backdrop-filter:blur(5px); }
#replay-dialog { position:absolute !important; left:50% !important; top:50% !important;
  transform:translate(-50%,-50%) !important; width:min(800px,calc(100vw - 32px)) !important;
  max-height:calc(100vh - 32px) !important; margin:0 !important; padding:8px !important; gap:6px !important;
  border:1px solid #bfdbfe !important; border-radius:14px !important;
  background:linear-gradient(135deg,#eef2ff,#eaf6ff) !important;
  box-shadow:0 24px 70px rgba(15,23,42,.38) !important; overflow:hidden !important; }
#replay-dialog > .form { gap:6px !important; }
#replay-dialog .section-head { margin:0 !important; }
#replay-dialog .section-eyebrow,
#replay-dialog .section-desc-new { display:none !important; }
#replay-dialog .section-title-new { margin:0 !important; padding-left:4px; font-size:17px !important; }
#replay-dialog button { min-width:64px !important; max-width:64px !important; min-height:32px !important; }
#replay-dialog video { display:block !important; width:100% !important; max-height:calc(100vh - 94px) !important;
  object-fit:contain !important; border-radius:9px !important; background:#000 !important; }
#replay-dialog [data-testid="video"] { margin:0 !important; padding:0 !important; border:0 !important; }
footer { display:none !important; }
"""

CAMERA_OPEN_JS = """async () => {
  const video = document.getElementById('browser-camera-video');
  const shell = document.getElementById('browser-camera-shell');
  const empty = document.getElementById('browser-camera-empty');
  if (!video || !shell) return [];
  shell.style.display = 'block';
  const analysisFrame = document.querySelector('.analysis-camera-frame');
  if (analysisFrame) analysisFrame.style.display = 'none';
  if (window.__almCameraStream) {
    video.srcObject = window.__almCameraStream;
    if (empty) empty.style.display = 'none';
    return [];
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
    window.__almCameraStream = stream;
    video.srcObject = stream;
    if (empty) empty.style.display = 'none';
  } catch (error) {
    if (empty) {
      empty.style.display = 'grid';
      empty.textContent = '无法打开摄像头，请检查浏览器权限';
    }
  }
  return [];
}"""

CAMERA_START_JS = """(mode, videoPath) => {
  if (window.__almCameraStream) {
    window.__almCameraStream.getTracks().forEach(track => track.stop());
    window.__almCameraStream = null;
  }
  const shell = document.getElementById('browser-camera-shell');
  if (shell) shell.style.display = 'none';
  return [mode, videoPath];
}"""

CAMERA_CLOSE_JS = """() => {
  if (window.__almCameraStream) {
    window.__almCameraStream.getTracks().forEach(track => track.stop());
    window.__almCameraStream = null;
  }
  return [];
}"""


# ============ HTML 片段 ============

def app_header() -> str:
    return (
        '<div class="app-header">'
        '<div class="brand-lockup">'
        f'<div class="app-logo">{icon("sparkles", 22, "#fff")}</div>'
        '<div>'
        '<div class="app-title">视界先知——VisionOracle</div>'
        '<div class="app-subtitle">从视频事件到可检索的学习行为记忆</div>'
        '</div>'
        '</div>'
        '<div class="judge-pill">Qwen2-VL · Local Intelligence</div>'
        '</div>'
    )


def status_html(state: str) -> str:
    label = {
        "idle": "待分析",
        "run": "运行中",
        "ending": "正在结束分析",
        "stop": "已停止",
        "done": "分析结束",
    }[state]
    return (
        '<div class="status-line">'
        f'<span class="status-dot {state}"></span><span>{label}</span>'
        '</div>'
    )


_runtime_runner = None
_analysis_running = False
UI_STATE_PATH = Path(PROJECT_ROOT) / "data" / "ui_state.json"


# ============ 回调（后续接 A/D 真实模块） ============

def section_head(eyebrow: str, title: str, description: str) -> str:
    return (
        '<div class="section-head">'
        f'<div class="section-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="section-title-new">{html.escape(title)}</div>'
        f'<div class="section-desc-new">{html.escape(description)}</div>'
        '</div>'
    )


def summary_html(summary: str = "") -> str:
    content = html.escape(summary.strip()) if summary.strip() else (
        '<span class="summary-empty">分析完成后，这里将生成覆盖全部事件的综合描述。</span>'
    )
    return f'<div class="summary-box">{content}</div>'


def _event_choices(records):
    return [
        (
            f"{event_label(item['event_type'])}\n"
            f"{item['start_time']:.2f}s – {item['end_time']:.2f}s\n"
            f"{item['caption']}",
            str(item.get("video_path", "")),
        )
        for item in records
    ]


def on_search(query: str):
    if _runtime_runner is None:
        gr.Warning("请先上传视频并完成分析")
        return (
            gr.update(choices=[], value=None, visible=False),
            gr.update(value="请先完成视频分析。", visible=True),
        )
    found = merge_events_for_display(
        _runtime_runner.search(query or "刚才发生了什么？", top_k=50)
    )
    for item in found:
        if item["merged_event_count"] > 1:
            item["video_path"] = _runtime_runner.create_replay(
                item["start_time"], item["end_time"]
            )
    choices = _event_choices(found)
    requested_type = match_event_type(query or "")
    if requested_type is None:
        feedback = "未识别到明确的事件类型，以下结果按内容相关度排序。"
    elif choices:
        feedback = (
            f"本次视频中发生 {len(choices)} 个“{event_label(requested_type)}”事件。"
        )
    else:
        feedback = f"本次视频中没有发生“{event_label(requested_type)}”事件。"
    return (
        gr.update(choices=choices, value=None, visible=bool(choices)),
        gr.update(value=feedback, visible=True),
    )


def on_select_event(seg_path):
    """时间线和搜索结果共用：点击事件卡片即播放。"""
    if not seg_path:
        gr.Warning("该事件没有可用的回放片段")
        return gr.update(visible=False), gr.update(visible=False)
    return gr.update(visible=True), gr.update(value=seg_path, visible=True)


def on_close_replay():
    return (
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None),
        gr.update(value=None),
    )


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


def _persist_ui_state(state: str, **values) -> None:
    save_ui_state(
        UI_STATE_PATH,
        {
            "state": state,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **values,
        },
    )


def on_restore():
    """页面刷新或点击恢复按钮时，恢复最近一次分析状态。"""
    saved = load_ui_state(UI_STATE_PATH)
    unchanged = gr.update()
    if not saved:
        return (unchanged,) * 6

    video_path = saved.get("video_path")
    video_update = (
        gr.update(value=video_path)
        if isinstance(video_path, str) and Path(video_path).is_file()
        else unchanged
    )
    state = saved.get("state")
    if state == "completed":
        ready = bool(saved.get("ready"))
        return (
            saved.get("status_html", status_html("done")),
            summary_html(str(saved.get("video_summary", ""))),
            gr.update(
                choices=saved.get("timeline_choices", []),
                value=None,
                visible=bool(saved.get("timeline_choices", [])),
            ),
            gr.update(
                interactive=ready,
                placeholder=(
                    "例如：刚才什么时候阅读了？"
                    if ready else "最近一次分析没有产生可检索事件"
                ),
            ),
            gr.update(interactive=ready),
            video_update,
        )
    if state == "running":
        return (
            status_html("run"), summary_html("正在分析中，请稍后……"),
            gr.update(choices=[], value=None, visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False), video_update,
        )
    if state == "error":
        message = str(saved.get("message", "最近一次分析失败"))
        return (
            status_html("stop"), summary_html(f"分析失败：{message}"),
            gr.update(choices=[], value=None, visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False), video_update,
        )
    return (unchanged,) * 6


def _preview_frame():
    if _runtime_runner is None or _runtime_runner.current_frame is None:
        return gr.update()
    frame = _runtime_runner.current_frame.frame
    return gr.update(value=frame[:, :, ::-1].copy(), visible=True)


def on_start(input_mode, video_path):
    global _runtime_runner, _analysis_running
    if _analysis_running:
        gr.Warning("当前分析尚未结束")
        yield (gr.update(),) * 8
        return
    if input_mode == "上传视频" and not video_path:
        gr.Warning("请先上传一个本地视频文件")
        yield (
            status_html("idle"), summary_html(),
            gr.update(choices=[], value=None, visible=False),
            gr.update(interactive=False), gr.update(interactive=False), gr.update(),
            gr.update(interactive=True), gr.update(interactive=True),
        )
        return

    gr.Info("首次加载 Qwen2-VL 需要一些时间")
    source = 0 if input_mode == "本机摄像头" else str(video_path)
    _persist_ui_state(
        "running",
        input_mode=input_mode,
        video_path=(str(video_path) if video_path else None),
    )
    _runtime_runner = EndToEndRunner(
        source=source,
        qwen_model_path=os.getenv("QWEN_VL_MODEL_PATH"),
        yolo_device=os.getenv("YOLO_DEVICE", "cpu"),
        yolo_confidence=float(os.getenv("YOLO_CONFIDENCE", "0.35")),
    )
    _analysis_running = True
    yield (
        status_html("run"), summary_html("正在分析中，请稍后……"),
        gr.update(choices=[], value=None, visible=False),
        gr.update(interactive=False), gr.update(interactive=False),
        gr.update(value=None),
        gr.update(interactive=False), gr.update(interactive=False),
    )
    result_box = {}
    error_box = {}

    def run_analysis():
        try:
            result_box["result"] = _runtime_runner.run()
        except Exception as exc:
            error_box["error"] = exc

    analysis_thread = threading.Thread(
        target=run_analysis,
        name="ui-video-analysis",
        daemon=True,
    )
    analysis_thread.start()
    while analysis_thread.is_alive():
        run_state = "ending" if _runtime_runner._stop_event.is_set() else "run"
        yield (
            status_html(run_state), summary_html("正在分析中，请稍后……"), gr.update(),
            gr.update(interactive=False), gr.update(interactive=False),
            _preview_frame(),
            gr.update(interactive=False), gr.update(interactive=False),
        )
        time.sleep(0.25)
    analysis_thread.join()
    _analysis_running = False
    if "error" in error_box:
        exc = error_box["error"]
        _persist_ui_state(
            "error",
            input_mode=input_mode,
            video_path=(str(video_path) if video_path else None),
            message=str(exc),
        )
        yield (
            status_html("stop"), summary_html(f"分析失败：{exc}"),
            gr.update(choices=[], value=None, visible=False),
            gr.update(interactive=False), gr.update(interactive=False),
            _preview_frame(),
            gr.update(interactive=True), gr.update(interactive=True),
        )
        return
    result = result_box["result"]
    records = merge_events_for_display(_runtime_runner.memory_store.list_all())
    for record in records:
        if record["merged_event_count"] > 1:
            record["video_path"] = _runtime_runner.create_replay(
                record["start_time"], record["end_time"]
            )

    timeline_choices = _event_choices(records)
    video_summary = str(result.get("video_summary", "")).strip()
    ready = bool(records)
    completed_status = status_html("done")
    _persist_ui_state(
        "completed",
        input_mode=input_mode,
        video_path=(str(video_path) if video_path else None),
        ready=ready,
        status_html=completed_status,
        video_summary=video_summary,
        timeline_choices=timeline_choices,
    )
    print(
        f"[分析完成] 处理 {int(result['stream']['frames'])} 帧 | "
        f"写入 {result['memories_saved']} 条记忆 | "
        f"错误 {len(result['errors'])} 个 | 页面状态已保存到 {UI_STATE_PATH}"
    )
    yield (
        completed_status, summary_html(video_summary),
        gr.update(
            choices=timeline_choices,
            value=None,
            visible=bool(timeline_choices),
        ),
        gr.update(
            interactive=ready,
            placeholder=(
                "例如：刚才什么时候阅读了？"
                if ready else "本次分析没有产生可检索的确认事件"
            ),
        ),
        gr.update(interactive=ready),
        _preview_frame(),
        gr.update(interactive=True), gr.update(interactive=True),
    )


def on_stop() -> str:
    if _analysis_running and _runtime_runner is not None:
        _runtime_runner.stop()
        gr.Info("已请求停止，正在完成剩余分析")
        return status_html("ending")
    return status_html("idle")


# ============ Gradio 界面 ============

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="视界先知——VisionOracle",
    ) as demo:
        gr.HTML(app_header())

        with gr.Column(elem_classes=["panel-card"]):
            gr.HTML(section_head("INPUT", "视频输入", "上传本地视频，或切换到实时摄像头。"))
            upload_mode = gr.State("上传视频")
            camera_mode = gr.State("本机摄像头")
            empty_video = gr.State(None)
            with gr.Tabs():
                with gr.Tab("上传视频") as upload_tab:
                    video_input = gr.Video(
                        sources=["upload"], label=None, height=450,
                        elem_classes=["media-frame"],
                    )
                    upload_start_btn = gr.Button(
                        "开始分析", variant="primary", elem_classes=["primary"]
                    )
                with gr.Tab("实时摄像头") as camera_tab:
                    gr.HTML(
                        '<div class="browser-camera-shell" id="browser-camera-shell">'
                        '<video id="browser-camera-video" autoplay muted playsinline></video>'
                        '<div class="browser-camera-empty" id="browser-camera-empty">正在打开摄像头…</div>'
                        '</div>'
                    )
                    live_preview = gr.Image(
                        label=None, interactive=False, height=450, visible=False,
                        elem_classes=["media-frame", "analysis-camera-frame"],
                    )
                    camera_start_btn = gr.Button(
                        "开始分析", variant="primary", elem_classes=["primary"]
                    )
            with gr.Row(equal_height=True):
                stop_btn = gr.Button("停止", elem_classes=["secondary"])
                restore_btn = gr.Button("恢复最近结果", elem_classes=["secondary"])
                status = gr.HTML(status_html("idle"), elem_classes=["compact-status"])

        with gr.Column(elem_classes=["panel-card"]):
            gr.HTML(section_head("OVERVIEW", "全事件总结", "千问综合全部已确认事件与逐事件描述生成。"))
            summary_panel = gr.HTML(summary_html())

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["panel-card"]):
                gr.HTML(section_head("TIMELINE", "具体事件时间线", "点击事件即可打开对应回放。"))
                timeline_list = gr.Radio(
                    choices=[], value=None, label=None, show_label=False, visible=False,
                    elem_classes=["event-list"],
                )

            with gr.Column(scale=1, elem_classes=["panel-card"]):
                gr.HTML(section_head("SEARCH", "自然语言搜索", "例如：什么时候阅读了？"))
                with gr.Row():
                    query = gr.Textbox(
                        placeholder="请先完成视频分析", label=None, show_label=False,
                        interactive=False, scale=5,
                    )
                    search_btn = gr.Button(
                        "搜索", scale=1, variant="primary",
                        interactive=False, elem_classes=["primary"],
                    )
                search_feedback = gr.Markdown(value="", visible=False)
                search_results = gr.Radio(
                    choices=[], value=None, label=None, show_label=False, visible=False,
                    elem_classes=["event-list"],
                )

        with gr.Group(visible=False, elem_id="replay-modal") as replay_modal:
            with gr.Column(elem_id="replay-dialog"):
                with gr.Row():
                    gr.HTML(section_head("REPLAY", "事件回放", "当前选中事件的视频片段"))
                    close_replay_btn = gr.Button("关闭", size="sm", elem_classes=["secondary"])
                replay_video = gr.Video(label=None, interactive=False, visible=False)
        # ---- 事件绑定 ----
        video_input.upload(on_video_upload, inputs=video_input, outputs=video_input, show_progress="hidden")
        start_outputs = [
            status, summary_panel, timeline_list,
            query, search_btn, live_preview,
            upload_start_btn, camera_start_btn,
        ]
        upload_start_btn.click(
            on_start,
            inputs=[upload_mode, video_input],
            outputs=start_outputs,
            show_progress="hidden",
        )
        camera_tab.select(
            fn=None,
            js=CAMERA_OPEN_JS,
            show_progress="hidden",
        )
        camera_start_btn.click(
            on_start,
            inputs=[camera_mode, empty_video],
            outputs=start_outputs,
            js=CAMERA_START_JS,
            show_progress="hidden",
        )
        upload_tab.select(
            on_stop,
            outputs=status,
            queue=False,
            js=CAMERA_CLOSE_JS,
            show_progress="hidden",
        )
        stop_btn.click(on_stop, outputs=status, queue=False, show_progress="hidden")
        search_btn.click(
            on_search,
            inputs=query,
            outputs=[search_results, search_feedback],
            show_progress="hidden",
        )
        query.submit(
            on_search,
            inputs=query,
            outputs=[search_results, search_feedback],
            show_progress="hidden",
        )
        timeline_list.input(
            on_select_event,
            inputs=timeline_list,
            outputs=[replay_modal, replay_video],
            show_progress="hidden",
        )
        search_results.input(
            on_select_event,
            inputs=search_results,
            outputs=[replay_modal, replay_video],
            show_progress="hidden",
        )
        close_replay_btn.click(
            on_close_replay,
            outputs=[replay_modal, replay_video, timeline_list, search_results],
            show_progress="hidden",
            queue=False,
        )
        restore_outputs = [
            status, summary_panel, timeline_list,
            query, search_btn, video_input,
        ]
        restore_btn.click(on_restore, outputs=restore_outputs, show_progress="hidden")

    return demo


if __name__ == "__main__":
    configure_utf8_tee(Path(PROJECT_ROOT) / "result.txt")
    print("[日志] 终端输出同步写入 UTF-8 result.txt")
    build_ui().launch(
        allowed_paths=[str(Path(PROJECT_ROOT) / "data")],
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CSS + PRESENTATION_CSS,
    )
