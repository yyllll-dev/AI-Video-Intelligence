"""面向评委展示的 AI 学习视频记忆界面。"""

import os
import sys
import html
import re
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

from src.detection.detector import YoloDetector
from src.pipeline.runtime import EndToEndRunner
from src.pipeline.media_archive import prepare_browser_video
from src.retrieval import match_event_type, merge_events_for_display
from src.vlm.qwen_vlm import load_model

from core import (
    event_label,
)
from core.ui_state import load_ui_state, save_ui_state
from core.utf8_tee import configure_utf8_tee


# 展示页始终使用同一套浅色视觉。Gradio 6 会跟随系统深色模式读取
# ``*_dark`` 变量，因此普通和 dark 变量必须同时设置。
APP_THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate").set(
    body_background_fill="#f5f7ff",
    body_background_fill_dark="#f5f7ff",
    body_text_color="#0f172a",
    body_text_color_dark="#0f172a",
    body_text_color_subdued="#64748b",
    body_text_color_subdued_dark="#64748b",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f8faff",
    background_fill_secondary_dark="#f8faff",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#cbd5e1",
    block_border_color_dark="#cbd5e1",
    block_label_background_fill="#eef2ff",
    block_label_background_fill_dark="#eef2ff",
    block_label_text_color="#334155",
    block_label_text_color_dark="#334155",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_background_fill_focus="#ffffff",
    input_background_fill_focus_dark="#ffffff",
    input_border_color="#cbd5e1",
    input_border_color_dark="#cbd5e1",
    input_placeholder_color="#64748b",
    input_placeholder_color_dark="#64748b",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#ffffff",
    button_secondary_text_color="#4338ca",
    button_secondary_text_color_dark="#4338ca",
    checkbox_label_background_fill="#ffffff",
    checkbox_label_background_fill_dark="#ffffff",
    checkbox_label_background_fill_hover="#f5f3ff",
    checkbox_label_background_fill_hover_dark="#f5f3ff",
    checkbox_label_background_fill_selected="#eef2ff",
    checkbox_label_background_fill_selected_dark="#eef2ff",
    checkbox_label_text_color="#172554",
    checkbox_label_text_color_dark="#172554",
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
.status-dot.loading, .status-dot.starting { background: #f59e0b; }
.status-dot.run { background: #22c55e; box-shadow: 0 0 0 3px rgba(34, 197, 94, .2); }
.status-dot.recording { background: #ef4444; box-shadow: 0 0 0 3px rgba(239, 68, 68, .18); }
.status-dot.stop { background: #94a3b8; }
.status-dot.ending { background: #f59e0b; }
.status-dot.done { background: #22c55e; }
.status-dot.ready { background: #22c55e; }
.status-dot.error { background: #ef4444; }

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
html, body { min-height:100%; margin:0 !important; background:#f5f7ff !important; }
html { color-scheme:light !important; }
body { overflow-x:hidden; color:#0f172a !important; }
.gradio-container,
body > gradio-app .gradio-container,
gradio-app .gradio-container {
  width:min(1180px,calc(100% - 32px)) !important;
  max-width:1180px !important;
  min-width:0 !important;
  margin:0 auto !important;
  padding:30px 28px 64px !important;
  box-sizing:border-box !important;
  background:radial-gradient(circle at 8% 0%,rgba(124,58,237,.16),transparent 31%),
  radial-gradient(circle at 94% 9%,rgba(14,165,233,.15),transparent 29%),
  linear-gradient(145deg,#f8faff 0%,#f5f3ff 48%,#f0f9ff 100%) !important; }
.gradio-container > main,
.gradio-container > .main,
.gradio-container .contain { width:100% !important; max-width:none !important; margin-inline:auto !important; }
.app-header { justify-content:space-between !important; padding:18px 4px 26px !important;
  border-bottom:1px solid transparent !important;
  border-image:linear-gradient(90deg,rgba(99,102,241,.38),rgba(14,165,233,.18),transparent) 1 !important; }
.brand-lockup { display:flex; align-items:center; gap:14px; }
.app-logo { width:46px !important; height:46px !important; border-radius:15px !important;
  background:linear-gradient(135deg,#7c3aed 0%,#4f46e5 48%,#0ea5e9 100%) !important;
  box-shadow:0 13px 32px rgba(79,70,229,.30); }
.app-title { font-size:24px !important; font-weight:800 !important; letter-spacing:-.5px;
  color:#312e81 !important; -webkit-text-fill-color:#312e81 !important; }
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
.summary-layout { display:grid; grid-template-columns:minmax(0,1fr) 465px; gap:30px; align-items:center; }
.summary-layout.no-chart { grid-template-columns:minmax(0,1fr); }
.summary-copy { color:#1e293b; font-size:15px; line-height:1.95; }
.summary-chart { display:flex; align-items:center; justify-content:flex-end; gap:20px;
  padding-left:22px; border-left:1px solid rgba(99,102,241,.16); }
.summary-pie { position:relative; width:148px; height:148px; border-radius:50%; flex:0 0 148px;
  box-shadow:0 10px 28px rgba(49,46,129,.16); }
.summary-pie::after { content:""; position:absolute; inset:31px; border-radius:50%;
  background:#f8faff; box-shadow:inset 0 0 0 1px rgba(199,210,254,.72); }
.summary-pie-center { position:absolute; inset:31px; z-index:1; display:flex; flex-direction:column;
  align-items:center; justify-content:center; color:#64748b; font-size:11px; line-height:1.35; text-align:center; }
.summary-pie-center strong { color:#312e81; font-size:17px; }
.summary-legend { display:flex; min-width:255px; flex:1 1 auto; flex-direction:column; gap:7px; }
.summary-legend-item { display:grid;
  grid-template-columns:9px minmax(92px,1fr) 58px minmax(62px,auto); gap:7px;
  align-items:center; color:#475569; font-size:12px; line-height:1.35; }
.summary-legend-item > span { min-width:0; white-space:nowrap; text-align:left; }
.summary-legend-dot { width:9px; height:9px; border-radius:50%; }
.summary-legend-percentage, .summary-legend-duration { color:#312e81; font-weight:700;
  font-variant-numeric:tabular-nums; text-align:left !important; }
.tab-nav { padding:4px !important; border-radius:13px !important; background:linear-gradient(90deg,#eef2ff,#f0f9ff) !important; }
.tab-nav button { min-height:44px !important; border-radius:10px !important; color:#475569 !important;
  -webkit-text-fill-color:#475569 !important; font-size:15px !important;
  font-weight:750 !important; opacity:1 !important; }
.tab-nav button.selected, .tab-nav button[aria-selected="true"] {
  color:#4338ca !important; -webkit-text-fill-color:#4338ca !important;
  background:linear-gradient(135deg,#fff,#f5f3ff) !important;
  box-shadow:0 5px 15px rgba(79,70,229,.12) !important; }
button.primary, button.btn-primary { border:0 !important; color:#fff !important;
  background:linear-gradient(100deg,#7c3aed,#4f46e5 52%,#0284c7) !important;
  box-shadow:0 9px 22px rgba(79,70,229,.22) !important; transition:transform .18s ease,box-shadow .18s ease !important; }
button.primary:hover, button.btn-primary:hover { transform:translateY(-1px); box-shadow:0 12px 28px rgba(79,70,229,.29) !important; }
button.secondary { border-color:#c7d2fe !important; color:#4338ca !important; background:linear-gradient(135deg,#fff,#f5f3ff) !important; }
button.secondary:hover { border-color:#818cf8 !important; background:linear-gradient(135deg,#f5f3ff,#eff6ff) !important; }
/* Gradio 6 将 elem_id 放在组件外壳上，按钮文字又可能包在 span 中。
   同时覆盖外壳、button 和内部文字，避免主题升级后文字变成白色或透明。 */
#upload-start-btn button, button#upload-start-btn,
#camera-start-btn button, button#camera-start-btn,
#search-btn button, button#search-btn {
  color:#fff !important; -webkit-text-fill-color:#fff !important;
  background:linear-gradient(100deg,#7c3aed,#4f46e5 52%,#0284c7) !important;
  border:0 !important; font-weight:700 !important;
}
#upload-start-btn, #camera-start-btn, #stop-btn, #restore-btn {
  min-height:52px !important; height:52px !important;
}
#upload-start-btn button, button#upload-start-btn,
#camera-start-btn button, button#camera-start-btn,
#stop-btn button, button#stop-btn,
#restore-btn button, button#restore-btn {
  min-height:52px !important; height:52px !important; font-size:16px !important;
}
#stop-btn button, button#stop-btn,
#restore-btn button, button#restore-btn,
#close-replay-btn button, button#close-replay-btn {
  color:#4338ca !important; -webkit-text-fill-color:#4338ca !important;
  background:linear-gradient(135deg,#fff,#f5f3ff) !important;
  border:1px solid #c7d2fe !important; font-weight:700 !important;
}
#upload-start-btn button *, #camera-start-btn button *, #search-btn button *,
#stop-btn button *, #restore-btn button *, #close-replay-btn button *,
button#upload-start-btn *, button#camera-start-btn *, button#search-btn *,
button#stop-btn *, button#restore-btn *, button#close-replay-btn * {
  color:inherit !important; -webkit-text-fill-color:currentColor !important;
  opacity:1 !important; visibility:visible !important;
}
#upload-start-btn button:disabled, #camera-start-btn button:disabled, #search-btn button:disabled,
button#upload-start-btn:disabled, button#camera-start-btn:disabled, button#search-btn:disabled {
  color:#fff !important; -webkit-text-fill-color:#fff !important; opacity:.58 !important;
}
#search-btn { min-width:96px !important; min-height:46px !important; height:46px !important; }
#search-btn button, button#search-btn { min-height:46px !important; height:46px !important; font-size:15px !important; }
.search-panel, .search-panel.form, .search-panel > .form {
  justify-content:flex-start !important; align-content:flex-start !important;
  align-items:stretch !important;
}
.search-panel > * { flex-grow:0 !important; flex-shrink:0 !important; }
#search-controls { width:100% !important; flex:0 0 auto !important; align-self:stretch !important;
  align-items:flex-start !important; gap:12px !important; margin:0 !important; }
#search-query { min-height:46px !important; margin:0 !important; padding:0 !important;
  border:0 !important; outline:0 !important; background:transparent !important;
  box-shadow:none !important; }
#search-query > .form, #search-query .form {
  padding:0 !important; border:0 !important; outline:0 !important;
  background:transparent !important; box-shadow:none !important;
}
#search-query textarea, #search-query input {
  min-height:46px !important;
  padding-left:13px !important;
  color:#0f172a !important; -webkit-text-fill-color:#0f172a !important;
  background:#fff !important; border-color:#cbd5e1 !important;
  opacity:1 !important;
}
#search-query textarea::placeholder, #search-query input::placeholder {
  color:#64748b !important; -webkit-text-fill-color:#64748b !important;
  opacity:1 !important;
}
#search-feedback { flex:0 0 auto !important; min-height:0 !important;
  margin:0 !important; padding-left:2px !important; }
#search-feedback > .prose { margin:8px 0 0 !important; padding:0 !important; }
#search-result-list { flex:0 0 auto !important; align-self:stretch !important; margin-top:8px !important; }
textarea:focus, input:focus { border-color:#818cf8 !important; box-shadow:0 0 0 3px rgba(99,102,241,.12) !important; }
.compact-status .status-line { display:inline-flex; padding:7px 11px; border:1px solid #e0e7ff;
  border-radius:999px; background:linear-gradient(90deg,rgba(238,242,255,.9),rgba(240,249,255,.9));
  color:#334155 !important; -webkit-text-fill-color:#334155 !important;
  font-size:15px !important; font-weight:700 !important; }
.compact-status .status-line *, #status-display .status-line * {
  color:inherit !important; -webkit-text-fill-color:currentColor !important;
  opacity:1 !important; visibility:visible !important;
}
#status-display { min-width:180px !important; min-height:52px !important;
  display:flex !important; align-items:center !important; }
.panel-card .prose { color:#475569 !important; }
.media-frame, .browser-camera-shell {
  box-sizing:border-box !important; width:min(100%,800px) !important; height:450px !important;
  min-height:450px !important; aspect-ratio:16/9 !important; margin:0 auto !important;
  flex:0 0 auto !important; align-self:center !important;
  border:1px solid #c7d2fe !important; border-radius:14px !important;
  overflow:hidden !important; background:#fff !important;
  box-shadow:0 12px 32px rgba(30,64,175,.10) !important;
}
.media-frame > div { height:100% !important; background:#fff !important; }
.media-frame [data-testid="video"], .media-frame .upload-container,
.media-frame button[aria-label*="上传"], .media-frame button[aria-label*="upload" i] {
  background:#fff !important; color:#475569 !important;
  -webkit-text-fill-color:#475569 !important;
}
.media-frame p, .media-frame span:not(.icon), .media-frame button {
  color:#475569 !important; -webkit-text-fill-color:#475569 !important;
  opacity:1 !important;
}
.media-frame video,
.media-frame img { display:block !important; width:100% !important; height:100% !important;
  object-fit:cover !important; vertical-align:top !important; }
.media-frame video { accent-color:#0f172a !important; }
.media-frame video::-webkit-media-controls-panel {
  background:linear-gradient(to top,rgba(49,46,129,.88),rgba(79,70,229,.28)) !important;
}
.media-frame video::-webkit-media-controls-timeline {
  background-color:rgba(224,231,255,.72) !important; border-radius:999px !important;
}
.media-frame .controls input[type="range"] {
  -webkit-appearance:none !important; appearance:none !important;
  height:14px !important; min-height:14px !important;
  background:transparent !important; border:0 !important;
  color:#0f172a !important; accent-color:#0f172a !important;
  cursor:pointer !important;
}
.media-frame .controls input[type="range"]::-webkit-slider-runnable-track {
  height:4px !important; background:#cbd5e1 !important;
  border:0 !important; border-radius:999px !important;
}
.media-frame .controls input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance:none !important; appearance:none !important;
  width:12px !important; height:12px !important; margin-top:-4px !important;
  border:2px solid #fff !important; border-radius:50% !important;
  background:#0f172a !important; box-shadow:0 0 0 1px #0f172a !important;
}
.media-frame .controls input[type="range"]::-moz-range-track {
  height:4px !important; background:#cbd5e1 !important;
  border:0 !important; border-radius:999px !important;
}
.media-frame .controls input[type="range"]::-moz-range-progress {
  height:4px !important; background:#0f172a !important; border-radius:999px !important;
}
.media-frame .controls input[type="range"]::-moz-range-thumb {
  width:12px !important; height:12px !important;
  border:2px solid #fff !important; border-radius:50% !important;
  background:#0f172a !important; box-shadow:0 0 0 1px #0f172a !important;
}
.media-frame .icon-button-wrapper.top-panel {
  display:flex !important; flex-direction:row !important; align-items:center !important;
  width:auto !important; min-width:44px !important; height:44px !important;
  min-height:44px !important; padding:0 !important; --bg-color:transparent !important;
  background:transparent !important; border:0 !important; box-shadow:none !important;
}
.media-frame .icon-button-wrapper.top-panel > button,
.media-frame button[aria-label="清除"], .media-frame button[aria-label="Clear"] {
  display:flex !important; align-items:center !important; justify-content:center !important;
  width:44px !important; min-width:44px !important; height:44px !important;
  min-height:44px !important; padding:10px !important; --bg-color:transparent !important;
  background:transparent !important; border:0 !important; box-shadow:none !important;
}
.media-frame .icon-button-wrapper.top-panel > button:not([aria-label="清除"]):not([aria-label="Clear"]) {
  display:none !important;
}
.media-frame progress {
  height:6px !important; min-height:6px !important;
  color:#0f172a !important; accent-color:#0f172a !important;
  background:#fff !important; border:0 !important;
  border-radius:999px !important; overflow:hidden !important;
}
.media-frame progress::-webkit-progress-bar {
  background:#fff !important; border-radius:999px !important;
}
.media-frame progress::-webkit-progress-value {
  background:#0f172a !important; border-radius:999px !important;
}
.media-frame progress::-moz-progress-bar {
  background:#0f172a !important; border-radius:999px !important;
}
.media-frame .controls,
.media-frame .controls .inner {
  background:#fff !important;
  color:#0f172a !important;
}
.media-frame .controls {
  height:36px !important; min-height:36px !important;
  padding-top:2px !important; padding-bottom:2px !important;
  border-top:1px solid #e2e8f0 !important;
  box-shadow:none !important;
}
.media-frame .controls .inner {
  height:32px !important; min-height:32px !important;
  padding-top:0 !important; padding-bottom:0 !important;
}
.media-frame .controls button,
.media-frame .controls .icon,
.media-frame .controls .time,
.media-frame .controls span,
.media-frame .controls svg {
  color:#0f172a !important;
  -webkit-text-fill-color:#0f172a !important;
}
.media-frame .controls svg {
  stroke:currentColor !important;
}
.media-frame button[aria-label*="Trim"],
.media-frame button[aria-label*="trim"],
.media-frame button[aria-label*="剪辑"],
.media-frame button[aria-label*="Reset"],
.media-frame button[aria-label*="reset"],
.media-frame button[aria-label*="重新"],
.media-frame button[title*="Trim"],
.media-frame button[title*="Reset"] { display:none !important; }
.browser-camera-shell { position:relative; display:block; }
.camera-preview-host,
#camera-preview-host,
#camera-preview-host > div,
#camera-preview-host .prose {
  box-sizing:border-box !important; width:100% !important;
  margin:0 !important; padding:0 !important; gap:0 !important;
  border:0 !important; background:transparent !important; box-shadow:none !important;
}
.input-mode-stack {
  width:100% !important; margin:0 !important; padding:0 !important;
  gap:12px !important; align-items:stretch !important;
}
.input-mode-stack > *,
.input-mode-stack > .form,
.input-mode-stack button { margin-top:0 !important; margin-bottom:0 !important; }
.analysis-camera-frame {
  box-sizing:border-box !important; width:min(100%,800px) !important;
  height:450px !important; min-height:450px !important; aspect-ratio:16/9 !important;
  margin:0 auto !important; padding:0 !important; overflow:hidden !important;
}
.analysis-camera-frame > div,
.analysis-camera-frame [data-testid="image"] {
  box-sizing:border-box !important; width:100% !important; height:100% !important;
  min-height:0 !important; margin:0 !important; padding:0 !important; overflow:hidden !important;
}
.analysis-camera-frame .label-wrap,
.analysis-camera-frame .icon-button-wrapper.top-panel { display:none !important; }
.analysis-camera-frame img { width:100% !important; height:100% !important; object-fit:cover !important; }
gradio-app[data-input-locked="true"] .tab-nav button {
  pointer-events:none !important; cursor:not-allowed !important; opacity:1 !important;
}
.browser-camera-shell video { width:100%; height:100%; display:block; object-fit:cover; background:#f8faff; }
.browser-camera-shell canvas { position:absolute; inset:0; z-index:1; display:none;
  width:100%; height:100%; object-fit:cover; background:#f8faff; }
.browser-camera-empty { position:absolute; inset:0; display:grid; place-items:center; color:#475569 !important;
  -webkit-text-fill-color:#475569 !important; background:#f8faff; font-size:14px; font-weight:600;
  pointer-events:none; }
.compact-status { min-height:0 !important; padding:0 !important; }
.event-list, .event-list *, .event-list > div, .event-list > .form, .event-list .form,
.event-list .wrap, .event-list fieldset {
  border:0 !important; outline:0 !important; background:transparent !important;
  box-shadow:none !important;
}
.panel-card > .form:has(> .event-list),
#search-controls > .form {
  border:0 !important; background:transparent !important; box-shadow:none !important;
}
.event-list { padding:0 !important; margin:0 !important; }
.event-list fieldset { display:grid !important; grid-template-columns:minmax(0,1fr) !important;
  gap:10px !important; width:100% !important; padding:0 !important; margin:0 !important; }
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
  font-size:14px !important; font-weight:400 !important; line-height:1.75 !important; }
.event-list label span::first-line { color:#0f172a !important;
  -webkit-text-fill-color:#0f172a !important; font-size:17px !important;
  font-weight:800 !important; line-height:1.9 !important; }
.event-list label::before,
.event-list label::after,
.event-list label span::before,
.event-list label span::after,
.event-list label .radio,
.event-list label .check,
.event-list label .checkmark { display:none !important; content:none !important; }
.toast-title { display:none !important; }
#replay-modal { position:fixed !important; inset:0 !important; z-index:9999 !important;
  padding:0 !important; background:rgba(226,232,240,.84) !important; backdrop-filter:blur(5px); }
#replay-dialog { position:absolute !important; left:50% !important; top:50% !important;
  transform:translate(-50%,-50%) !important; width:min(800px,calc(100vw - 32px)) !important;
  max-height:calc(100vh - 32px) !important; margin:0 !important; padding:8px !important; gap:6px !important;
  border:1px solid #bfdbfe !important; border-radius:14px !important;
  background:linear-gradient(135deg,#eef2ff,#eaf6ff) !important;
  box-shadow:0 24px 70px rgba(15,23,42,.38) !important; overflow:hidden !important; }
#replay-dialog > .form, #replay-dialog .form {
  gap:6px !important; background:transparent !important; border:0 !important;
  box-shadow:none !important;
}
#replay-dialog > div, #replay-dialog [data-testid="video"] {
  color:#0f172a !important; background:transparent !important;
}
#replay-dialog .section-head { margin:0 !important; }
#replay-dialog .section-eyebrow { display:none !important; }
#replay-dialog .section-desc-new {
  display:block !important; margin:2px 0 0 4px !important;
  color:#64748b !important; font-size:12px !important; line-height:1.4 !important;
}
#replay-dialog .section-title-new { margin:0 !important; padding-left:4px; font-size:17px !important; }
#close-replay-btn button, button#close-replay-btn {
  min-width:64px !important; max-width:64px !important; min-height:32px !important;
}
#replay-dialog video { display:block !important; width:100% !important; max-height:calc(100vh - 94px) !important;
  object-fit:cover !important; border-radius:9px !important; background:#fff !important; }
#replay-dialog [data-testid="video"] { margin:0 !important; padding:0 !important; border:0 !important; }
footer { display:none !important; }
@media (max-width: 768px) {
  .gradio-container,
  body > gradio-app .gradio-container,
  gradio-app .gradio-container {
    width:calc(100% - 16px) !important; padding:18px 10px 40px !important;
  }
  .app-header { align-items:flex-start !important; gap:12px !important; }
  .judge-pill { display:none !important; }
  .app-title { font-size:20px !important; }
  .panel-card { padding:15px !important; border-radius:16px !important; }
  .media-frame, .browser-camera-shell { height:auto !important; min-height:220px !important; }
  #status-display { min-width:130px !important; }
  .summary-layout { grid-template-columns:minmax(0,1fr); }
  .summary-chart { justify-content:flex-start; padding:18px 0 0; border-left:0;
    border-top:1px solid rgba(99,102,241,.16); }
}
"""

CAMERA_OPEN_JS = """async () => {
  const app = document.querySelector('gradio-app');
  if (app && app.dataset.inputLocked === 'true') return [];
  const video = document.getElementById('browser-camera-video');
  const shell = document.getElementById('browser-camera-shell');
  const host = document.getElementById('camera-preview-host');
  const empty = document.getElementById('browser-camera-empty');
  const freeze = document.getElementById('browser-camera-freeze');
  if (!video || !shell) return [];
  if (host) host.style.display = 'block';
  shell.style.display = 'block';
  if (freeze) freeze.style.display = 'none';
  const analysisFrame = document.querySelector('.analysis-camera-frame');
  if (analysisFrame) analysisFrame.style.display = 'none';
  const recording = document.querySelector('.camera-recording');
  if (recording) recording.style.display = 'none';
  if (window.__almCameraStream) {
    video.srcObject = window.__almCameraStream;
    if (empty) empty.style.display = 'none';
    return [];
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: {ideal: 1280},
        height: {ideal: 720},
        aspectRatio: {ideal: 16 / 9}
      },
      audio: false
    });
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
  const app = document.querySelector('gradio-app');
  if (app) app.dataset.inputLocked = 'true';
  document.querySelectorAll('.tab-nav button').forEach(button => {
    button.disabled = true;
    button.setAttribute('aria-disabled', 'true');
  });
  if (window.__almInputUnlockTimer) clearInterval(window.__almInputUnlockTimer);
  let observedActiveState = false;
  window.__almInputUnlockTimer = setInterval(() => {
    const status = document.getElementById('status-display');
    const text = status ? status.textContent : '';
    if (text.includes('正在加载模型') || text.includes('正在启动摄像头') ||
        text.includes('正在录制并分析') || text.includes('运行中') ||
        text.includes('正在结束分析')) {
      observedActiveState = true;
      return;
    }
    if (!observedActiveState ||
        (!text.includes('分析结束') && !text.includes('已停止') &&
         !text.includes('模型加载失败'))) return;
    clearInterval(window.__almInputUnlockTimer);
    window.__almInputUnlockTimer = null;
    if (app) delete app.dataset.inputLocked;
    document.querySelectorAll('.tab-nav button').forEach(button => {
      button.disabled = false;
      button.removeAttribute('aria-disabled');
    });
  }, 200);
  if (window.__almCameraHandoffTimer) clearInterval(window.__almCameraHandoffTimer);
  window.__almCameraHandoffTimer = setInterval(() => {
    const status = document.getElementById('status-display');
    if (!status || !status.textContent.includes('正在启动摄像头')) return;
    clearInterval(window.__almCameraHandoffTimer);
    window.__almCameraHandoffTimer = null;
    const video = document.getElementById('browser-camera-video');
    const freeze = document.getElementById('browser-camera-freeze');
    if (video && freeze && video.videoWidth > 0 && video.videoHeight > 0) {
      freeze.width = video.videoWidth;
      freeze.height = video.videoHeight;
      const context = freeze.getContext('2d');
      if (context) {
        context.drawImage(video, 0, 0, freeze.width, freeze.height);
        freeze.style.display = 'block';
      }
    }
    if (window.__almCameraStream) {
      window.__almCameraStream.getTracks().forEach(track => track.stop());
      window.__almCameraStream = null;
    }
    // 先保留浏览器视频的最后一帧。只有 OpenCV 的第一张真实预览已经
    // 渲染出来后才隐藏它，避免两路摄像头交接时露出白色图片占位框。
    if (window.__almCameraFirstFrameTimer) clearInterval(window.__almCameraFirstFrameTimer);
    window.__almCameraFirstFrameTimer = setInterval(() => {
      const analysisFrame = document.querySelector('.analysis-camera-frame');
      const image = analysisFrame ? analysisFrame.querySelector('img') : null;
      const src = image ? (image.currentSrc || image.getAttribute('src') || '') : '';
      const status = document.getElementById('status-display');
      const isRecording = status && status.textContent.includes('正在录制并分析');
      const isVisible = analysisFrame && getComputedStyle(analysisFrame).display !== 'none';
      if (!isRecording || !isVisible || !image || !src || image.naturalWidth <= 0) return;
      clearInterval(window.__almCameraFirstFrameTimer);
      window.__almCameraFirstFrameTimer = null;
      const shell = document.getElementById('browser-camera-shell');
      const host = document.getElementById('camera-preview-host');
      if (shell) shell.style.display = 'none';
      if (host) host.style.display = 'none';
    }, 50);
  }, 100);
  return [mode, videoPath];
}"""

LOCK_UPLOAD_TABS_JS = """(mode, videoPath) => {
  if (videoPath) {
    const app = document.querySelector('gradio-app');
    if (app) app.dataset.inputLocked = 'true';
    document.querySelectorAll('.tab-nav button').forEach(button => {
      button.disabled = true;
      button.setAttribute('aria-disabled', 'true');
    });
    if (window.__almInputUnlockTimer) clearInterval(window.__almInputUnlockTimer);
    let observedActiveState = false;
    window.__almInputUnlockTimer = setInterval(() => {
    const status = document.getElementById('status-display');
    const text = status ? status.textContent : '';
    if (text.includes('正在加载模型') || text.includes('运行中') ||
        text.includes('正在结束分析')) {
      observedActiveState = true;
      return;
    }
    if (!observedActiveState ||
        (!text.includes('分析结束') && !text.includes('已停止') &&
         !text.includes('模型加载失败'))) return;
      clearInterval(window.__almInputUnlockTimer);
      window.__almInputUnlockTimer = null;
      if (app) delete app.dataset.inputLocked;
      document.querySelectorAll('.tab-nav button').forEach(button => {
        button.disabled = false;
        button.removeAttribute('aria-disabled');
      });
    }, 200);
  }
  return [mode, videoPath];
}"""

CAMERA_STOP_ANALYSIS_JS = """() => {
  if (window.__almCameraStream) {
    window.__almCameraStream.getTracks().forEach(track => track.stop());
    window.__almCameraStream = null;
  }
  const shell = document.getElementById('browser-camera-shell');
  const host = document.getElementById('camera-preview-host');
  if (shell) shell.style.display = 'none';
  if (host) host.style.display = 'none';
  const analysisFrame = document.querySelector('.analysis-camera-frame');
  if (analysisFrame && analysisFrame.querySelector('img')) {
    analysisFrame.style.display = 'block';
  }
  return [];
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
        "loading": "正在加载模型",
        "loading_camera": "正在加载模型（尚未开始录制）",
        "ready": "模型已就绪",
        "starting": "正在启动摄像头（尚未录制）",
        "run": "运行中",
        "recording": "正在录制并分析",
        "ending": "正在结束分析",
        "stop": "已停止",
        "done": "分析结束",
        "error": "模型加载失败",
    }[state]
    return (
        '<div class="status-line">'
        f'<span class="status-dot {state}"></span><span>{label}</span>'
        '</div>'
    )


_runtime_runner = None
_analysis_running = False
_prepared_detector = None
_models_ready = threading.Event()
_model_prepare_lock = threading.Lock()
_model_prepare_error = None
UI_STATE_PATH = Path(PROJECT_ROOT) / "data" / "ui_state.json"


def _prepare_models() -> None:
    """加载一次并缓存两种输入方式共用的模型权重。"""
    global _prepared_detector, _model_prepare_error
    if _models_ready.is_set():
        return
    with _model_prepare_lock:
        if _models_ready.is_set():
            return
        try:
            if _prepared_detector is None:
                _prepared_detector = YoloDetector(
                    device=os.getenv("YOLO_DEVICE", "0"),
                    conf_threshold=float(os.getenv("YOLO_CONFIDENCE", "0.35")),
                )
            # load_model 自带进程级缓存；后续每轮 EndToEndRunner 会直接复用。
            load_model(model_path=os.getenv("QWEN_VL_MODEL_PATH"))
        except Exception as exc:
            _model_prepare_error = exc
            raise
        _model_prepare_error = None
        _models_ready.set()


def on_page_load():
    """页面出现后立即预热模型，完成前禁止开始分析。"""
    yield (
        status_html("loading"),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )
    try:
        _prepare_models()
    except Exception as exc:
        gr.Warning(f"模型加载失败：{exc}")
        yield (
            status_html("error"),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
        return
    yield (
        status_html("ready"),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


# ============ 回调（后续接 A/D 真实模块） ============

def section_head(eyebrow: str, title: str, description: str) -> str:
    return (
        '<div class="section-head">'
        f'<div class="section-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="section-title-new">{html.escape(title)}</div>'
        f'<div class="section-desc-new">{html.escape(description)}</div>'
        '</div>'
    )


_SUMMARY_CHART_COLORS = {
    "sit_at_study_position": "#6366f1",
    "leave_study_position": "#64748b",
    "reading": "#0ea5e9",
    "writing": "#8b5cf6",
    "phone_usage": "#f59e0b",
    "computer_usage": "#2563eb",
    "communication_distraction": "#ef4444",
    "other_behavior": "#94a3b8",
}


def _format_second_value(value: float) -> str:
    rounded = round(max(0.0, float(value)), 2)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _format_display_time(seconds: float) -> str:
    """将秒数转换为适合中文界面展示的秒、分或小时格式。"""
    total = round(max(0.0, float(seconds)), 2)
    hours = int(total // 3600)
    remainder = total - hours * 3600
    minutes = int(remainder // 60)
    remaining_seconds = remainder - minutes * 60
    second_text = _format_second_value(remaining_seconds)
    if hours:
        parts = [f"{hours}小时"]
        if minutes:
            parts.append(f"{minutes}分")
        if remaining_seconds > 0:
            parts.append(f"{second_text}秒")
        return "".join(parts)
    if minutes:
        result = f"{minutes}分"
        if remaining_seconds > 0:
            result += f"{second_text}秒"
        return result
    return f"{_format_second_value(total)}秒"


def _parse_display_time(value: str) -> float | None:
    """解析新中文时间格式及旧版 ``12.34s`` 格式。"""
    text = str(value or "").strip()
    old_match = re.fullmatch(r"([0-9.]+)s", text, flags=re.IGNORECASE)
    if old_match:
        return float(old_match.group(1))
    match = re.fullmatch(
        r"(?:(\d+)小时)?(?:(\d+)分)?(?:([0-9.]+)秒)?",
        text,
    )
    if not match or not any(match.groups()):
        return None
    hours = float(match.group(1) or 0)
    minutes = float(match.group(2) or 0)
    seconds = float(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _summary_distribution(records) -> tuple[float, list[dict]]:
    if not records:
        return 0.0, []
    durations: dict[str, float] = {}
    for record in records:
        event_type = str(record.get("event_type", "other_behavior"))
        duration = max(
            0.0,
            float(record.get("end_time", 0.0)) - float(record.get("start_time", 0.0)),
        )
        durations[event_type] = durations.get(event_type, 0.0) + duration
    total = sum(durations.values())
    colors = _SUMMARY_CHART_COLORS
    items = []
    for event_type, duration in durations.items():
        if duration <= 0 or total <= 0:
            continue
        items.append({
            "event_type": event_type,
            "label": event_label(event_type),
            "duration": duration,
            "percentage": duration / total * 100.0,
            "color": colors.get(event_type, "#94a3b8"),
        })
    items.sort(key=lambda item: item["percentage"], reverse=True)
    return total, items


def _summary_records_from_choices(choices) -> list[dict]:
    """兼容旧版 UI 状态：从时间线显示文字恢复统计所需的最小字段。"""
    label_to_event = {
        event_label(event_type): event_type
        for event_type in _SUMMARY_CHART_COLORS
    }
    records = []
    for choice in choices or []:
        display = choice[0] if isinstance(choice, (list, tuple)) and choice else choice
        lines = str(display or "").splitlines()
        if len(lines) < 2 or lines[0].strip() not in label_to_event:
            continue
        time_parts = re.split(r"\s*[–—]\s*", lines[1], maxsplit=1)
        if len(time_parts) != 2:
            continue
        start_time = _parse_display_time(time_parts[0])
        end_time = _parse_display_time(time_parts[1])
        if start_time is None or end_time is None:
            continue
        records.append({
            "start_time": start_time,
            "end_time": end_time,
            "event_type": label_to_event[lines[0].strip()],
        })
    return records


def _normalize_timeline_choice_times(choices):
    """将持久化的旧时间线文字升级为当前中文时间格式。"""
    normalized = []
    for choice in choices or []:
        if isinstance(choice, (list, tuple)) and choice:
            display = choice[0]
            value = choice[1] if len(choice) > 1 else choice[0]
        else:
            display = choice
            value = choice
        lines = str(display or "").splitlines()
        if len(lines) >= 2:
            time_parts = re.split(r"\s*[–—]\s*", lines[1], maxsplit=1)
            if len(time_parts) == 2:
                start_time = _parse_display_time(time_parts[0])
                end_time = _parse_display_time(time_parts[1])
                if start_time is not None and end_time is not None:
                    lines[1] = (
                        f"{_format_display_time(start_time)} – "
                        f"{_format_display_time(end_time)}"
                    )
                    if (
                        end_time - start_time > 60.0
                        and not str(value).startswith("__SUMMARY_REPLAY__:")
                    ):
                        value = "__SUMMARY_REPLAY__:" + str(value)
        normalized.append(("\n".join(lines), value))
    return normalized


def summary_html(summary: str = "", records=None) -> str:
    content = html.escape(summary.strip()) if summary.strip() else (
        '<span class="summary-empty">分析完成后，这里将生成覆盖全部事件的综合描述。</span>'
    )
    total, distribution = _summary_distribution(records or [])
    if not distribution:
        return (
            '<div class="summary-box"><div class="summary-layout no-chart">'
            f'<div class="summary-copy">{content}</div></div></div>'
        )
    start = 0.0
    sectors = []
    legend = []
    for item in distribution:
        end = start + item["percentage"] * 3.6
        sectors.append(f'{item["color"]} {start:.2f}deg {end:.2f}deg')
        legend.append(
            '<div class="summary-legend-item">'
            f'<span class="summary-legend-dot" style="background:{item["color"]}"></span>'
            f'<span>{html.escape(item["label"])}</span>'
            f'<span class="summary-legend-percentage">{item["percentage"]:.1f}%</span>'
            f'<span class="summary-legend-duration">{_format_display_time(item["duration"])}</span>'
            '</div>'
        )
        start = end
    chart = (
        '<div class="summary-chart">'
        f'<div class="summary-pie" style="background:conic-gradient({",".join(sectors)})">'
        '<div class="summary-pie-center"><span>有效总时长</span>'
        f'<strong>{_format_display_time(total)}</strong></div></div>'
        f'<div class="summary-legend">{"".join(legend)}</div></div>'
    )
    return (
        '<div class="summary-box"><div class="summary-layout">'
        f'<div class="summary-copy">{content}</div>{chart}</div></div>'
    )


def _event_choices(records):
    return [
        (
            f"{event_label(item['event_type'])}\n"
            f"{_format_display_time(item['start_time'])} – "
            f"{_format_display_time(item['end_time'])}\n"
            f"{item['caption']}",
            (
                "__SUMMARY_REPLAY__:" + str(item.get("video_path", ""))
                if float(item.get("end_time", 0.0)) - float(item.get("start_time", 0.0)) > 60.0
                else str(item.get("video_path", ""))
            ),
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
    requested_type = match_event_type(query or "")
    if requested_type is not None:
        # 与最终时间线使用同一份去抖结果，避免搜索再次展示已被消除的
        # 三秒内 other，或把同一持续行为拆成多个结果。
        found = [
            item
            for item in merge_events_for_display(
                _runtime_runner.memory_store.list_all()
            )
            if item["event_type"] == requested_type
        ]
    else:
        found = merge_events_for_display(
            _runtime_runner.search(query or "刚才发生了什么？", top_k=50)
        )
    for item in found:
        if item["merged_event_count"] > 1:
            item["video_path"] = _runtime_runner.create_replay(
                item["start_time"], item["end_time"]
            )
    choices = _event_choices(found)
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


def _replay_header(summary: bool = False) -> str:
    if summary:
        return section_head(
            "REPLAY SUMMARY",
            "事件摘要回放",
            "该事件超过60秒，当前展示开头、中间和结尾的摘要片段。",
        )
    return section_head("REPLAY", "事件回放", "当前选中事件的完整视频片段")


def on_select_event(seg_path):
    """时间线和搜索结果共用：点击事件卡片即播放。"""
    if not seg_path:
        gr.Warning("该事件没有可用的回放片段")
        return gr.update(visible=False), gr.update(visible=False), gr.update()
    value = str(seg_path)
    is_summary = value.startswith("__SUMMARY_REPLAY__:")
    if is_summary:
        value = value.removeprefix("__SUMMARY_REPLAY__:")
        gr.Info("该事件超过60秒，当前打开的是开头、中间和结尾的摘要回放")
    return (
        gr.update(visible=True),
        gr.update(value=value, visible=True),
        gr.update(value=_replay_header(is_summary)),
    )


def on_close_replay():
    return (
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=_replay_header()),
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
        timeline_choices = _normalize_timeline_choice_times(
            saved.get("timeline_choices", [])
        )
        summary_records = saved.get("summary_records") or _summary_records_from_choices(
            timeline_choices
        )
        restored_summary = str(saved.get("video_summary", ""))
        if summary_records:
            restored_summary = EndToEndRunner._fallback_video_summary(summary_records)
        return (
            saved.get("status_html", status_html("done")),
            summary_html(
                restored_summary,
                summary_records,
            ),
            gr.update(
                choices=timeline_choices,
                value=None,
                visible=bool(timeline_choices),
            ),
            gr.update(
                interactive=ready,
                placeholder=(
                    "输入搜索内容"
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


def _camera_recording_updates(input_mode: str):
    if input_mode == "本机摄像头" and _runtime_runner is not None:
        recording_path = _runtime_runner.create_full_recording()
        if recording_path:
            return (
                gr.update(value=None, visible=False),
                gr.update(value=recording_path, visible=True),
            )
    return _preview_frame(), gr.update(value=None, visible=False)


def on_start(input_mode, video_path):
    global _runtime_runner, _analysis_running
    if _analysis_running:
        gr.Warning("当前分析尚未结束")
        yield (gr.update(),) * 10
        return
    if input_mode == "上传视频" and not video_path:
        gr.Warning("请先上传一个本地视频文件")
        yield (
            status_html("idle"), summary_html(),
            gr.update(choices=[], value=None, visible=False),
            gr.update(interactive=False), gr.update(interactive=False), gr.update(),
            gr.update(value=None, visible=False),
            gr.update(interactive=True), gr.update(interactive=True),
            gr.update(interactive=True),
        )
        return

    source = 0 if input_mode == "本机摄像头" else str(video_path)
    _analysis_running = True
    _persist_ui_state(
        "running",
        input_mode=input_mode,
        video_path=(str(video_path) if video_path else None),
    )
    if not _models_ready.is_set():
        loading_state = (
            "loading_camera" if input_mode == "本机摄像头" else "loading"
        )
        loading_message = (
            "模型正在准备，当前尚未开始录制……"
            if input_mode == "本机摄像头"
            else "正在加载模型并准备视频分析……"
        )
        yield (
            status_html(loading_state), summary_html(loading_message),
            gr.update(choices=[], value=None, visible=False),
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=False),
        )
        try:
            _prepare_models()
        except Exception as exc:
            _analysis_running = False
            _persist_ui_state(
                "error",
                input_mode=input_mode,
                video_path=(str(video_path) if video_path else None),
                message=str(exc),
            )
            yield (
                status_html("error"), summary_html(f"模型加载失败：{exc}"),
                gr.update(choices=[], value=None, visible=False),
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(interactive=True), gr.update(interactive=True),
                gr.update(interactive=False),
            )
            return

    try:
        _runtime_runner = EndToEndRunner(
            source=source,
            detector=_prepared_detector,
            qwen_model_path=os.getenv("QWEN_VL_MODEL_PATH"),
            yolo_device=os.getenv("YOLO_DEVICE", "0"),
            yolo_confidence=float(os.getenv("YOLO_CONFIDENCE", "0.35")),
        )
    except Exception as exc:
        _analysis_running = False
        _persist_ui_state(
            "error",
            input_mode=input_mode,
            video_path=(str(video_path) if video_path else None),
            message=str(exc),
        )
        yield (
            status_html("stop"), summary_html(f"分析启动失败：{exc}"),
            gr.update(choices=[], value=None, visible=False),
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=True), gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    initial_state = "starting" if input_mode == "本机摄像头" else "run"
    initial_message = (
        "模型已就绪，正在启动摄像头；第一帧写入后开始录制……"
        if input_mode == "本机摄像头"
        else "正在分析中，请稍后……"
    )
    yield (
        status_html(initial_state), summary_html(initial_message),
        gr.update(choices=[], value=None, visible=False),
        gr.update(interactive=False), gr.update(interactive=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(interactive=False), gr.update(interactive=False),
        gr.update(interactive=False),
    )
    if input_mode == "本机摄像头":
        time.sleep(0.6)
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
        if _runtime_runner._stop_event.is_set():
            run_state = "ending"
        elif input_mode == "本机摄像头":
            run_state = (
                "recording" if _runtime_runner.recording_started() else "starting"
            )
        else:
            run_state = "run"
        recording_is_ready = (
            input_mode == "本机摄像头"
            and _runtime_runner._stop_event.is_set()
            and _runtime_runner.wait_for_recording_ready(timeout=0.0)
        )
        if recording_is_ready:
            preview_update, recording_update = _camera_recording_updates(input_mode)
        elif input_mode == "本机摄像头" and not _runtime_runner.recording_started():
            preview_update = gr.update(value=None, visible=False)
            recording_update = gr.update(value=None, visible=False)
        else:
            preview_update = _preview_frame()
            recording_update = gr.update(value=None, visible=False)
        progress_message = (
            "正在结束分析，请稍后……"
            if run_state == "ending"
            else (
                "模型已就绪，正在启动摄像头；第一帧写入后开始录制……"
                if run_state == "starting"
                else "正在分析中，请稍后……"
            )
        )
        yield (
            status_html(run_state), summary_html(progress_message), gr.update(),
            gr.update(interactive=False), gr.update(interactive=False),
            preview_update, recording_update,
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=False),
        )
        time.sleep(0.25)
    analysis_thread.join()
    _analysis_running = False
    if "error" in error_box:
        exc = error_box["error"]
        preview_update, recording_update = _camera_recording_updates(input_mode)
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
            preview_update, recording_update,
            gr.update(interactive=True), gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return
    result = result_box["result"]
    preview_update, recording_update = _camera_recording_updates(input_mode)
    records = merge_events_for_display(_runtime_runner.memory_store.list_all())
    for record in records:
        if record["merged_event_count"] > 1:
            record["video_path"] = _runtime_runner.create_replay(
                record["start_time"], record["end_time"]
            )

    timeline_choices = _event_choices(records)
    video_summary = str(result.get("video_summary", "")).strip()
    summary_records = [
        {
            "start_time": float(record["start_time"]),
            "end_time": float(record["end_time"]),
            "event_type": str(record["event_type"]),
        }
        for record in records
    ]
    ready = bool(records)
    completed_status = status_html("done")
    _persist_ui_state(
        "completed",
        input_mode=input_mode,
        video_path=(str(video_path) if video_path else None),
        ready=ready,
        status_html=completed_status,
        video_summary=video_summary,
        summary_records=summary_records,
        timeline_choices=timeline_choices,
    )
    print(
        f"[分析完成] 处理 {int(result['stream']['frames'])} 帧 | "
        f"写入 {result['memories_saved']} 条记忆 | "
        f"错误 {len(result['errors'])} 个 | 页面状态已保存到 {UI_STATE_PATH}"
    )
    yield (
        completed_status, summary_html(video_summary, summary_records),
        gr.update(
            choices=timeline_choices,
            value=None,
            visible=bool(timeline_choices),
        ),
        gr.update(
            interactive=ready,
            placeholder=(
                "输入搜索内容"
                if ready else "本次分析没有产生可检索的确认事件"
            ),
        ),
        gr.update(interactive=ready),
        preview_update, recording_update,
        gr.update(interactive=True), gr.update(interactive=True),
        gr.update(interactive=False),
    )


def on_stop():
    if _analysis_running and _runtime_runner is not None:
        _runtime_runner.stop()
        gr.Info("已请求停止，正在完成剩余分析")
        if _runtime_runner.recorder is not None:
            if _runtime_runner.wait_for_recording_ready(timeout=10.0):
                recording_path = _runtime_runner.create_full_recording()
                if recording_path:
                    return (
                        status_html("ending"),
                        gr.update(value=None, visible=False),
                        gr.update(value=recording_path, visible=True),
                    )
        return status_html("ending"), gr.update(), gr.update()
    return status_html("idle"), gr.update(), gr.update()


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
            with gr.Tabs(elem_classes=["tab-nav"]):
                with gr.Tab("上传视频") as upload_tab:
                    with gr.Column(elem_classes=["input-mode-stack"]):
                        video_input = gr.Video(
                            sources=["upload"], label=None, height=450,
                            elem_id="video-input", elem_classes=["media-frame"],
                        )
                        upload_start_btn = gr.Button(
                            "开始分析", variant="primary", elem_id="upload-start-btn",
                            elem_classes=["primary"]
                        )
                with gr.Tab("实时摄像头") as camera_tab:
                    with gr.Column(elem_classes=["input-mode-stack"]):
                        gr.HTML(
                            '<div class="browser-camera-shell" id="browser-camera-shell">'
                            '<video id="browser-camera-video" autoplay muted playsinline></video>'
                            '<canvas id="browser-camera-freeze" aria-hidden="true"></canvas>'
                            '<div class="browser-camera-empty" id="browser-camera-empty">正在打开摄像头…</div>'
                            '</div>',
                            elem_id="camera-preview-host",
                            elem_classes=["camera-preview-host"],
                        )
                        live_preview = gr.Image(
                            label=None, show_label=False, container=False,
                            interactive=False, height=450, visible=False,
                            elem_classes=["media-frame", "analysis-camera-frame"],
                        )
                        camera_recording = gr.Video(
                            label=None, show_label=False, interactive=False,
                            visible=False, height=450,
                            elem_classes=["media-frame", "camera-recording"],
                        )
                        camera_start_btn = gr.Button(
                            "开始分析", variant="primary", elem_id="camera-start-btn",
                            elem_classes=["primary"]
                        )
            with gr.Row(equal_height=True):
                stop_btn = gr.Button("停止", elem_id="stop-btn", elem_classes=["secondary"])
                restore_btn = gr.Button(
                    "恢复最近结果", elem_id="restore-btn", elem_classes=["secondary"]
                )
                status = gr.HTML(
                    status_html("idle"), elem_id="status-display",
                    elem_classes=["compact-status"],
                )

        with gr.Column(elem_classes=["panel-card"]):
            gr.HTML(section_head("OVERVIEW", "全事件总结", "按照已确认事件及其时间顺序生成客观过程描述。"))
            summary_panel = gr.HTML(summary_html())

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["panel-card"]):
                gr.HTML(section_head("TIMELINE", "具体事件时间线", "点击事件即可打开对应回放。"))
                timeline_list = gr.Radio(
                    choices=[], value=None, label=None, show_label=False, visible=False,
                    elem_id="timeline-list", elem_classes=["event-list"],
                )

            with gr.Column(scale=1, elem_classes=["panel-card", "search-panel"]):
                gr.HTML(section_head("SEARCH", "自然语言搜索", "例如：什么时候阅读了？"))
                with gr.Row(equal_height=True, elem_id="search-controls"):
                    query = gr.Textbox(
                        placeholder="请先完成视频分析", label=None, show_label=False,
                        interactive=False, scale=5, elem_id="search-query",
                    )
                    search_btn = gr.Button(
                        "搜索", scale=1, variant="primary",
                        interactive=False, elem_id="search-btn", elem_classes=["primary"],
                    )
                search_feedback = gr.Markdown(
                    value="", visible=False, elem_id="search-feedback"
                )
                search_results = gr.Radio(
                    choices=[], value=None, label=None, show_label=False, visible=False,
                    elem_id="search-result-list", elem_classes=["event-list"],
                )

        with gr.Group(visible=False, elem_id="replay-modal") as replay_modal:
            with gr.Column(elem_id="replay-dialog"):
                with gr.Row():
                    replay_header = gr.HTML(_replay_header())
                    close_replay_btn = gr.Button(
                        "关闭", size="sm", elem_id="close-replay-btn",
                        elem_classes=["secondary"],
                    )
                replay_video = gr.Video(
                    label=None, interactive=False, visible=False,
                    elem_classes=["media-frame"],
                )
        # ---- 事件绑定 ----
        video_input.upload(on_video_upload, inputs=video_input, outputs=video_input, show_progress="hidden")
        start_outputs = [
            status, summary_panel, timeline_list,
            query, search_btn, live_preview, camera_recording,
            upload_start_btn, camera_start_btn, restore_btn,
        ]
        upload_start_btn.click(
            on_start,
            inputs=[upload_mode, video_input],
            outputs=start_outputs,
            js=LOCK_UPLOAD_TABS_JS,
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
            fn=None,
            js=CAMERA_CLOSE_JS,
            show_progress="hidden",
        )
        stop_btn.click(
            on_stop, outputs=[status, live_preview, camera_recording], queue=False,
            js=CAMERA_STOP_ANALYSIS_JS, show_progress="hidden",
        )
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
            outputs=[replay_modal, replay_video, replay_header],
            show_progress="hidden",
        )
        search_results.input(
            on_select_event,
            inputs=search_results,
            outputs=[replay_modal, replay_video, replay_header],
            show_progress="hidden",
        )
        close_replay_btn.click(
            on_close_replay,
            outputs=[replay_modal, replay_video, replay_header, timeline_list, search_results],
            show_progress="hidden",
            queue=False,
        )
        restore_outputs = [
            status, summary_panel, timeline_list,
            query, search_btn, video_input,
        ]
        restore_btn.click(on_restore, outputs=restore_outputs, show_progress="hidden")
        demo.load(
            on_page_load,
            outputs=[status, upload_start_btn, camera_start_btn],
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    configure_utf8_tee(Path(PROJECT_ROOT) / "result.txt")
    print("[日志] 终端输出同步写入 UTF-8 result.txt")
    build_ui().launch(
        allowed_paths=[str(Path(PROJECT_ROOT) / "data")],
        theme=APP_THEME,
        css=CSS + PRESENTATION_CSS,
    )
