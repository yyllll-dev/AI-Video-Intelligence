"""Gradio 页面最近一次分析状态的轻量持久化。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def save_ui_state(path: str | Path, state: dict[str, Any]) -> None:
    """原子保存页面状态，避免刷新恰好读到半个 JSON 文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, target)


def load_ui_state(path: str | Path) -> dict[str, Any] | None:
    """读取最近状态；文件缺失或损坏时返回 None，不阻断 UI 启动。"""
    target = Path(path)
    if not target.is_file():
        return None
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
