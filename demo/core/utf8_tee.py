"""同时写终端和 UTF-8 日志，且不长期锁住日志文件。"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TextIO


class Utf8Tee:
    def __init__(self, terminal: TextIO, path: str | Path, lock: threading.Lock):
        self.terminal = terminal
        self.path = Path(path)
        self.lock = lock
        self.encoding = getattr(terminal, "encoding", "utf-8")

    def write(self, text: str) -> int:
        written = self.terminal.write(text)
        if text:
            # 每次短暂打开后立即关闭，VS Code/Codex 可在服务运行时读取。
            with self.lock:
                with self.path.open("a", encoding="utf-8", newline="") as handle:
                    handle.write(text)
        return written if isinstance(written, int) else len(text)

    def flush(self) -> None:
        self.terminal.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.terminal.fileno()


def configure_utf8_tee(path: str | Path) -> None:
    """覆盖旧日志并安装一次进程级 stdout/stderr 双写。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    lock = threading.Lock()
    if not isinstance(sys.stdout, Utf8Tee):
        sys.stdout = Utf8Tee(sys.stdout, target, lock)
    if not isinstance(sys.stderr, Utf8Tee):
        sys.stderr = Utf8Tee(sys.stderr, target, lock)
