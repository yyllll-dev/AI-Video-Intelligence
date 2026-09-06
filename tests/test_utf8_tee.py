import io
import threading

from demo.core.utf8_tee import Utf8Tee


def test_utf8_tee_writes_terminal_and_readable_utf8_file(tmp_path):
    terminal = io.StringIO()
    path = tmp_path / "result.txt"
    tee = Utf8Tee(terminal, path, threading.Lock())

    tee.write("[分析窗口] 人物正在写字。\n")
    tee.write("没有 NULL。\n")

    assert terminal.getvalue() == "[分析窗口] 人物正在写字。\n没有 NULL。\n"
    assert path.read_text(encoding="utf-8") == terminal.getvalue()
    assert b"\x00" not in path.read_bytes()
