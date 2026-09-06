from demo.core.ui_state import load_ui_state, save_ui_state


def test_ui_state_round_trip(tmp_path):
    path = tmp_path / "ui_state.json"
    expected = {
        "state": "completed",
        "video_path": "D:/video.mp4",
        "ready": True,
        "timeline_html": "<div>阅读</div>",
    }

    save_ui_state(path, expected)

    assert load_ui_state(path) == expected
    assert not path.with_suffix(".json.tmp").exists()


def test_ui_state_ignores_broken_json(tmp_path):
    path = tmp_path / "ui_state.json"
    path.write_text('{"state":', encoding="utf-8")

    assert load_ui_state(path) is None
