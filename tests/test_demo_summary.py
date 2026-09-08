from demo.app import (
    _event_choices,
    _format_display_time,
    _normalize_timeline_choice_times,
    _summary_distribution,
    _summary_records_from_choices,
    summary_html,
)


def test_summary_distribution_and_pie_chart_use_event_durations():
    records = [
        {"start_time": 0.0, "end_time": 3.0, "event_type": "reading"},
        {"start_time": 3.0, "end_time": 5.0, "event_type": "other_behavior"},
    ]

    total, items = _summary_distribution(records)
    rendered = summary_html("人物先阅读，随后进行整理。", records)

    assert total == 5.0
    assert [(item["label"], item["percentage"]) for item in items] == [
        ("阅读", 60.0),
        ("其他", 40.0),
    ]
    assert "summary-pie" in rendered
    assert "conic-gradient" in rendered
    assert "阅读" in rendered
    assert "60.0%" in rendered
    assert "3秒" in rendered
    assert "总时长</span><strong>5秒" in rendered


def test_display_time_switches_from_seconds_to_minutes_and_hours():
    assert _format_display_time(15.25) == "15.25秒"
    assert _format_display_time(75) == "1分15秒"
    assert _format_display_time(90.5) == "1分30.5秒"
    assert _format_display_time(3600) == "1小时"
    assert _format_display_time(3723.25) == "1小时2分3.25秒"


def test_timeline_and_search_choices_use_readable_time_ranges():
    choices = _event_choices([
        {
            "start_time": 75.0,
            "end_time": 3723.25,
            "event_type": "reading",
            "caption": "人物持续阅读。",
        }
    ])

    assert "1分15秒 – 1小时2分3.25秒" in choices[0][0]


def test_old_timeline_choices_can_restore_summary_chart_records():
    records = _summary_records_from_choices([
        ["阅读\n2.00s – 6.00s\n人物持续阅读。", "replay.mp4"],
        ["写字\n6.00s – 10.00s\n人物持续书写。", "replay-2.mp4"],
    ])

    assert records == [
        {"start_time": 2.0, "end_time": 6.0, "event_type": "reading"},
        {"start_time": 6.0, "end_time": 10.0, "event_type": "writing"},
    ]


def test_chinese_timeline_choices_can_restore_summary_chart_records():
    records = _summary_records_from_choices([
        ["阅读\n1分15秒 – 1小时2分3.25秒\n人物持续阅读。", "replay.mp4"],
    ])

    assert records == [
        {"start_time": 75.0, "end_time": 3723.25, "event_type": "reading"},
    ]


def test_old_timeline_display_is_upgraded_to_chinese_time_units():
    choices = _normalize_timeline_choice_times([
        ["阅读\n75.00s – 3723.25s\n人物持续阅读。", "replay.mp4"],
    ])

    assert choices == [
        ("阅读\n1分15秒 – 1小时2分3.25秒\n人物持续阅读。", "replay.mp4")
    ]
