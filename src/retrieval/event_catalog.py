"""Retrieval 使用的统一事件检索词典。

事件名称以 A 模块 ``src.event.event_types.ALL_EVENTS`` 为唯一来源。本文件只补充
自然语言检索所需的中文含义和常见问法，不修改 Event 或 EventEngine 接口。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..event.event_types import ALL_EVENTS


@dataclass(frozen=True, slots=True)
class EventSearchTerms:
    """一个正式事件名称对应的中文含义和常见查询表达。"""

    label: str
    aliases: tuple[str, ...]


EVENT_SEARCH_TERMS: dict[str, EventSearchTerms] = {
    "sit_at_study_position": EventSearchTerms(
        "坐到学习位置",
        ("坐到学习位置", "坐到座位", "坐下", "入座"),
    ),
    "leave_study_position": EventSearchTerms(
        "离开学习位置",
        ("离开学习位置", "离开座位", "起身离开", "不在座位"),
    ),
    "study_preparation": EventSearchTerms(
        "学习准备",
        ("学习准备", "准备学习", "准备文具", "摆放书本"),
    ),
    "start_study": EventSearchTerms(
        "开始学习",
        ("开始学习", "进入学习状态", "正式学习"),
    ),
    "end_study": EventSearchTerms(
        "结束学习",
        ("结束学习", "停止学习", "学习结束"),
    ),
    "reading": EventSearchTerms(
        "阅读",
        ("阅读资料", "阅读", "读书", "看书"),
    ),
    "writing": EventSearchTerms(
        "书写",
        ("写作业", "记笔记", "书写", "写字", "做题"),
    ),
    "phone_learning": EventSearchTerms(
        "使用手机学习",
        (
            "用手机查学习资料",
            "用手机查资料",
            "手机查资料",
            "用手机学习",
            "手机学习",
            "手机搜题",
        ),
    ),
    "computer_learning": EventSearchTerms(
        "使用电脑学习",
        ("用电脑学习", "电脑查资料", "电脑学习", "上网课"),
    ),
    "other_study_behavior": EventSearchTerms(
        "其他学习行为",
        ("其他学习行为", "其它学习行为"),
    ),
    "phone_distraction": EventSearchTerms(
        "手机分心",
        ("手机分心", "玩手机", "刷手机", "看手机", "使用手机", "手机"),
    ),
    "computer_distraction": EventSearchTerms(
        "电脑分心",
        ("电脑分心", "玩电脑游戏", "电脑游戏", "玩电脑", "打游戏"),
    ),
    "communication_distraction": EventSearchTerms(
        "交流分心",
        ("与人交流", "交流分心", "和别人聊天", "聊天", "说话"),
    ),
    "study_end_cleanup": EventSearchTerms(
        "学习结束整理",
        ("学习结束整理", "收拾学习用品", "收拾书本", "整理桌面"),
    ),
}


SUPPORTED_EVENT_TYPES = tuple(ALL_EVENTS)

_missing_event_types = set(SUPPORTED_EVENT_TYPES) - set(EVENT_SEARCH_TERMS)
_extra_event_types = set(EVENT_SEARCH_TERMS) - set(SUPPORTED_EVENT_TYPES)
if _missing_event_types or _extra_event_types:
    raise RuntimeError(
        "Retrieval 事件词典与 A 模块 ALL_EVENTS 不一致："
        f"missing={sorted(_missing_event_types)}, "
        f"extra={sorted(_extra_event_types)}"
    )


def match_event_type(text: str) -> str | None:
    """从事件名称或常见中文问法中识别正式事件名称。

    匹配时优先采用最长短语，避免“用手机查学习资料”被较短的“手机”误判为
    ``phone_distraction``。
    """

    normalized = "".join(text.lower().split())
    best_event_type: str | None = None
    best_length = 0

    for event_type in SUPPORTED_EVENT_TYPES:
        terms = EVENT_SEARCH_TERMS[event_type]
        candidates = (event_type, terms.label, *terms.aliases)
        for candidate in candidates:
            compact_candidate = "".join(candidate.lower().split())
            if compact_candidate in normalized and len(compact_candidate) > best_length:
                best_event_type = event_type
                best_length = len(compact_candidate)

    return best_event_type


def build_event_embedding_text(event_type: str, description: str = "") -> str:
    """生成供任意 Embedding 实现编码的稳定检索文本。"""

    normalized_event_type = event_type.strip()
    normalized_description = description.strip()
    terms = EVENT_SEARCH_TERMS.get(normalized_event_type)

    parts = [f"事件类型：{normalized_event_type}"]
    if terms is not None:
        parts.append(f"中文含义：{terms.label}")
        parts.append(f"相关表达：{'、'.join(terms.aliases)}")
    if normalized_description:
        parts.append(f"事件描述：{normalized_description}")
    return "；".join(parts)
