"""Qwen2-VL 的八事件三步判断协议。"""

from __future__ import annotations

from ..event.event_types import ALL_EVENTS


EVENT_TYPES = list(ALL_EVENTS)
DISABLED_EVENT_TYPES: frozenset[str] = frozenset()

EVENT_TYPE_CN = {
    "sit_at_study_position": "坐到学习位置",
    "leave_study_position": "离开学习位置",
    "reading": "阅读",
    "writing": "书写",
    "phone_usage": "使用手机",
    "computer_usage": "使用电脑",
    "communication_distraction": "交流分心",
    "other_behavior": "其他",
}

EVENT_FOCUS = {
    "sit_at_study_position": "人物是否从别处来到桌前并坐下；一直坐着不算。",
    "leave_study_position": "人物是否从桌前起身离开；一直不在座位不算。",
    "reading": "人物是否持续看书或阅读资料；阅读中的短暂翻页仍算阅读。",
    "writing": "人物是否持续落笔写字、做题或记录。",
    "phone_usage": "人物是否持续注视或操作手机；只拿起、移动或放下不算。",
    "computer_usage": "人物是否持续注视电脑屏幕、操作键盘或鼠标；打开、移动或收起不算。",
    "communication_distraction": "人物是否与他人交谈、通话并偏离原动作。",
    "other_behavior": "无法判断，或正在准备、收拾、整理、摆放和切换学习用品。",
}


_OBJECT_CLUE_CN = {
    "reading": "书本相关物体",
    "writing": "书写工具",
    "phone_usage": "手机",
    "computer_usage": "电脑/键盘/鼠标",
    "communication_distraction": "交流相关线索",
}


def _candidate_context(
    candidate_event_type: str,
    candidate_scores: dict[str, float] | None,
) -> tuple[str, str]:
    scores = {
        str(name): float(score)
        for name, score in (candidate_scores or {}).items()
        if str(name) in EVENT_TYPES
        and str(name) != "other_behavior"
    }
    evidence_text = (
        "，".join(
            f"{_OBJECT_CLUE_CN.get(name, name)}约出现在{score:.0%}的分析帧"
            for name, score in scores.items()
        )
        if scores
        else "没有具体物体候选"
    )
    candidate = candidate_event_type
    if candidate == "sit_at_study_position":
        candidate_text = "位置状态机提示可能发生入座，仍须根据连续画面核实"
    elif candidate == "leave_study_position":
        candidate_text = "位置状态机提示可能发生离座，仍须根据连续画面核实"
    else:
        candidate_text = "不提供预设事件结论"
    return candidate_text, evidence_text


def build_activity_prompt(
    candidate_event_type: str,
    frame_count: int | None = None,
    candidate_scores: dict[str, float] | None = None,
    unclassified_ratio: float | None = None,
) -> str:
    """构造短而严格的三步 Prompt，适配 Qwen2-VL-2B。"""
    candidate = candidate_event_type
    if candidate not in EVENT_TYPES:
        raise ValueError(
            f"Unknown event_type: {candidate_event_type!r}. Expected one of {EVENT_TYPES!r}"
        )
    frame_count = int(frame_count or 0)
    frame_rule = (
        f"共 {frame_count} 张图，编号 1-{frame_count}。"
        if frame_count > 0
        else "图片编号从 1 开始。"
    )
    candidate_text, evidence_text = _candidate_context(candidate, candidate_scores)
    unknown_text = (
        f"约 {float(unclassified_ratio):.0%} 的分析帧没有具体物体证据。"
        if unclassified_ratio is not None
        else ""
    )

    return f"""
观察按时间排列的视频关键帧，严格完成三步判断，最后只输出一个 JSON。
上游提示：{candidate_text}。YOLO只提供物体线索：{evidence_text}。{unknown_text}
这些线索彼此平等且都不是事件结论。物体出现不等于人物正在使用它，必须以连续动作和前后变化为准。

第一步 objective_description：
- 客观描述人物实际动作，按时间顺序，最多80个汉字。
- 只写看到的正向事实，不写事件英文名，不罗列“没有做什么”，不要重复。

第二步 events：只判断以下八类：
- sit_at_study_position：从别处到桌前并坐下；一直坐着不算。
- leave_study_position：从桌前起身离开；一直不在不算。
- reading：持续注视书本或文字资料；正常阅读中的短暂翻页仍算阅读。刚拿出书、打开书、寻找页码或翻到目标页，但尚未持续阅读，不算阅读。
- writing：持续看到笔接触纸面并写字、做题或记录；只拿笔、笔袋或摆放纸张不算。
- phone_usage：持续注视、滑动、点击或操作手机；只拿起、移动或放下手机不算。
- computer_usage：持续注视电脑屏幕、操作键盘或鼠标；只打开、合上、移动或收起电脑不算。
- communication_distraction：与他人交谈、通话并偏离原动作。
- other_behavior：无法判断具体动作，或正在收拾、整理、摆放、拿出/收起学习用品，以及两个主要事件之间的准备和切换过程。

规则：
- 同一窗口先后出现多个动作时，对应 events 可以同时为 true，但必须分别填写 activity_segments。
- other_behavior 可以和具体事件在同一窗口的不同时间片中同时为 true，但它们的 activity_segments 不得重叠；同一时间片只能选一类。
- 上下文规则：reading→短暂翻页→reading 仍算 reading；非阅读状态→拿书/翻到目标页→开始阅读，前面的准备片段算 other_behavior。
- 上下文规则：computer_usage→收起电脑/拿出笔袋或书本→writing，中间整理片段算 other_behavior。
- primary_event 必须是占主要时间的 true 事件；没有任何 true 时填 none。
- activity_segments 标出每个 true 事件的图片区间。{frame_rule}
- 只要动作发生变化或出现整理/准备过程，activity_segments 就是必填项，并应覆盖每个阶段；同一动作中断后再次出现，写成两个 segment。

第三步 final_description：
- 在第一步描述基础上做最小改写，自然融入所有 true 事件的中文名。
- 保留动作顺序和画面细节，不新增事实，不改变 events，不在句尾机械罗列标签。
- 没有 true 时与 objective_description 保持一致。

只输出合法 JSON，不要输出 Markdown 或解释：
{{
  "objective_description": "不超过80字的客观动作描述",
  "final_description": "基于客观描述最小改写后的描述",
  "primary_event": "none",
  "observed_activities": [],
  "activity_segments": [],
  "event_confirmed": false,
  "is_phone_usage": false,
  "is_studying": false,
  "events": {{
    "sit_at_study_position": false,
    "leave_study_position": false,
    "reading": false,
    "writing": false,
    "phone_usage": false,
    "computer_usage": false,
    "communication_distraction": false,
    "other_behavior": false
  }}
}}
event_confirmed 等于 events 中是否存在 true。输出首字符必须是 {{，末字符必须是 }}。
"""


def build_caption_prompt(event_type: str) -> str:
    return build_activity_prompt(event_type)


def build_all_events_prompt() -> str:
    return build_activity_prompt("other_behavior")


CLASSIFY_PROMPT = build_activity_prompt("other_behavior")
