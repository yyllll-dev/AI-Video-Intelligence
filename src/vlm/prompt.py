"""Qwen2-VL 的八事件三步判断协议。"""

from __future__ import annotations

from typing import Any

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
    "communication_distraction": "至少两人同框并朝向彼此；出现交谈、眼神交流、回应、倾听或轮流互动任一项即可。",
    "other_behavior": "无法判断，或正在准备、收拾、整理、摆放和切换学习用品。",
}


def build_video_summary_prompt(records: list[dict[str, Any]]) -> str:
    """根据已确认事件及其描述构造全视频总结 Prompt。"""
    if not records:
        raise ValueError("records 不能为空")

    timeline = "\n".join(
        f"{index}. {float(record['start_time']):.2f}s-"
        f"{float(record['end_time']):.2f}s | "
        f"{EVENT_TYPE_CN.get(str(record['event_type']), str(record['event_type']))} | "
        f"{str(record.get('caption', '')).strip()}"
        for index, record in enumerate(records, start=1)
    )
    return f"""你是学习行为视频总结助手。下面是系统已经确认的完整事件时间线，每条包含起止时间、事件类型和该事件的视觉描述。

{timeline}

请综合全部事件生成一段自然、客观、适合展示给评委的中文总结：
- 必须覆盖主要行为、行为先后顺序和明显的状态变化。
- 可以概括重复事件，但不得遗漏重要事件类型。
- 只能使用上面提供的事实，不得虚构人物身份、物品、原因、情绪或学习效果。
- 不评价好坏，不提模型、检测、事件标签、置信度或“根据记录”。
- 控制在 120 至 220 个汉字；事件很少时可以更短。
- 直接输出总结正文，不要标题、列表、Markdown、JSON 或额外解释。"""


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
观察按时间排列的视频关键帧，严格完成三步判断，最后只输出一个短 JSON。
上游提示：{candidate_text}。YOLO只提供物体线索：{evidence_text}。{unknown_text}
这些线索彼此平等且都不是事件结论。物体出现不等于人物正在使用它，必须以连续动作和前后变化为准。

第一步 objective_description：
- 只描述与事件判断有关的核心动作和时间先后，最多40个汉字。
- 禁止描述性别、年龄、外貌、眼镜、服装、颜色、背景、房间或交通场景。
- 不写事件英文名，不罗列“没有做什么”，不要重复。

第二步 events/event_labels：从以下八类中选择实际发生的事件：
- sit_at_study_position：从别处到桌前并坐下；一直坐着不算。
- leave_study_position：从桌前起身离开；一直不在不算。
- reading：持续注视书本或文字资料；正常阅读中的短暂翻页仍算阅读。刚拿出书、打开书、寻找页码或翻到目标页，但尚未持续阅读，不算阅读。
- writing：持续看到笔接触纸面并写字、做题或记录；只拿笔、笔袋或摆放纸张不算。
- phone_usage：持续注视、滑动、点击或操作手机；只拿起、移动或放下手机不算。
- computer_usage：持续注视电脑屏幕、操作键盘或鼠标；只打开、合上、移动或收起电脑不算。
- communication_distraction：至少两人同框并朝向彼此，出现交谈、眼神交流、回应、倾听或轮流互动中的任一项即可；不要求持续说话，也不要求明确中断原动作。交流时即使桌上有书或电脑，也优先算交流分心。仅路过或各做各的不算。
- other_behavior：无法判断具体动作，或正在收拾、整理、摆放、拿出/收起学习用品，以及两个主要事件之间的准备和切换过程。

规则：
- 同一窗口先后出现多个动作时，event_labels 可包含多项，但必须分别填写 activity_segments。
- other_behavior 可以和具体事件在同一窗口的不同时间片中同时为 true，但它们的 activity_segments 不得重叠；同一时间片只能选一类。
- 上下文规则：reading→短暂翻页→reading 仍算 reading；非阅读状态→拿书/翻到目标页→开始阅读，前面的准备片段算 other_behavior。
- 上下文规则：computer_usage→收起电脑/拿出笔袋或书本→writing，中间整理片段算 other_behavior。
- primary_event 只是日志摘要：填写占主要时间的 true 事件；没有时填 none。
- primary_event 不决定时间轴，也不能代替 activity_segments。
- activity_segments 标出每个 true 事件的图片区间。{frame_rule}
- 只要动作发生变化或出现整理/准备过程，activity_segments 就是必填项，并应覆盖每个阶段；同一动作中断后再次出现，写成两个 segment。

第三步 activity_segments：
- 有动作变化时按帧号填写分段；整窗只有一个一致动作时留空。
- 再填写只供日志摘要使用的 primary_event。

只输出以下字段的合法 JSON，不要补充字段、Markdown或解释。event_labels 没有事件时为空数组：
{{
  "objective_description": "不超过40字的核心动作和先后顺序",
  "primary_event": "none",
  "activity_segments": [],
  "event_labels": []
}}
输出首字符必须是 {{，末字符必须是 }}。
"""


def build_caption_prompt(event_type: str) -> str:
    return build_activity_prompt(event_type)


def build_all_events_prompt() -> str:
    return build_activity_prompt("other_behavior")


CLASSIFY_PROMPT = build_activity_prompt("other_behavior")


def build_frame_labels_prompt(
    frame_count: int,
    allowed_event_types: list[str] | tuple[str, ...] | None = None,
) -> str:
    """构造边界复核 Prompt；只让 2B 模型逐帧输出一个正式事件。"""
    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count 必须大于 0")
    selected = list(dict.fromkeys(allowed_event_types or EVENT_TYPES))
    if not selected or any(name not in EVENT_TYPES for name in selected):
        raise ValueError("allowed_event_types 包含未知事件")
    allowed = ", ".join(selected)
    return f"""按时间判断这 {frame_count} 张视频帧，每帧只选一个标签。
本窗口只允许：{allowed}
坐到学习位置必须看到站立/走近到坐下的变化；静态坐着不是入座。
拿书找页、收电脑、拿笔袋、整理摆放、动作切换和不完整收尾选 other_behavior。
至少两人同框并朝向彼此，出现交谈、眼神交流、回应、倾听或轮流互动任一项时选 communication_distraction；仅路过或各做各的不算。
阅读中的短暂翻页仍选 reading。只有真正落笔才选 writing。
不要按标签列表顺序轮流填写。相邻画面动作没变时必须保持同一标签。
只输出一个 JSON 对象：键名为 frame_labels，值为恰好 {frame_count} 个标签的数组。不要解释。"""


def build_window_label_prompt(
    allowed_event_types: list[str] | tuple[str, ...] | None = None,
) -> str:
    """构造整窗单标签复核 Prompt，供主 JSON 失败时使用。"""
    selected = list(dict.fromkeys(allowed_event_types or (
        "reading",
        "writing",
        "phone_usage",
        "computer_usage",
        "communication_distraction",
        "other_behavior",
    )))
    if not selected or any(name not in EVENT_TYPES for name in selected):
        raise ValueError("allowed_event_types 包含未知事件")
    allowed = ", ".join(selected)
    return f"""按时间观察全部视频帧，只判断整个窗口的持续动作。
只能选择以下一个英文标签：{allowed}
reading：眼睛持续看印刷书页；短暂翻页仍是阅读。
writing：清楚看到手持笔，笔尖接触纸面并持续移动书写；桌面有书不等于阅读。
computer_usage：持续看电脑屏幕，或手在键盘、触控板、鼠标上操作；笔记本电脑是电脑，不是书。
phone_usage：持续看或操作手机。
communication_distraction：至少两人同框并朝向彼此，出现交谈、眼神交流、回应、倾听或轮流互动任一项即可；不要求持续说话。交流时即使桌上有书或电脑也选此项。仅路过或各做各的不算。
other_behavior：拿出、移动、收起、整理物品，准备/切换动作，动作混合或无法确认。
    若前后动作不一致，选择 other_behavior。不要描述场景，不要 JSON，不要解释，只输出一个允许的英文标签。"""


def build_first_transition_frame_prompt(
    frame_count: int,
    from_event_type: str,
    to_event_type: str,
) -> str:
    """构造只定位新动作首次稳定出现帧号的短 Prompt。"""
    frame_count = int(frame_count)
    if frame_count <= 1:
        raise ValueError("frame_count 必须大于 1")
    if from_event_type not in EVENT_TYPES or to_event_type not in EVENT_TYPES:
        raise ValueError("from/to event_type 包含未知事件")
    target_rule = EVENT_FOCUS[to_event_type]
    return f"""按时间观察编号 1-{frame_count} 的视频帧。
窗口正在从 {from_event_type} 过渡到 {to_event_type}。
目标动作标准：{target_rule}
找出 {to_event_type} 第一次已经明确、稳定开始的帧；仅有相关物体、拿取、打开、摆放或准备动作不算开始。
如果全部帧都没有稳定开始，输出 0。不要描述场景，不要 JSON，不要解释，只输出 0-{frame_count} 的一个整数。"""


def build_transition_presence_prompt(
    frame_count: int,
    to_event_type: str,
) -> str:
    """构造逐帧二值复核 Prompt，作为首帧号输出失败时的后备。"""
    frame_count = int(frame_count)
    if frame_count <= 1:
        raise ValueError("frame_count 必须大于 1")
    if to_event_type not in EVENT_TYPES:
        raise ValueError("to_event_type 包含未知事件")
    target_rule = EVENT_FOCUS[to_event_type]
    return f"""按顺序判断 {frame_count} 张视频帧中是否已经明确进行 {to_event_type}。
判断标准：{target_rule}
每帧只能写 0 或 1：准备、拿取、摆放或看不清写 0；目标动作已明确进行写 1。
严格输出连续 {frame_count} 位数字，例如 000111111。不要 JSON、标点、空格或解释。"""


def build_position_transition_prompt(event_type: str) -> str:
    """位置变化专用短 Prompt，避免普通动作描述漏掉动态入座/离座。"""
    if event_type == "sit_at_study_position":
        question = "连续画面是否明确表现人物从站立/走近学习位置到坐下？"
        positive = "standing_to_sitting"
        static = "static_sitting"
    elif event_type == "leave_study_position":
        question = "连续画面是否明确表现人物从学习位置坐着到起身离开？"
        positive = "sitting_to_leaving"
        static = "static_away"
    else:
        raise ValueError(f"不是位置变化事件: {event_type}")
    return f"""{question}
只判断前后位置变化，不描述人物、服装、颜色、背景或其他动作。
position_change 只能是 {positive}、{static}、none。
只有连续帧明确显示完整变化才选 {positive}；只看到最终状态选 {static}。
只输出合法 JSON：{{"position_change":"none"}}"""


def build_position_endpoint_prompt(event_type: str, edge: str) -> str:
    """构造位置事件首尾姿态校验 Prompt。"""
    if event_type not in {"sit_at_study_position", "leave_study_position"}:
        raise ValueError(f"不是位置变化事件: {event_type}")
    if edge not in {"start", "end"}:
        raise ValueError("edge 必须是 start 或 end")
    edge_cn = "窗口开头" if edge == "start" else "窗口结尾"
    return f"""这些图片只来自{edge_cn}，判断人物相对学习座椅的姿态。
standing：人物站立或正在走动；sitting：人物臀部已在椅子上并稳定坐着；away：人物已经离开；unclear：看不清。
不要根据问题猜测动作，不描述场景，不输出 JSON，只输出 standing、sitting、away、unclear 中的一个。"""
