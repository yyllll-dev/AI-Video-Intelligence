"""
inference.py —— Qwen-VL 推理逻辑

B模块职责：
    1. 接收 A / EventEngine 生成的宽松活动候选 Event
    2. 接收该事件时间窗口内的多帧图片
    3. 调用 Qwen-VL 分析真实视觉动作
    4. 补充 Event.description
    5. 返回结构化主活动、活动顺序和关键帧区间

正式接口：

    analyze_event(
        model,
        processor,
        event,
        frame_paths,
    ) -> Event

正式流程：

    EventEngine
        ↓
    Event(event_type=...)
        +
    多帧图片
        ↓
    Qwen-VL
        ↓
    description
        ↓
    Event

2026-09-02 与A讨论后确定的两条架构结论：

    ① event_confirmed=false 之后怎么处理？
       不进入正式Video Memory，仅保留debug/log信息。
       职责划分：VLM（B）负责"确认事件"，D（Memory）只存储
       "已经确认"的事件，E（UI）只展示"已经确认"的事件。
       对应实现：should_persist_to_memory() / _write_debug_log()。

    ② 一个窗口横跨两个动作怎么办？
       Event Engine负责"切事件"，VLM负责"验事件"：
       event_confirmed=true的门槛改为"这个事件是否为当前时间
       窗口内的主要/主导行为"，而不是"动作是否在某一帧里出现
       过"。对应实现：prompt.py中build_caption_prompt()第二节
       第4条的判定规则。
"""

import argparse
from difflib import SequenceMatcher
import json
import re
import time
from pathlib import Path
from typing import Any

from .qwen_vlm import load_model
from .prompt import (
    build_activity_prompt,
    EVENT_TYPES,
    EVENT_TYPE_CN,
)
from ..event.schemas import Event


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Qwen-VL 多帧推理
# ============================================================

def run_vlm(
    model,
    processor,
    image_paths: list[str],
    prompt: str,
    do_sample: bool = False,
) -> str:
    """
    输入：
        image_paths：
            多帧图片路径，必须按照时间顺序排列。

        prompt：
            本次Qwen-VL分析使用的Prompt。

        do_sample：
            2026-09-02 新增，默认False（贪婪解码/确定性生成）。

            背景：
                原代码调用model.generate()时完全没有指定
                do_sample/temperature，会直接沿用模型自带的
                generation_config默认值——Qwen2-VL系列的默认
                配置通常是开启采样的（do_sample=True，带
                temperature/top_p）。这意味着即使Prompt和输入
                图片完全不变，每次生成结果也会不一样。

                实测现象也印证了这一点：同一组5张帧，14次调用
                里出现了"电脑前有手机""正在阅读学习资料""桌上
                放有书本"等好几种互不相同、画面里都不存在的
                物体，且各自集中出现在不同的event_type调用里
                ——这不是"模型看错了画面"，而是"模型每次都在
                随机生成一套看起来合理的细节"，是采样随机性
                主导的幻觉，光靠改Prompt措辞压不住。

                把do_sample改成False（贪婪解码）后，模型每次
                都会选择概率最高的token，生成结果对同一输入
                基本确定、且更贴近"最保守、最不容易无中生有"
                的输出，是目前对2B模型幻觉问题性价比最高的
                单项修复。

            如果确实需要采样（比如以后想要多样性），可以显式
            传入 do_sample=True，此时应同时传入合理的
            temperature，不要沿用模型的隐式默认值。

    输出：
        Qwen-VL生成的原始文本。

    注意：
        本函数只负责调用模型，
        不负责解析JSON，
        不负责修改Event。
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "缺少 qwen-vl-utils；只有启用 Qwen-VL 推理时才需要安装"
        ) from exc

    if not image_paths:
        raise ValueError(
            "image_paths cannot be empty"
        )

    content = []

    total = len(image_paths)

    for idx, path in enumerate(image_paths, start=1):

        image_path = Path(path)

        if not image_path.is_absolute():
            project_relative_path = _PROJECT_ROOT / image_path
            if project_relative_path.exists():
                image_path = project_relative_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file does not exist: {path}"
            )

        # 告诉模型当前图片在时间序列中的位置
        content.append(
            {
                "type": "text",
                "text": f"【第{idx}帧，共{total}帧】",
            }
        )

        content.append(
            {
                "type": "image",
                "image": str(image_path),
            }
        )

    # Prompt放在所有图片之后
    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # ========================================================
    # 构造Qwen-VL输入
    # ========================================================

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 将输入移动到模型所在设备
    inputs = inputs.to(model.device)

    # ========================================================
    # 模型生成
    #
    # 2026-09-02：显式传入do_sample（默认False）。
    # 同时显式传入temperature/top_p/top_k=None，
    # 避免transformers在do_sample=False时因为
    # generation_config里残留的采样参数而打印警告，
    # 也避免未来有人不小心把do_sample改成True后，
    # 又意外沿用了模型自带的、我们完全不了解的采样参数。
    # ========================================================

    generate_kwargs: dict[str, Any] = {
        # 8 个布尔值加少量分段在 512 token 内足够。限制输出长度并增加
        # 重复惩罚，避免 2B 模型循环复读否定句直到 JSON 被截断。
        "max_new_tokens": 512,
        "do_sample": do_sample,
        "repetition_penalty": 1.10,
    }

    if not do_sample:
        # 贪婪解码下，temperature/top_p/top_k无意义，
        # 显式关闭以覆盖generation_config里的默认值，
        # 避免出现"do_sample=False但底层配置仍带
        # temperature"的警告或不确定行为。
        generate_kwargs["temperature"] = None
        generate_kwargs["top_p"] = None
        generate_kwargs["top_k"] = None
        generate_kwargs["num_beams"] = 1

    generated_ids = model.generate(
        **inputs,
        **generate_kwargs,
    )

    # 去掉输入Prompt对应的token
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if not output_text:
        return ""

    return output_text[0].strip()


# ============================================================
# JSON解析
# ============================================================

def _compact_description(text: str, max_chars: int = 120) -> str:
    """压缩模型描述：去重退化句、移除已知模板复读并限制长度。"""
    sentences = re.split(r"(?<=[。！？!?])", text)
    unique_sentences: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        sentence = sentence.strip()
        if (
            not sentence
            or sentence in seen
            or "没有在学习结束" in sentence
            or "没有在学习准备" in sentence
        ):
            continue
        seen.add(sentence)
        unique_sentences.append(sentence)
        if len("".join(unique_sentences)) >= max_chars:
            break
    compact = "".join(unique_sentences).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip("，,；; ") + "。"
    return compact


def _salvage_string_field(
    raw_text: str,
    field_name: str,
    max_chars: int = 120,
) -> str:
    """从未闭合的 Qwen JSON 中抢救一个字符串字段。"""
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"', raw_text)
    if match is None:
        return ""

    tail = raw_text[match.end():]
    characters: list[str] = []
    escaped = False
    for character in tail:
        if escaped:
            characters.append(character)
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            break
        characters.append(character)

    text = "".join(characters).replace("\\n", " ").strip()
    return _compact_description(text, max_chars)


def _salvage_description(raw_text: str, max_chars: int = 120) -> str:
    """兼容新旧协议，从未闭合 JSON 中优先抢救客观描述。"""
    for field_name in (
        "objective_description",
        "final_description",
        "description",
    ):
        description = _salvage_string_field(raw_text, field_name, max_chars)
        if description:
            return description
    return "[VLM未返回有效description]"


def parse_json(raw_text: str) -> dict[str, Any]:
    """
    解析Qwen-VL输出的JSON。

    Qwen-VL正常情况下应该直接返回JSON，
    这里增加容错，避免模型偶尔输出额外文字导致解析失败。

    返回字段至少保证：
        event_confirmed
        objective_description
        final_description
        description
        is_phone_usage
        is_studying
    """

    raw_text = raw_text.strip()

    # --------------------------------------------------------
    # 情况1：标准 JSON（先去掉 markdown 代码块包裹）
    # --------------------------------------------------------
    stripped = raw_text.strip()
    # 去掉 ```json ... ``` 包裹
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        clean_lines = [
            line for line in lines
            if not line.strip().startswith("```")
        ]
        stripped = "\n".join(clean_lines).strip()

    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Qwen2-VL-2B 偶尔会输出完整字段，却漏掉最外层对象括号，例如直接从
    # "description": ... 开始。必须先修复整个顶层片段；否则下面的括号
    # 提取会误把 activity_segments 中的第一个小对象当成完整回答。
    repaired = _repair_missing_outer_object(stripped)
    if repaired is not None:
        try:
            result = json.loads(repaired)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # 情况2：提取第一个括号配平的 JSON 对象。不能使用非贪婪正则，
    # 因为当前协议包含嵌套的 events 对象，会在内层右括号处被截断。
    # --------------------------------------------------------
    candidate = _extract_first_json_object(stripped)
    if candidate:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # 情况3：多行 JSON（逐行拼合）
    # --------------------------------------------------------
    lines = stripped.split("\n")
    candidates = []
    for line in lines:
        line = line.strip()
        if not line or line == '"' or line.count('"') < 3:
            continue
        candidates.append(line)

    for candidate in candidates[:3]:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    combined = " ".join(candidates[:5])
    if combined.strip():
        try:
            result = json.loads(combined)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # 情况4：解析失败
    # --------------------------------------------------------

    salvaged_objective = _salvage_description(raw_text)
    salvaged_final = _salvage_string_field(raw_text, "final_description")
    return {
        "event_confirmed": False,
        "objective_description": salvaged_objective,
        "final_description": salvaged_final,
        "description": salvaged_final or salvaged_objective,
        "is_phone_usage": False,
        "is_studying": False,
        "_parse_error": "VLM 输出不是闭合的合法 JSON",
    }


def _extract_first_json_object(text: str) -> str | None:
    """提取首个完整 JSON 对象，正确处理嵌套对象及字符串中的括号。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _repair_missing_outer_object(text: str) -> str | None:
    """修复模型漏写最外层 `{`、`}` 的顶层 JSON 字段片段。"""
    first_quote = text.find('"')
    if first_quote < 0:
        return None
    fragment = text[first_quote:].strip()
    fragment = re.sub(r"(?:\r?\n)?\s*[-`]+\s*$", "", fragment).strip()
    if not fragment.startswith('"'):
        return None
    has_description_field = any(
        f'"{field_name}"' in fragment
        for field_name in (
            "objective_description",
            "final_description",
            "description",
        )
    )
    if not has_description_field or '"events"' not in fragment:
        return None

    balance = 0
    in_string = False
    escaped = False
    for char in fragment:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            balance += 1
        elif char == "}":
            balance -= 1

    # 前面补一个顶层左括号。balance=-1 表示模型保留了顶层右括号；
    # balance=0 表示顶层左右括号都漏掉了，需要再补一个右括号。
    missing_closing = balance + 1
    if missing_closing < 0:
        return None
    return "{" + fragment + ("}" * missing_closing)


# ============================================================
# 批量模式专用：解析"14件事全判"输出
# ============================================================

def parse_json_multi_event(raw_text: str) -> dict[str, Any]:
    """
    解析批量模式（test_all_event_types）下的VLM输出。

    新格式（2026-09-02）：
    {
        "description": "...",
        "events": {
            "sit_at_study_position": true/false,
            ... 8个事件
        },
        "is_phone_usage": true/false,
        "is_studying": true/false
    }

    返回：
        与 parse_json() 兼容的 dict，额外保留 events 字段
    """

    raw_text = raw_text.strip()

    # 复用通用JSON解析
    parsed = parse_json(raw_text)

    events_dict = parsed.get("events", {})

    # 容错：events可能缺失或不是dict
    if not isinstance(events_dict, dict):
        events_dict = {}

    # 规范化每个事件字段为bool
    normalized_events: dict[str, bool] = {}
    for event_type in EVENT_TYPES:
        normalized_events[event_type] = _to_bool(
            events_dict.get(event_type, False)
        )

    parsed["events"] = normalized_events

    return parsed


# ============================================================
# 输出字段规范化（新增）
#
# 背景：
#     parse_json() 只处理“JSON本身解析失败”的情况。
#     但Qwen-VL偶尔会出现“JSON合法，但字段类型不对”的情况，例如：
#         "is_phone_usage": "true"      （字符串而不是bool）
#         "is_phone_usage": 1           （数字而不是bool）
#         缺少某个字段
#         description 为 null 或非字符串
#     这类脏数据如果直接透传给D（Retrieval/Memory）或E（UI），
#     会导致后续模块类型判断出错，且很难排查。
#
# 因此这里统一做一次“字段规范化”，
# 保证不管模型输出多不规范，
# 最终拿到的都是类型正确、字段齐全的结果。
# ============================================================

def _to_bool(value: Any) -> bool:
    """将各种可能的“类真值”输出统一转换为标准bool。"""

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in (
            "true", "1", "yes", "是", "真",
        )

    return False


# ============================================================
# 描述内容校验关键词
#
# 用 description 全文是否包含对应事件的特征词，
# 对 VLM 的 confirmed 输出做一次内容一致性校验。
# 关键词取中文描述里最可能出现的动作词/名词。
# ============================================================
EVENT_CONTENT_KEYWORDS: dict[str, list[str]] = {
    "sit_at_study_position": [
        "坐下", "坐在", "坐着", "入座", "落座", "到桌前", "走到", "移到", "靠近"
    ],
    "leave_study_position": ["离开", "起身", "站起来", "站起", "走开", "走离"],
    "reading": [
        "持续看书", "专注看书", "看书", "读书", "看资料", "阅读",
        "看教材", "阅读书籍", "阅读资料", "注视书页", "浏览文字",
    ],
    "writing": ["写字", "记笔记", "做笔记", "做题", "画图", "书写", "写笔记"],
    "phone_usage": [
        "使用手机", "操作手机", "看手机", "查看手机", "注视手机",
        "滑动手机", "点击手机",
    ],
    "computer_usage": [
        "使用电脑", "使用笔记本电脑", "使用笔记本", "操作电脑", "操作笔记本电脑",
        "看着电脑屏幕", "注视电脑屏幕", "操作键盘", "操作鼠标",
        "看电脑", "点击电脑", "敲键盘", "使用键盘", "使用鼠标",
    ],
    "communication_distraction": ["说话", "通话", "讨论"],
    "other_behavior": [
        "整理", "收拾", "摆放", "归位", "拿出书", "取出书", "收起书",
        "收好书", "打开书", "翻到", "寻找页码", "拿出笔袋", "打开笔袋",
        "收起笔袋", "拿出电脑", "打开电脑", "合上电脑", "关闭电脑",
        "移动电脑", "收起电脑", "收好电脑", "放回", "放好",
        "准备用品", "切换物品", "动作看不清", "无法判断",
    ],
}

# 与 EventEngine 共用当前完整事件清单。顺序也作为缺少 segments 时的稳定兜底顺序。
_ACTIVITY_RECLASSIFICATION_ORDER = tuple(EVENT_TYPES)

# 这些事件表达的是状态变化，不能只凭某一帧里的静态状态成立。
# VLM 若要把上游候选改判成这些事件，必须给出合法时间片；只有
# EventEngine 已经给出同类候选时，候选窗口本身才可作为时序证据。
_TRANSITION_EVENT_TYPES = {
    "sit_at_study_position",
    "leave_study_position",
}

_PHONE_EVENT_TYPES = {"phone_usage"}
_STUDY_EVENT_TYPES = {
    "reading",
    "writing",
    "computer_usage",
    "phone_usage",
    "other_behavior",
}

# 旧版自然语言兼容入口只推断具体行为。位置词（例如“坐在”）太宽泛，
# 若混入这里会抢先吞掉“坐在桌前写字”中的真正行为。
_DESCRIPTION_ACTIVITY_ORDER = (
    "writing",
    "reading",
    "phone_usage",
    "computer_usage",
    "communication_distraction",
    "other_behavior",
)

_EVENT_TYPE_BY_CN = {
    event_cn: event_type
    for event_type, event_cn in EVENT_TYPE_CN.items()
}


def _normalize_event_type_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip()
    if name in EVENT_TYPES:
        return name
    return _EVENT_TYPE_BY_CN.get(name)


def _event_type_name(value: Any) -> str | None:
    """只接受当前八事件协议中的正式名称。"""
    return _normalize_event_type_name(value)


def infer_activity_from_description(description: str) -> str | None:
    """从 VLM 描述中的明确动作词纠正 YOLO 产生的活动候选类型。"""
    if not isinstance(description, str):
        return None
    # 过渡动作不是“低优先级未知项”。当描述明确说正在拿出、收起、
    # 整理或切换物品时，不能再被同一句中的 book/laptop 名词抢回阅读/电脑。
    if description_has_transition_evidence(description):
        return "other_behavior"
    for event_type in _DESCRIPTION_ACTIVITY_ORDER:
        if event_type == "other_behavior":
            continue
        if _description_supports_event(description, event_type):
            return event_type
    return None


def description_has_transition_evidence(description: str) -> bool:
    """是否明确描述了准备、收拾或事件切换动作。"""
    return isinstance(description, str) and _description_supports_event(
        description,
        "other_behavior",
    )


def _description_supports_event(description: str, event_type: str) -> bool:
    """仅供“从自然语言描述兜底推断”使用，不否定结构化 events。"""
    negations = ("没有", "并未", "未在", "未", "不是", "不再", "无")
    for keyword in EVENT_CONTENT_KEYWORDS.get(event_type, []):
        start = 0
        while True:
            index = description.find(keyword, start)
            if index < 0:
                break
            prefix = description[max(0, index - 5):index]
            if not any(prefix.endswith(negation) for negation in negations):
                return True
            start = index + len(keyword)
    return False


def infer_activity_from_vlm_meta(meta: dict[str, Any]) -> str | None:
    """返回 VLM 确认的第一个活动，兼容原有单事件调用方。"""
    activities = infer_activities_from_vlm_meta(meta)
    return activities[0] if activities else None


def infer_activities_from_vlm_meta(meta: dict[str, Any]) -> list[str]:
    """返回按画面发生顺序排列的全部可信活动。"""
    raw_events = meta.get("events", {})
    events: dict[str, bool] = {}
    if isinstance(raw_events, dict):
        for raw_name, value in raw_events.items():
            name = _event_type_name(raw_name)
            if name in _ACTIVITY_RECLASSIFICATION_ORDER:
                events[name] = events.get(name, False) or _to_bool(value)

    activities: list[str] = []
    segments = meta.get("activity_segments", [])
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            event_type = _event_type_name(segment.get("event_type"))
            if (
                event_type in _ACTIVITY_RECLASSIFICATION_ORDER
                and _to_bool(events.get(event_type, False))
                and event_type not in activities
            ):
                activities.append(event_type)

    observed = meta.get("observed_activities", [])
    if isinstance(observed, list):
        for raw_event_type in observed:
            event_type = _event_type_name(raw_event_type)
            if (
                event_type in _ACTIVITY_RECLASSIFICATION_ORDER
                and _to_bool(events.get(event_type, False))
                and event_type not in activities
            ):
                activities.append(event_type)

    remaining = [
        event_type
        for event_type in _ACTIVITY_RECLASSIFICATION_ORDER
        if _to_bool(events.get(event_type, False)) and event_type not in activities
    ]
    activities.extend(remaining)
    # other 可以与具体事件出现在同一窗口的不重叠时间片中。只有缺少
    # other 的合法 segment 时，才把它视为模型误勾的全窗兜底项。
    if len(activities) > 1 and "other_behavior" in activities:
        has_other_segment = any(
            isinstance(segment, dict)
            and _event_type_name(segment.get("event_type")) == "other_behavior"
            for segment in segments
        ) if isinstance(segments, list) else False
        if not has_other_segment:
            activities.remove("other_behavior")
    return activities


def _reconcile_confirmed_with_description(
    description: str,
    event_type: str,
    event_cn: str,
    model_confirmed: bool,
) -> bool:
    """结构化 events 是主契约；description 只用于展示和兼容兜底。"""
    return model_confirmed


def _append_confirmed_event_labels(
    description: str,
    events: dict[str, bool],
) -> str:
    """第三步失效时的确定性兜底；绝不反向影响事件判定。"""
    labels = [
        EVENT_TYPE_CN[event_type]
        for event_type in EVENT_TYPES
        if events.get(event_type, False)
    ]
    # 模型可能已自行点题。先移除末尾连续的标准事件名，再按 events 的
    # 统一顺序追加一次，避免重复，也避免 description 与 events 不一致。
    known_labels = set(EVENT_TYPE_CN.values())
    parts = [
        part.strip()
        for part in re.split(r"[，,]", description.strip().rstrip("。！？!?；;，, "))
        if part.strip()
    ]
    while parts and parts[-1] in known_labels:
        parts.pop()
    base = "，".join(parts)
    if not labels:
        return f"{base}。" if base else ""
    suffix = "，".join(labels)
    return f"{base}，{suffix}。" if base else f"{suffix}。"


def _description_comparison_text(text: str) -> str:
    """去掉事件标签和标点，用于判断第三步是否仍保留第一步事实。"""
    result = text
    for label in sorted(EVENT_TYPE_CN.values(), key=len, reverse=True):
        result = result.replace(label, "")
    return re.sub(r"[\s，,。！？!?；;：:]", "", result)


def _choose_final_description(
    objective_description: str,
    model_final_description: Any,
    events: dict[str, bool],
    contract_warnings: list[str],
) -> tuple[str, str]:
    """校验第三步最小改写；无效时回退为客观描述加结构化标签。"""
    if not isinstance(model_final_description, str) or not model_final_description.strip():
        contract_warnings.append(
            "final_description 缺失，已使用客观描述与结构化事件生成兜底描述"
        )
        return _append_confirmed_event_labels(objective_description, events), "fallback"

    final_description = _compact_description(model_final_description.strip())
    true_labels = [
        EVENT_TYPE_CN[name]
        for name in EVENT_TYPES
        if events.get(name, False)
    ]
    false_labels = [
        EVENT_TYPE_CN[name]
        for name in EVENT_TYPES
        if not events.get(name, False)
    ]
    missing_labels = [label for label in true_labels if label not in final_description]
    forbidden_labels = [label for label in false_labels if label in final_description]

    objective_text = _description_comparison_text(objective_description)
    final_text = _description_comparison_text(final_description)
    similarity = (
        SequenceMatcher(None, objective_text, final_text).ratio()
        if objective_text and final_text
        else 1.0
    )
    length_limit = min(
        120,
        max(40, len(objective_description) + sum(len(label) for label in true_labels) + 28),
    )

    reasons = []
    if missing_labels:
        reasons.append("缺少true事件中文名:" + ",".join(missing_labels))
    if forbidden_labels:
        reasons.append("包含false事件中文名:" + ",".join(forbidden_labels))
    if len(final_description) > length_limit:
        reasons.append("相对客观描述改写过长")
    if objective_text and final_text and similarity < 0.45:
        reasons.append("与客观描述差异过大")
    if not true_labels and final_description.strip() != objective_description.strip():
        reasons.append("没有true事件却改写了客观描述")

    if reasons:
        contract_warnings.append(
            "final_description 未通过第三步校验（"
            + "；".join(reasons)
            + "），已使用确定性兜底"
        )
        return _append_confirmed_event_labels(objective_description, events), "fallback"

    if final_description[-1:] not in "。！？!?":
        final_description += "。"
    return final_description, "model"


def normalize_vlm_result(
    parsed: dict[str, Any],
    event_type: str,
    frame_count: int | None = None,
) -> dict[str, Any]:
    """
    对 parse_json() 的输出做二次校验，
    保证返回值包含以下稳定字段，且类型固定：

        event_confirmed: bool
        objective_description: str
        final_description: str
        description: str
        is_phone_usage: bool
        is_studying: bool

    这是给B自己、以及未来D/E使用的“稳定契约”，
    不管Qwen-VL输出多不规范，调用方都不需要再做防御性判断。
    """

    candidate_event_type = event_type
    objective_description = parsed.get("objective_description")
    if not isinstance(objective_description, str) or not objective_description.strip():
        # 兼容旧模型输出；旧 description 被视为第一步客观描述。
        objective_description = parsed.get("description", "")
    if not isinstance(objective_description, str) or not objective_description.strip():
        objective_description = "[VLM未返回有效description]"
    else:
        objective_description = _compact_description(objective_description.strip())

    # 新契约中 events 是唯一分类依据。primary_event、observed_activities
    # 和 activity_segments 只描述顺序/时间，不允许把 events=false 反向改成 true。
    events_dict = parsed.get("events", {})
    # 只要模型显式返回了 events，就采用严格新契约；即使它为空或类型错误，
    # 也不能再由 primary/description/event_confirmed 偷偷制造确认事件。
    events_supplied = "events" in parsed
    normalized_event_values = {name: False for name in EVENT_TYPES}
    if isinstance(events_dict, dict):
        for raw_name, value in events_dict.items():
            name = _normalize_event_type_name(raw_name)
            if name in normalized_event_values:
                normalized_event_values[name] = _to_bool(value)
    raw_events = normalized_event_values

    raw_primary_event = _normalize_event_type_name(parsed.get("primary_event"))
    primary_event = _event_type_name(parsed.get("primary_event"))
    raw_observed_activities = parsed.get("observed_activities", [])
    raw_observed_event_names = (
        [
            name
            for item in raw_observed_activities
            if (name := _normalize_event_type_name(item)) is not None
        ]
        if isinstance(raw_observed_activities, list)
        else []
    )
    observed_activities = list(dict.fromkeys(raw_observed_event_names))
    raw_segments = parsed.get("activity_segments", [])
    if not events_supplied:
        # 仅兼容旧调用方；有 events 时绝不使用其他字段升级结果。
        legacy_names = []
        if isinstance(primary_event, str):
            legacy_names.append(primary_event)
        if isinstance(observed_activities, list):
            legacy_names.extend(observed_activities)
        if isinstance(raw_segments, list):
            legacy_names.extend(
                _event_type_name(segment.get("event_type"))
                for segment in raw_segments
                if isinstance(segment, dict)
            )
        for name in legacy_names:
            if name in raw_events:
                raw_events[name] = True
        if not any(raw_events.values()) and _to_bool(parsed.get("event_confirmed", False)):
            raw_events[candidate_event_type] = True

    contract_warnings: list[str] = []
    parse_error = parsed.get("_parse_error")
    if isinstance(parse_error, str) and parse_error:
        contract_warnings.append(parse_error)
    raw_events_snapshot = dict(raw_events)

    normalized_segments = []
    if isinstance(raw_segments, list):
        for segment in raw_segments:
            if not isinstance(segment, dict):
                continue
            name = _event_type_name(segment.get("event_type"))
            try:
                start_frame = int(segment.get("start_frame", 0))
                end_frame = int(segment.get("end_frame", 0))
            except (TypeError, ValueError):
                continue
            within_frame_range = (
                frame_count is None
                or frame_count <= 0
                or end_frame <= frame_count
            )
            if (
                isinstance(name, str)
                and raw_events.get(name, False)
                and start_frame >= 1
                and within_frame_range
                and end_frame >= start_frame
            ):
                normalized_segments.append(
                    {
                        "event_type": name,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                )
            elif isinstance(name, str) and name in raw_events:
                contract_warnings.append(
                    f"忽略与 events 不一致或帧区间无效的 segment: {segment!r}"
                )
    normalized_segments.sort(key=lambda item: item["start_frame"])

    represented_types = {item["event_type"] for item in normalized_segments}

    # events 是模型第二步的原始判断，但正式事件还必须满足模型自己承诺的
    # 时序契约。有合法 segment 的事件可信；上游候选已有 EventEngine
    # 的时序证据；primary_event 可用于改判，但状态转换事件改判时仍必须
    # 给出 segment，防止一帧静态画面同时制造“开始”和“结束”。
    supported_types = set(represented_types)
    candidate_confirmed = raw_events.get(candidate_event_type, False)
    primary_has_segment = primary_event in represented_types
    if candidate_confirmed:
        # 通用 other 候选若已被 VLM 明确改判成具体主事件，不再与主事件
        # 并列保留；但 other 自己有合法时间片时，它代表真实过渡动作，
        # 不再只是上游占位符。
        if not (
            candidate_event_type == "other_behavior"
            and isinstance(primary_event, str)
            and primary_event != candidate_event_type
            and raw_events.get(primary_event, False)
            and candidate_event_type not in represented_types
        ):
            supported_types.add(candidate_event_type)
    if isinstance(primary_event, str) and raw_events.get(primary_event, False):
        if (
            primary_event not in _TRANSITION_EVENT_TYPES
            or primary_event == candidate_event_type
            or primary_has_segment
        ):
            # 没有任何 segment 时，一个明确的原候选优先于不同的主事件，
            # 防止“入座候选”同时整窗生成“其他/准备”。有真实 segment
            # 时则允许两个事件按各自区间并存。
            if (
                primary_has_segment
                or not candidate_confirmed
                or primary_event == candidate_event_type
                or candidate_event_type == "other_behavior"
            ):
                supported_types.add(primary_event)

    raw_true_types = [name for name, confirmed in raw_events.items() if confirmed]
    # 兼容最小但明确的输出：如果只有一个非状态转换事件为 true，即使
    # 2B 漏写 primary/segment，也允许它改判，不再因为小格式错误清零。
    if len(raw_true_types) == 1 and raw_true_types[0] not in _TRANSITION_EVENT_TYPES:
        supported_types.add(raw_true_types[0])
    # 歧义窗口会被 Runtime 拆成约 4 秒、最多 5 张图的短序列重新判断。
    # 这时整段本身已经提供了连续时序证据，允许唯一的位置变化主事件
    # 在漏写 segment 时成立；完整 8 秒窗口仍维持严格分段要求。
    if (
        frame_count is not None
        and 1 < frame_count <= 5
        and len(raw_true_types) == 1
        and raw_true_types[0] in _TRANSITION_EVENT_TYPES
        and primary_event == raw_true_types[0]
    ):
        supported_types.add(raw_true_types[0])

    normalized_events = {
        name: bool(confirmed and name in supported_types)
        for name, confirmed in raw_events.items()
    }
    dropped_true_types = [
        name
        for name in raw_true_types
        if not normalized_events.get(name, False)
    ]
    if dropped_true_types:
        contract_warnings.append(
            "已忽略缺少时序证据且不是候选/主事件的 events=true: "
            + ", ".join(dropped_true_types)
        )

    normalized_segments = [
        segment
        for segment in normalized_segments
        if normalized_events.get(segment["event_type"], False)
    ]

    def has_ordered_transition(first: str, second: str) -> bool:
        first_segments = [
            segment for segment in normalized_segments
            if segment["event_type"] == first
        ]
        second_segments = [
            segment for segment in normalized_segments
            if segment["event_type"] == second
        ]
        return any(
            first_segment["end_frame"] < second_segment["start_frame"]
            for first_segment in first_segments
            for second_segment in second_segments
        )

    for first, second in (
        ("sit_at_study_position", "leave_study_position"),
    ):
        if not (normalized_events.get(first) and normalized_events.get(second)):
            continue
        if has_ordered_transition(first, second):
            continue

        keep = candidate_event_type if candidate_event_type in (first, second) else None
        if keep is None and primary_event in (first, second):
            keep = primary_event
        for name in (first, second):
            if name != keep:
                normalized_events[name] = False
        contract_warnings.append(
            f"{first} 与 {second} 缺少先后分离的时间片，"
            + (f"仅保留 {keep}" if keep else "已全部忽略")
        )

    # 三步结果一致性修复：2B 模型偶尔能在第一步准确写出“使用笔记本
    # 电脑”，却在第二步漏勾布尔值。仅当第二步没有任何具体行为（或只勾
    # 了 other）时，才用第一步中的明确正向动作补回一个具体行为。
    concrete_behaviors = {
        "reading",
        "writing",
        "phone_usage",
        "computer_usage",
        "communication_distraction",
    }
    description_activity = infer_activity_from_description(objective_description)
    confirmed_concrete = {
        name for name in concrete_behaviors if normalized_events.get(name, False)
    }
    if description_activity in concrete_behaviors and not confirmed_concrete:
        normalized_events[description_activity] = True
        normalized_events["other_behavior"] = False
        primary_event = description_activity
        if description_activity not in observed_activities:
            observed_activities.append(description_activity)
        contract_warnings.append(
            "events 与 objective_description 不一致，已按明确动作修复为 "
            + description_activity
        )

    # 明确的准备/收拾动作可以推翻无分段的具体事件。这里要求描述中存在
    # 动作短语，而不是仅凭“书/电脑”等名词，因此不会把正常学习中的
    # 偶发物体误改成 other。混合窗口已有合法分段时则保留各阶段。
    has_other_segment = any(
        segment["event_type"] == "other_behavior"
        for segment in normalized_segments
    )
    has_concrete_segment = any(
        segment["event_type"] in concrete_behaviors
        for segment in normalized_segments
    )
    if (
        description_activity == "other_behavior"
        and not (has_other_segment and has_concrete_segment)
    ):
        replaced = [
            name for name in concrete_behaviors
            if normalized_events.get(name, False)
        ]
        for name in replaced:
            normalized_events[name] = False
        normalized_events["other_behavior"] = True
        primary_event = "other_behavior"
        if "other_behavior" not in observed_activities:
            observed_activities.append("other_behavior")
        if replaced:
            contract_warnings.append(
                "客观描述包含明确整理/准备动作，已将无分段具体事件改为 other_behavior: "
                + ", ".join(sorted(replaced))
            )
        elif not confirmed_concrete:
            contract_warnings.append(
                "events 与 objective_description 不一致，已按明确过渡动作修复为 other_behavior"
            )

    # other 与具体行为只有在合法、不重叠的时间片中才能并存。没有分段时，
    # 优先遵循明确的过渡描述/primary；否则保留具体事件并取消误勾的 other。
    if any(normalized_events.get(name, False) for name in concrete_behaviors):
        if normalized_events.get("other_behavior", False):
            other_segments = [
                segment for segment in normalized_segments
                if segment["event_type"] == "other_behavior"
            ]
            concrete_segments = [
                segment for segment in normalized_segments
                if segment["event_type"] in concrete_behaviors
            ]
            segments_are_separate = bool(other_segments and concrete_segments) and all(
                other["end_frame"] < concrete["start_frame"]
                or concrete["end_frame"] < other["start_frame"]
                for other in other_segments
                for concrete in concrete_segments
            )
            if not segments_are_separate:
                if primary_event == "other_behavior" or description_activity == "other_behavior":
                    for name in concrete_behaviors:
                        normalized_events[name] = False
                    contract_warnings.append(
                        "other_behavior 与具体行为缺少不重叠分段，已保留明确过渡行为"
                    )
                else:
                    normalized_events["other_behavior"] = False
                    contract_warnings.append(
                        "other_behavior 与具体行为缺少不重叠分段，已保留具体行为"
                    )

    normalized_segments = [
        segment
        for segment in normalized_segments
        if normalized_events.get(segment["event_type"], False)
    ]
    normalized_observed = [
        name
        for name in observed_activities
        if isinstance(name, str) and normalized_events.get(name, False)
    ] if isinstance(observed_activities, list) else []
    description, final_description_source = _choose_final_description(
        objective_description,
        parsed.get("final_description"),
        normalized_events,
        contract_warnings,
    )

    final_event_confirmed = any(confirmed for confirmed in normalized_events.values())
    if _to_bool(parsed.get("event_confirmed", False)) != final_event_confirmed:
        contract_warnings.append("event_confirmed 与 events 不一致，已以 events 为准")

    # primary_event 兜底：如果 VLM 写了 "none" 但存在 true 事件，取第一个 true 事件
    final_primary = primary_event
    if (
        not isinstance(final_primary, str)
        or final_primary == "none"
        or not normalized_events.get(final_primary, False)
    ):
        for name, confirmed in normalized_events.items():
            if confirmed:
                final_primary = name
                break
        else:
            final_primary = "none"

    normalized_phone_usage = _to_bool(parsed.get("is_phone_usage", False))
    phone_event_confirmed = any(
        normalized_events.get(name, False) for name in _PHONE_EVENT_TYPES
    )
    if phone_event_confirmed and not normalized_phone_usage:
        normalized_phone_usage = True
        contract_warnings.append(
            "手机事件为 true 但 is_phone_usage=false，已按结构化事件修正为 true"
        )

    normalized_studying = _to_bool(parsed.get("is_studying", False))
    study_event_confirmed = any(
        normalized_events.get(name, False) for name in _STUDY_EVENT_TYPES
    )
    if study_event_confirmed and not normalized_studying:
        normalized_studying = True
        contract_warnings.append(
            "学习事件为 true 但 is_studying=false，已按结构化事件修正为 true"
        )

    return {
        "event_confirmed": final_event_confirmed,
        "objective_description": objective_description,
        "final_description": description,
        "final_description_source": final_description_source,
        "description": description,
        "is_phone_usage": normalized_phone_usage,
        "is_studying": normalized_studying,
        "events": normalized_events,
        "raw_events": raw_events_snapshot,
        "primary_event": final_primary,
        "observed_activities": normalized_observed,
        "activity_segments": normalized_segments,
        "contract_warnings": contract_warnings,
        "parse_error": parse_error if isinstance(parse_error, str) else "",
    }


# ============================================================
# event_confirmed=false 的下游处理契约（新增，与A讨论后确定）
#
# 讨论结论：
#     event_confirmed=false → 不进入正式Video Memory，
#     只保留debug/log信息用于后续分析。
#
#     职责划分：
#         VLM（B）负责"事件确认"；
#         D（Memory）只负责存储"已经确认"的事件；
#         E（UI）只负责展示"已经确认"的事件。
#     D不应该再对一个已经被VLM判false的事件做二次取舍，
#     E也不应该展示已经被VLM否定的事件。
#
#     所以B在这里把"要不要进正式Memory"这个判断结果显式
#     暴露出来（should_persist_to_memory），而不是让D/E各自
#     再解释一遍event_confirmed的含义；同时把被拒绝的事件
#     写入本地debug日志（而不是直接丢弃），方便后续复盘
#     "这组帧到底为什么没通过"。
# ============================================================

def should_persist_to_memory(meta: dict[str, Any]) -> bool:
    """
    【给D / EventEngine使用】

    判断这次VLM分析结果是否应该进入正式Video Memory。

    规则（与A讨论后确定）：
        event_confirmed=True  → True（正常进入 Memory → Embedding → UI）
        event_confirmed=False → False（不进入正式Memory，仅保留debug/log）

    注意：
        这里只是把判断规则封装成一个函数，方便D调用，
        不代表B会替D去调用Memory的写入接口——
        B不知道、也不应该知道D的存储实现细节。
    """

    return bool(meta.get("event_confirmed", False))


def _write_debug_log(
    event: Event,
    meta: dict[str, Any],
    log_path: str = "debug_logs/vlm_rejected_events.jsonl",
) -> None:
    """
    【仅供本地调试/复盘使用】

    当event_confirmed=false时，把这次分析结果追加写入本地
    debug日志（JSON Lines格式，一行一条），而不是直接丢弃。

    背景：
        false本身就是VLM"判断这个事件不成立"，按照与A讨论的
        结论不应该进入正式Memory；但如果什么记录都不留，
        后续想排查"哪些窗口总是被拒绝、是不是Prompt某个门槛
        设置得不合理"就无从查起。所以单独落一份debug日志，
        和正式Memory完全隔离。

    注意：
        这里只做本地文件写入，不涉及任何D的存储接口；
        写入失败（例如目录权限问题）不应该中断主流程，
        所以异常只打印警告，不向上抛出。
    """

    try:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "event_type": event.event_type,
            "start_time": getattr(event, "start_time", None),
            "end_time": getattr(event, "end_time", None),
            "track_id": getattr(event, "track_id", None),
            "description": meta.get("description", ""),
            "is_phone_usage": meta.get("is_phone_usage", False),
            "is_studying": meta.get("is_studying", False),
            "logged_at": time.time(),
        }

        with log_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    except OSError as exc:
        print(
            "[VLM Warning] "
            f"debug日志写入失败（不影响主流程）：{exc}"
        )


# ============================================================
# 正式接口
# ============================================================

def analyze_event(
    model,
    processor,
    event: Event,
    frame_paths: list[str],
    return_meta: bool = False,
):
    """
    【正式接口】

    给A / EventEngine使用。

    输入：
        event：
            A已经确定event_type的Event对象。

        frame_paths：
            当前事件时间窗口中的多帧图片，
            必须按照时间顺序排列。
            正式流程中来自C的VideoBuffer.get_frames()，
            推荐3~12张（非强制，Buffer在视频开头等边界情况
            可能不足，这里只警告不报错）。

        return_meta：
            默认False，兼容A现有调用方式，只返回Event。
            设为True时，额外返回本次VLM分析的完整结构化结果
            （event_confirmed / is_phone_usage / is_studying），
            供D（Retrieval/Memory）或E（UI）在需要展示
            "VLM分析结果"时使用，而不必去解析print日志。
            D拿到meta后应配合 should_persist_to_memory(meta)
            判断这次结果要不要写入正式Video Memory
            （event_confirmed=false的情况不应该进入正式Memory，
            B已经把这类情况记录到本地debug日志，详见
            should_persist_to_memory() / _write_debug_log()）。

    输出：
        return_meta=False（默认）：
            返回补充description后的同一个Event对象。

        return_meta=True：
            返回 (event, meta)，
            meta为 normalize_vlm_result() 规范化后的dict。

    注意：

        event.event_type 是 A 给出的弱候选，不是最终结论。

        B / Qwen-VL 对完整事件清单输出结构化 events；它可以否定候选、
        改成其他类型或确认多个连续事件。为保持 Event 数据类接口兼容，
        本函数把最终类型放在 meta 中，由 Runtime 重建最终 Event。
    """

    # ========================================================
    # 参数检查
    # ========================================================

    if not isinstance(event, Event):
        raise TypeError(
            "event must be an Event object"
        )

    if not event.event_type:
        raise ValueError(
            "event.event_type cannot be empty"
        )

    candidate_event_type = event.event_type
    if candidate_event_type not in EVENT_TYPES:
        raise ValueError(
            f"Unknown event_type: {event.event_type!r}. "
            f"Allowed events: {EVENT_TYPES}"
        )

    if not frame_paths:
        raise ValueError(
            "frame_paths cannot be empty"
        )

    if not (3 <= len(frame_paths) <= 12):
        print(
            "[VLM Warning] "
            f"frame_paths数量为{len(frame_paths)}张，"
            "推荐范围是3~12张关键帧（对接C的VideoBuffer规范），"
            "数量异常不会中断流程，但建议检查上游抽帧逻辑。"
        )

    # ========================================================
    # 构造Prompt
    # ========================================================

    prompt = build_activity_prompt(
        candidate_event_type,
        frame_count=len(frame_paths),
        candidate_scores=event.candidate_scores,
        unclassified_ratio=event.unclassified_ratio,
    )

    # ========================================================
    # 调用Qwen-VL
    # ========================================================

    raw_output = run_vlm(
        model=model,
        processor=processor,
        image_paths=frame_paths,
        prompt=prompt,
    )

    # ========================================================
    # 解析JSON
    # ========================================================

    parsed = parse_json(
        raw_output
    )
    if parsed.get("_parse_error"):
        print(
            f"[VLM解析] 失败 | 原始输出长度={len(raw_output)} | "
            f"已抢救描述={parsed.get('description', '')}"
        )
    else:
        print(f"[VLM解析] 成功 | 原始输出长度={len(raw_output)}")

    # ========================================================
    # 字段规范化（新增）
    #
    # parse_json只保证“是不是合法JSON”，
    # normalize_vlm_result进一步保证“字段类型是不是对的”。
    # 之后无论parsed里是什么脏数据，meta都是可信的。
    # ========================================================

    meta = normalize_vlm_result(
        parsed,
        event_type=candidate_event_type,
        frame_count=len(frame_paths),
    )
    raw_true = [
        name for name, value in meta.get("raw_events", {}).items() if value
    ]
    final_true = [
        name for name, value in meta.get("events", {}).items() if value
    ]
    print(
        f"[VLM判断] 原始true={raw_true or ['无']} | "
        f"校验后true={final_true or ['无']} | "
        f"primary={meta.get('primary_event', 'none')} | "
        f"segments={meta.get('activity_segments', [])}"
    )
    print(
        "[VLM三步] "
        f"①客观描述={meta.get('objective_description', '')} | "
        f"②事件={final_true or ['无']} | "
        f"③最终描述={meta.get('final_description', '')} | "
        f"来源={meta.get('final_description_source', 'unknown')}"
    )
    for warning in meta.get("contract_warnings", []):
        print(f"[VLM Contract Warning] {warning}")

    # 把 VLM 原始输出注入 meta，方便全链路 debug
    meta["_vlm_raw"] = raw_output

    # ========================================================
    # VLM 可推翻任意上游候选。Runtime 读取 confirmed_event_type、events
    # 和 activity_segments，重建最终要写入 Memory 的 Event。
    # ========================================================
    primary = meta.get("primary_event")
    if isinstance(primary, str) and meta["events"].get(primary, False):
        meta["confirmed_event_type"] = primary
    else:
        meta["confirmed_event_type"] = next(
            (name for name, confirmed in meta["events"].items() if confirmed),
            event.event_type,
        )

    # ========================================================
    # 这里仅原地补充 description；event_type 的最终回写由 Runtime 完成。
    # ========================================================

    event.description = meta["description"]

    # ========================================================
    # event_confirmed 日志（改用规范化后的meta，避免脏值）
    # ========================================================

    if not meta["event_confirmed"]:
        # ====================================================
        # event_confirmed=false 的下游处理（与A讨论后确定）：
        #
        #     不进入正式Video Memory，仅保留debug/log信息。
        #
        # B自己不调用D的Memory写入接口，只负责：
        #   1) 把"这次不该进Memory"这个判断结果通过
        #      should_persist_to_memory()暴露给D；
        #   2) 把被拒绝的这次分析结果记一份本地debug日志，
        #      避免连排查依据都没有。
        # ====================================================

        _write_debug_log(event, meta)

    # ========================================================
    # 职责边界（与A讨论后确定）：
    #
    #     event_confirmed=true  → Event 正常返回，D 用
    #       should_persist_to_memory(meta) 判断是否写 Memory
    #
    #     event_confirmed=false → Event 不返回（D 拿到 None），
    #       B 已写 debug 日志，D/E 不需要再解释 event_confirmed
    #       的含义，职责边界彻底清晰。
    #
    # event_confirmed / is_phone_usage / is_studying
    # 不塞入 Event 结构（B 不擅自扩展 A 定义的结构），
    # D/E 需要这些字段时通过 return_meta 获取。
    # ========================================================

    if not meta["event_confirmed"]:
        if return_meta:
            return None, meta
        return None

    if return_meta:
        return event, meta

    return event


# ============================================================
# 独立视觉理解测试
# ============================================================

def classify_frames_standalone(
    model,
    processor,
    frame_paths: list[str],
) -> dict[str, Any]:
    """
    【仅供本地测试】

    不提供event_type。

    用于测试Qwen-VL是否能够：

        1. 理解连续视频帧
        2. 描述人物动作
        3. 判断是否使用手机
        4. 判断是否处于学习状态

    注意：

        这个函数不是正式Pipeline接口。

        正式Pipeline必须使用：

            analyze_event()
    """

    if not frame_paths:
        raise ValueError(
            "frame_paths cannot be empty"
        )

    raw_output = run_vlm(
        model=model,
        processor=processor,
        image_paths=frame_paths,
        prompt=build_activity_prompt(
            "other_behavior",
            frame_count=len(frame_paths),
        ),
    )

    return parse_json(
        raw_output
    )


# ============================================================
# 跨事件类型批量测试（新增）
#
# 对应组长要求：
#     "3. 检查不同事件类型下的识别效果"
#
    # 用同一组真实关键帧，一次判断当前完整事件清单，
# 一次性看出哪些事件类型的Prompt效果差、容易误判，
# 避免明天只能一个一个手动跑。
#
# 注意：这不是正式Pipeline接口，仅用于B自测。
# ============================================================

# ============================================================
# 跨事件类型批量测试（自测用）
# ============================================================

def test_all_event_types(
    model,
    processor,
    frame_paths: list[str],
) -> dict[str, dict[str, Any]]:
    """
    用同一组frame_paths，调用一次VLM，
    让它一次性判断当前全部event_type是否发生。

    返回：
        {event_type: meta, ...}
        每个event_type的confirmed直接来自VLM对"events"字段的判断。

    一次调用，由VLM看完所有帧后输出八个事件的判断，
    每个event_type的confirmed直接来自VLM对"events"字段的输出。
    """

    if not frame_paths:
        raise ValueError(
            "frame_paths cannot be empty"
        )

    t0 = time.time()

    prompt = build_activity_prompt(
        "other_behavior",
        frame_count=len(frame_paths),
    )

    raw_output = run_vlm(
        model=model,
        processor=processor,
        image_paths=frame_paths,
        prompt=prompt,
    )

    latency = time.time() - t0

    parsed = parse_json_multi_event(raw_output)

    description = parsed.get("description", "")
    if not isinstance(description, str) or not description.strip():
        description = "[VLM未返回有效description]"

    events_map: dict[str, bool] = parsed.get("events", {})

    print(f"\n[Qwen-VL批量分析] latency={latency:.2f}s")
    print(f"  description: {description}")
    print(f"  events: {events_map}")

    results: dict[str, dict[str, Any]] = {}

    for event_type in EVENT_TYPES:

        event_confirmed = bool(events_map.get(event_type, False))

        results[event_type] = {
            "event_confirmed": event_confirmed,
            "description": description,
            "is_phone_usage": _to_bool(
                parsed.get("is_phone_usage", False)
            ),
            "is_studying": _to_bool(
                parsed.get("is_studying", False)
            ),
        }

    return results


# ============================================================
# 本地测试入口
# ============================================================

def main():
    """
    本地测试Qwen-VL。

    两种模式：

    1. 正式流程模拟：

        python -m src.vlm.inference \
            --image frame1.jpg frame2.jpg \
            --event_type reading

    2. 独立视觉测试：

        python -m src.vlm.inference \
            --image frame1.jpg frame2.jpg
    """

    parser = argparse.ArgumentParser(
        description="Qwen-VL学习场景事件分析"
    )

    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        required=True,
        help="多帧图片路径，必须按照时间顺序",
    )

    parser.add_argument(
        "--event_type",
        type=str,
        choices=EVENT_TYPES,
        default=None,
        help=(
            "模拟A已经确定的事件类型。"
            "不传则进入独立视觉理解测试模式。"
        ),
    )

    parser.add_argument(
        "--test_all_events",
        action="store_true",
        help=(
            "用同一组--image测试当前全部event_type，"
            "用于检查不同事件类型下的识别效果（忽略--event_type）。"
        ),
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="可选的本地 Qwen2-VL 模型目录；也可使用 QWEN_VL_MODEL_PATH",
    )

    args = parser.parse_args()

    # ========================================================
    # 加载模型
    # ========================================================

    model, processor = load_model(model_path=args.model_path)

    print(
        f"[输入帧数] {len(args.image)}"
    )

    print(
        f"[输入图片] {args.image}"
    )

    print(
        "[Qwen-VL 正在分析...]"
    )

    t0 = time.time()

    # ========================================================
    # 跨事件类型批量测试
    # ========================================================

    if args.test_all_events:

        results = test_all_event_types(
            model=model,
            processor=processor,
            frame_paths=args.image,
        )

        latency = time.time() - t0

        print(
            f"\n[全部事件类型测试完成] "
            f"总耗时 {latency:.2f} 秒"
        )

        print(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    # ========================================================
    # 模拟正式Pipeline
    # ========================================================

    if args.event_type:

        incoming_event = Event(
            event_type=args.event_type,
            start_time=0.0,
            end_time=0.0,
            track_id=1,
            confidence=1.0,
            description="",
        )

        result_event, meta = analyze_event(
            model=model,
            processor=processor,
            event=incoming_event,
            frame_paths=args.image,
            return_meta=True,
        )

        latency = time.time() - t0

        print(
            f"\n[VLM Latency] "
            f"{latency:.2f} 秒"
        )

        print(
            "\n[补全后的Event]"
        )

        if result_event is None:
            print(
                "[Event Rejected] "
                "VLM判断画面证据不支持事件类型："
                f"{incoming_event.event_type}\n"
                f"  description: {meta['description']}\n"
                f"  is_phone_usage: {meta['is_phone_usage']}\n"
                f"  is_studying: {meta['is_studying']}\n"
                "  （已记录到debug日志，不进入Memory）"
            )
        else:
            print(result_event)
            print(
                f"  is_phone_usage: {meta['is_phone_usage']}\n"
                f"  is_studying: {meta['is_studying']}"
            )

    # ========================================================
    # 独立视觉测试
    # ========================================================

    else:

        result = classify_frames_standalone(
            model=model,
            processor=processor,
            frame_paths=args.image,
        )

        latency = time.time() - t0

        print(
            f"\n[VLM Latency] "
            f"{latency:.2f} 秒"
        )

        print(
            "\n[VLM独立视觉理解结果]"
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )


# ============================================================
# Python入口
# ============================================================

if __name__ == "__main__":
    main()
