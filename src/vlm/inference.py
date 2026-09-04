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
import json
import re
import time
from pathlib import Path
from typing import Any

from qwen_vl_utils import process_vision_info

from .qwen_vlm import load_model
from .prompt import (
    build_activity_prompt,
    CLASSIFY_PROMPT,
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
        "max_new_tokens": 256,
        "do_sample": do_sample,
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

def parse_json(raw_text: str) -> dict[str, Any]:
    """
    解析Qwen-VL输出的JSON。

    Qwen-VL正常情况下应该直接返回JSON，
    这里增加容错，避免模型偶尔输出额外文字导致解析失败。

    返回字段至少保证：
        event_confirmed
        description
        is_phone_usage
        is_studying
    """

    raw_text = raw_text.strip()

    # --------------------------------------------------------
    # 情况1：标准JSON
    # --------------------------------------------------------

    try:
        result = json.loads(raw_text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # 情况2：JSON前后存在额外文字
    # --------------------------------------------------------

    match = re.search(
        r"\{.*\}",
        raw_text,
        re.DOTALL,
    )

    if match:

        try:
            result = json.loads(
                match.group()
            )

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    # --------------------------------------------------------
    # 情况3：解析失败
    # --------------------------------------------------------

    return {
        "event_confirmed": False,
        "description": (
            f"[VLM JSON解析失败] {raw_text}"
        ),
        "is_phone_usage": False,
        "is_studying": False,
    }


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
            ... 14个事件
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
    "study_preparation": ["整理", "摆放", "准备", "拿取", "拿出", "摊开", "布置"],
    "start_study": ["开始", "翻开", "打开", "提笔", "动笔", "拿起"],
    "end_study": ["合上", "收起", "收拾", "停止", "放下"],
    "reading": [
        "看书", "读书", "看资料", "翻页", "阅读", "看教材", "看书本",
        "拿书", "拿着书", "手里拿着书",
    ],
    "writing": ["写字", "记笔记", "做笔记", "做题", "画图", "书写", "写笔记"],
    "phone_learning": ["手机学习", "手机看", "手机查", "手机背"],
    "computer_learning": ["电脑学习", "电脑看", "电脑查"],
    "other_study_behavior": ["学习"],
    "phone_distraction": [
        "刷手机", "玩手机", "手机聊天", "手机游戏", "手机刷",
        "拿起手机", "使用手机", "查看手机", "看手机", "手机内容",
    ],
    "computer_distraction": ["打游戏", "玩电脑", "电脑游戏", "电脑刷"],
    "communication_distraction": ["说话", "通话", "讨论"],
    "study_end_cleanup": ["收拾", "整理", "收", "归位"],
}

_ACTIVITY_RECLASSIFICATION_ORDER = (
    "writing",
    "reading",
    "phone_learning",
    "computer_learning",
    "phone_distraction",
    "computer_distraction",
    "communication_distraction",
    "other_study_behavior",
)


def infer_activity_from_description(description: str) -> str | None:
    """从 VLM 描述中的明确动作词纠正 YOLO 产生的活动候选类型。"""
    if not isinstance(description, str):
        return None
    for event_type in _ACTIVITY_RECLASSIFICATION_ORDER:
        if _description_supports_event(description, event_type):
            return event_type
    return None


def _description_supports_event(description: str, event_type: str) -> bool:
    if event_type == "phone_learning":
        return any(word in description for word in ("手机", "移动设备")) and any(
            word in description
            for word in ("学习资料", "课程", "题目", "背单词", "教材")
        )
    if event_type == "computer_learning":
        return any(word in description for word in ("电脑", "笔记本电脑")) and any(
            word in description
            for word in ("学习资料", "课程", "题目", "编程", "教材")
        )
    return any(
        keyword in description
        for keyword in EVENT_CONTENT_KEYWORDS.get(event_type, [])
    )


def infer_activity_from_vlm_meta(meta: dict[str, Any]) -> str | None:
    """返回 VLM 确认的第一个活动，兼容原有单事件调用方。"""
    activities = infer_activities_from_vlm_meta(meta)
    return activities[0] if activities else None


def infer_activities_from_vlm_meta(meta: dict[str, Any]) -> list[str]:
    """返回按画面发生顺序排列的全部可信活动。"""
    description = meta.get("description", "")
    if not isinstance(description, str):
        description = ""
    events = meta.get("events", {})
    if not isinstance(events, dict):
        events = {}

    activities: list[str] = []
    segments = meta.get("activity_segments", [])
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            event_type = segment.get("event_type")
            if (
                event_type in _ACTIVITY_RECLASSIFICATION_ORDER
                and _to_bool(events.get(event_type, False))
                and _description_supports_event(description, event_type)
                and event_type not in activities
            ):
                activities.append(event_type)

    observed = meta.get("observed_activities", [])
    if isinstance(observed, list):
        for event_type in observed:
            if (
                event_type in _ACTIVITY_RECLASSIFICATION_ORDER
                and _to_bool(events.get(event_type, False))
                and _description_supports_event(description, event_type)
                and event_type not in activities
            ):
                activities.append(event_type)

    remaining = [
        event_type
        for event_type in _ACTIVITY_RECLASSIFICATION_ORDER
        if (
            (
                _to_bool(events.get(event_type, False))
                or _description_supports_event(description, event_type)
            )
            and _description_supports_event(description, event_type)
            and event_type not in activities
        )
    ]
    remaining.sort(
        key=lambda event_type: min(
            (
                description.find(keyword)
                for keyword in EVENT_CONTENT_KEYWORDS[event_type]
                if keyword in description
            ),
            default=len(description) + _ACTIVITY_RECLASSIFICATION_ORDER.index(event_type),
        )
    )
    activities.extend(remaining)
    if len(activities) > 1 and "other_study_behavior" in activities:
        activities.remove("other_study_behavior")
    return activities


def _reconcile_confirmed_with_description(
    description: str,
    event_type: str,
    event_cn: str,
    model_confirmed: bool,
) -> bool:
    """
    代码层兜底：用 description 内容校验 confirmed 是否合理。

    2026-09-03 改写：
        旧逻辑（suffix 匹配）只检查句尾，无法捕获 VLM 的幻觉
        （如"坐着用手机"被判为 leave_study_position=true）。
        新逻辑：检查 description 全文是否包含该事件的特征关键词。

    规则：
        - confirmed=true，关键词全不在 description → 强制 false
        - confirmed=false → 保留 false，不用关键词反向制造确认结果
        - 没有定义关键词的事件类型 → 保留模型原始判断
    """
    keywords = EVENT_CONTENT_KEYWORDS.get(event_type, [])
    if not keywords:
        return model_confirmed

    # confirmed=true 但没有任何关键词 → VLM 在乱说
    if model_confirmed and not _description_supports_event(description, event_type):
        return False

    return model_confirmed


def normalize_vlm_result(
    parsed: dict[str, Any],
    event_type: str,
) -> dict[str, Any]:
    """
    对 parse_json() 的输出做二次校验，
    保证返回值包含以下稳定字段，且类型固定：

        event_confirmed: bool
        description: str
        is_phone_usage: bool
        is_studying: bool

    这是给B自己、以及未来D/E使用的“稳定契约”，
    不管Qwen-VL输出多不规范，调用方都不需要再做防御性判断。
    """

    description = parsed.get("description", "")

    if not isinstance(description, str) or not description.strip():
        description = "[VLM未返回有效description]"

    event_cn = EVENT_TYPE_CN.get(event_type, event_type)

    # prompt 同时输出了 events{} 和 event_confirmed，
    # 优先取 events[event_type]，event_confirmed 兜底读取。
    events_dict = parsed.get("events", {})
    raw_events = {
        name: _to_bool(events_dict.get(name, False))
        for name in EVENT_TYPES
    } if isinstance(events_dict, dict) else {name: False for name in EVENT_TYPES}

    primary_event = parsed.get("primary_event")
    if isinstance(primary_event, str) and primary_event in raw_events:
        raw_events[primary_event] = True
    observed_activities = parsed.get("observed_activities", [])
    if isinstance(observed_activities, list):
        for name in observed_activities:
            if isinstance(name, str) and name in raw_events:
                raw_events[name] = True

    if raw_events.get(event_type, False):
        model_confirmed = True
    else:
        model_confirmed = _to_bool(parsed.get("event_confirmed", False))

    reconciled_confirmed = _reconcile_confirmed_with_description(
        description=description,
        event_type=event_type,
        event_cn=event_cn,
        model_confirmed=model_confirmed,
    )

    normalized_events = {}
    for name in EVENT_TYPES:
        normalized_events[name] = _reconcile_confirmed_with_description(
            description=description,
            event_type=name,
            event_cn=EVENT_TYPE_CN.get(name, name),
            model_confirmed=raw_events[name],
        )

    normalized_observed = [
        name
        for name in observed_activities
        if isinstance(name, str) and normalized_events.get(name, False)
    ] if isinstance(observed_activities, list) else []

    normalized_segments = []
    raw_segments = parsed.get("activity_segments", [])
    if isinstance(raw_segments, list):
        for segment in raw_segments:
            if not isinstance(segment, dict):
                continue
            name = segment.get("event_type")
            try:
                start_frame = int(segment.get("start_frame", 0))
                end_frame = int(segment.get("end_frame", 0))
            except (TypeError, ValueError):
                continue
            if (
                isinstance(name, str)
                and normalized_events.get(name, False)
                and start_frame >= 1
                and end_frame >= start_frame
            ):
                normalized_segments.append(
                    {
                        "event_type": name,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                )
    normalized_segments.sort(key=lambda item: item["start_frame"])

    return {
        "event_confirmed": reconciled_confirmed,
        "description": description,
        "is_phone_usage": _to_bool(parsed.get("is_phone_usage", False)),
        "is_studying": _to_bool(parsed.get("is_studying", False)),
        "events": normalized_events,
        "primary_event": (
            primary_event
            if isinstance(primary_event, str) and normalized_events.get(primary_event, False)
            else "none"
        ),
        "observed_activities": normalized_observed,
        "activity_segments": normalized_segments,
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

        event.event_type由A决定。

        B / Qwen-VL：
            不修改event_type。

        B只负责：
            观察视频帧
            ↓
            分析真实动作
            ↓
            生成description
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

    if event.event_type not in EVENT_TYPES:
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

    prompt = build_activity_prompt(event.event_type)

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

    # ========================================================
    # 字段规范化（新增）
    #
    # parse_json只保证“是不是合法JSON”，
    # normalize_vlm_result进一步保证“字段类型是不是对的”。
    # 之后无论parsed里是什么脏数据，meta都是可信的。
    # ========================================================

    meta = normalize_vlm_result(parsed, event_type=event.event_type)

    # ========================================================
    # 只补充description
    # ========================================================

    event.description = meta["description"]

    # ========================================================
    # event_confirmed 日志（改用规范化后的meta，避免脏值）
    # ========================================================

    if meta["event_confirmed"]:

        print(
            "[VLM Confirmed] "
            f"画面支持事件："
            f"{event.event_type}"
        )

    else:

        print(
            "[VLM Warning] "
            f"画面证据可能不支持事件："
            f"{event.event_type}"
        )

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
        prompt=CLASSIFY_PROMPT,
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
# 用同一组真实关键帧，依次套用14种event_type对应的Prompt，
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
    让它一次性判断全部14种event_type是否发生。

    返回：
        {event_type: meta, ...}
        每个event_type的confirmed直接来自VLM对"events"字段的判断。

    一次调用，由VLM看完所有帧后输出14个事件的判断，
    每个event_type的confirmed直接来自VLM对"events"字段的输出。
    """

    if not frame_paths:
        raise ValueError(
            "frame_paths cannot be empty"
        )

    t0 = time.time()

    from .prompt import build_all_events_prompt

    prompt = build_all_events_prompt()

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
            "用同一组--image，依次测试全部14种event_type，"
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
            f"\n[全部14种事件类型测试完成] "
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
