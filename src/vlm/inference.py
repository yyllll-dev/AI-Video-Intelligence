"""
inference.py —— Qwen-VL 推理逻辑

B模块职责：
    1. 接收 A / EventEngine 已经确定的 Event
    2. 接收该事件时间窗口内的多帧图片
    3. 调用 Qwen-VL 分析真实视觉动作
    4. 补充 Event.description
    5. 返回同一个 Event 对象

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
    build_caption_prompt,
    CLASSIFY_PROMPT,
    EVENT_TYPES,
)
from ..event.schemas import Event


# ============================================================
# Qwen-VL 多帧推理
# ============================================================

def run_vlm(
    model,
    processor,
    image_paths: list[str],
    prompt: str,
) -> str:
    """
    输入：
        image_paths：
            多帧图片路径，必须按照时间顺序排列。

        prompt：
            本次Qwen-VL分析使用的Prompt。

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
    # ========================================================

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
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
# 正式接口
# ============================================================

def analyze_event(
    model,
    processor,
    event: Event,
    frame_paths: list[str],
) -> Event:
    """
    【正式接口】

    给A / EventEngine使用。

    输入：
        event：
            A已经确定event_type的Event对象。

        frame_paths：
            当前事件时间窗口中的多帧图片，
            必须按照时间顺序排列。

    输出：
        补充description后的同一个Event对象。

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

    # ========================================================
    # 构造Prompt
    # ========================================================

    prompt = build_caption_prompt(
        event.event_type
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

    # ========================================================
    # 只补充description
    # ========================================================

    description = parsed.get(
        "description",
        "",
    )

    if description is None:
        description = ""

    event.description = str(
        description
    )

    # ========================================================
    # event_confirmed
    # ========================================================

    event_confirmed = parsed.get(
        "event_confirmed",
        None,
    )

    if event_confirmed is False:

        print(
            "[VLM Warning] "
            f"画面证据可能不支持事件："
            f"{event.event_type}"
        )

    elif event_confirmed is True:

        print(
            "[VLM Confirmed] "
            f"画面支持事件："
            f"{event.event_type}"
        )

    else:

        print(
            "[VLM Warning] "
            "模型没有返回有效的 event_confirmed"
        )

    # ========================================================
    # 注意：
    #
    # 以下字段目前只作为VLM分析结果使用，
    # 不直接塞入Event。
    #
    # Event统一结构目前只有：
    #
    # event_type
    # start_time
    # end_time
    # track_id
    # confidence
    # description
    # ========================================================

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

    args = parser.parse_args()

    # ========================================================
    # 加载模型
    # ========================================================

    model, processor = load_model()

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

        result_event = analyze_event(
            model=model,
            processor=processor,
            event=incoming_event,
            frame_paths=args.image,
        )

        latency = time.time() - t0

        print(
            f"\n[VLM Latency] "
            f"{latency:.2f} 秒"
        )

        print(
            "\n[补全后的Event]"
        )

        print(result_event)

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