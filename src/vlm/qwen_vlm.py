"""
qwen_vlm.py —— Qwen-VL 模型加载模块

B 模块职责：
    只负责加载 Qwen2-VL 模型和 Processor。

不负责：
    - Prompt 构造
    - 视频帧处理
    - Event 判断
    - Event description 生成
    - JSON 解析

正式数据流：

    Event
      ↓
    inference.py
      ↓
    Qwen2-VL
      ↓
    description
"""

import os
from pathlib import Path
from typing import Optional

from modelscope import snapshot_download
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)


# ============================================================
# 模型配置
# ============================================================

# 当前默认使用 2B 版本。
# 优点：显存/内存要求相对较低，适合当前开发和联调。
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# 如果机器显存充足，可以改成：
#
# MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"


# ============================================================
# 模型加载
# ============================================================

def load_model(
    device_map: str = "auto",
    model_path: Optional[str] = None,
):
    """
    加载 Qwen2-VL 模型和 Processor。

    参数：
        device_map:
            "auto"：
                有 GPU 时由 Transformers 自动分配；
                没有 GPU 时通常使用 CPU。

            "cpu"：
                强制使用 CPU。

    返回：
        model:
            Qwen2-VL 模型

        processor:
            Qwen2-VL Processor
    """

    configured_path = model_path or os.getenv("QWEN_VL_MODEL_PATH")

    # --------------------------------------------------------
    # 1. 下载或定位模型
    # --------------------------------------------------------

    if configured_path:
        model_dir = Path(configured_path).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Qwen2-VL 本地模型目录不存在: {model_dir}"
            )
        print(f"[模型] 使用本地目录：{model_dir}")
    else:
        print(
            f"[模型] 正在通过 ModelScope 下载/定位：{MODEL_NAME}"
        )
        model_dir = snapshot_download(MODEL_NAME)

    print(
        f"[模型本地路径] {model_dir}"
    )

    # --------------------------------------------------------
    # 2. 加载模型
    # --------------------------------------------------------

    print("[模型] 正在加载 Qwen2-VL ...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map=device_map,
    )

    # --------------------------------------------------------
    # 3. 加载 Processor
    # --------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        model_dir
    )

    print("[模型加载完成]")

    return model, processor
