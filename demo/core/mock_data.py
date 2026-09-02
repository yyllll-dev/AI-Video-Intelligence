"""模拟各后端模块的输出（联调前的假数据占位）。

每个函数对应一个后端模块的输出，字段结构尽量贴近 src/ 里的真实 schema：

    - mock_detections   → 对应 A（Detection/Tracking）的检测结果
    - mock_current_event → 对应 A（EventEngine）的当前事件
    - mock_timeline     → 对应 A（EventEngine）的历史事件记录
    - mock_vlm_result   → 对应 B（Qwen-VL）的语义理解结果
    - mock_search       → 对应 D（Retrieval）的检索结果

明天各模块就绪后，只需把对应函数替换成真实调用，UI 层无需改动。
"""


def mock_detections():
    """当前帧检测到的人/物（模拟 Detection/Tracking 输出）。"""
    return [
        {"track_id": 1, "class_name": "person", "confidence": 0.95,
         "bbox": [150, 80, 430, 460]},
        {"track_id": 2, "class_name": "cell phone", "confidence": 0.88,
         "bbox": [260, 300, 360, 380]},
        {"track_id": 3, "class_name": "book", "confidence": 0.82,
         "bbox": [180, 320, 420, 440]},
    ]


def mock_current_event():
    """当前正在发生的事件（模拟 EventEngine 输出）。"""
    return {
        "event_type": "phone_distraction",
        "start_time": "10:32",
        "end_time": "10:40",
        "track_id": 1,
        "confidence": 0.91,
        "description": "学生使用手机，学习被打断",
    }


def mock_timeline():
    """历史事件记录列表，元素为 (时间 MM:SS, 中文显示文字)。"""
    return [
        ("00:12", "开始学习"),
        ("00:15", "学习准备"),
        ("02:05", "学生正在书写"),
        ("05:30", "阅读"),
        ("08:12", "电脑学习"),
        ("10:32", "手机分心"),  # 当前事件（与 mock_current_event 对齐）
    ]


def mock_vlm_result():
    """VLM 语义理解结果（模拟 Qwen-VL 输出）。"""
    return "画面中一名学生正低头看手机，桌面摊开一本书，学习状态被手机使用打断。"


def mock_search(query: str):
    """检索结果（模拟 Retrieval 输出），列：时间 / 事件 / 描述 / 视频片段。"""
    # 忽略 query，返回固定假结果；接入 D 后替换为真实 search(query)
    return [
        ["10:32", "手机分心", "学生使用手机，学习被打断", "segment_632.mp4"],
        ["08:12", "手机分心", "学生拿起手机查看消息", "segment_412.mp4"],
        ["02:05", "写字", "学生正在书写作业", "segment_205.mp4"],
    ]
