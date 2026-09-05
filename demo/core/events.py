"""学习场景的事件类型与物体类别映射（中英文对照）。

事件名（英文 key）是跨模块通信的规范标识。手机、电脑不再区分学习
和分心用途；系统只接受当前八种正式事件。

中文 label 仅用于界面展示。
"""

# ============ 当前八种正式事件 ============

EVENT_LABELS = {
    "sit_at_study_position": "坐到学习位置",
    "leave_study_position": "离开学习位置",
    "reading": "阅读",
    "writing": "写字",
    "phone_usage": "使用手机",
    "computer_usage": "使用电脑",
    "other_behavior": "其他",
    "communication_distraction": "交流分心",
}


# ============ 检测物体类别（YOLO class_name → 中文） ============

CLASS_LABELS = {
    "person": "人",
    "book": "书",
    "pen": "笔",
    "cell phone": "手机",
    "phone": "手机",
    "mobile phone": "手机",
    "laptop": "电脑",
    "computer": "电脑",
    "keyboard": "键盘",
    "mouse": "鼠标",
}


def event_label(event_type: str) -> str:
    """把英文事件名转成中文显示名，未知类型原样返回。"""
    return EVENT_LABELS.get(event_type, event_type)


def class_label(class_name: str) -> str:
    """把检测物体的英文 class_name 转成中文显示名，未知类型原样返回。"""
    return CLASS_LABELS.get(class_name, class_name)
