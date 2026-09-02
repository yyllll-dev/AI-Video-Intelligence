"""学习场景的事件类型与物体类别映射（中英文对照）。

事件名（英文 key）是跨模块通信的规范标识，与团队「明日项目任务安排」PDF 对齐：
全组统一使用以下 14 种事件名，不要再出现不同模块使用不同事件名的情况。

中文 label 仅用于界面展示。
"""

# ============ 14 种事件类型（与团队 PDF 对齐） ============

EVENT_LABELS = {
    "sit_at_study_position": "坐到学习位置",
    "leave_study_position": "离开学习位置",
    "study_preparation": "学习准备",
    "start_study": "开始学习",
    "end_study": "结束学习",
    "reading": "阅读",
    "writing": "写字",
    "phone_learning": "手机学习",
    "computer_learning": "电脑学习",
    "other_study_behavior": "其他学习行为",
    "phone_distraction": "手机分心",
    "computer_distraction": "电脑分心",
    "communication_distraction": "交流分心",
    "study_end_cleanup": "学习结束整理",
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
}


def event_label(event_type: str) -> str:
    """把英文事件名转成中文显示名，未知类型原样返回。"""
    return EVENT_LABELS.get(event_type, event_type)


def class_label(class_name: str) -> str:
    """把检测物体的英文 class_name 转成中文显示名，未知类型原样返回。"""
    return CLASS_LABELS.get(class_name, class_name)
