"""学习场景事件类型定义

本文件统一定义项目中的学习行为事件。
所有事件类型均在 ALL_EVENTS 中注册，供 EventEngine、
VLM 以及后续 Memory / Retrieval 模块统一使用。
"""


# ============================================================
# 学习位置
# ============================================================

# 学生坐到学习位置
EVENT_SIT_AT_STUDY_POSITION = "sit_at_study_position"

# 学生离开学习位置
EVENT_LEAVE_STUDY_POSITION = "leave_study_position"


# ============================================================
# 学习过程
# ============================================================

# 学生开始进行学习前的准备行为
EVENT_STUDY_PREPARATION = "study_preparation"

# 学生正式开始学习
EVENT_START_STUDY = "start_study"

# 学生结束本次学习
EVENT_END_STUDY = "end_study"


# ============================================================
# 具体学习行为
# ============================================================

# 阅读书籍、资料等
EVENT_READING = "reading"

# 书写、做题、记笔记等
EVENT_WRITING = "writing"

# 使用手机进行学习
EVENT_PHONE_LEARNING = "phone_learning"

# 使用电脑进行学习
EVENT_COMPUTER_LEARNING = "computer_learning"

# 无法归入上述类别的其他学习行为
EVENT_OTHER_STUDY_BEHAVIOR = "other_study_behavior"


# ============================================================
# 分心行为
# ============================================================

# 使用手机，但不是为了学习
EVENT_PHONE_DISTRACTION = "phone_distraction"

# 使用电脑，但不是为了学习
EVENT_COMPUTER_DISTRACTION = "computer_distraction"

# 与他人交流导致的分心
EVENT_COMMUNICATION_DISTRACTION = "communication_distraction"


# ============================================================
# 学习结束
# ============================================================

# 学习结束后的整理行为
EVENT_STUDY_END_CLEANUP = "study_end_cleanup"


# ============================================================
# 全部事件
# ============================================================

ALL_EVENTS = [
    EVENT_SIT_AT_STUDY_POSITION,
    EVENT_LEAVE_STUDY_POSITION,

    EVENT_STUDY_PREPARATION,
    EVENT_START_STUDY,
    EVENT_END_STUDY,

    EVENT_READING,
    EVENT_WRITING,
    EVENT_PHONE_LEARNING,
    EVENT_COMPUTER_LEARNING,
    EVENT_OTHER_STUDY_BEHAVIOR,

    EVENT_PHONE_DISTRACTION,
    EVENT_COMPUTER_DISTRACTION,
    EVENT_COMMUNICATION_DISTRACTION,

    EVENT_STUDY_END_CLEANUP,
]