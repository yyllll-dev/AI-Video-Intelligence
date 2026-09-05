"""学习场景正式事件类型。

对外协议只包含八个可展示事件。学习会话的开始、结束仍由 EventEngine
内部状态维护，但不再作为 Event/VLM/Memory 字段暴露。
"""


EVENT_SIT_AT_STUDY_POSITION = "sit_at_study_position"
EVENT_LEAVE_STUDY_POSITION = "leave_study_position"

EVENT_READING = "reading"
EVENT_WRITING = "writing"
EVENT_PHONE_USAGE = "phone_usage"
EVENT_COMPUTER_USAGE = "computer_usage"
EVENT_COMMUNICATION_DISTRACTION = "communication_distraction"
EVENT_OTHER_BEHAVIOR = "other_behavior"


ALL_EVENTS = [
    EVENT_SIT_AT_STUDY_POSITION,
    EVENT_LEAVE_STUDY_POSITION,
    EVENT_READING,
    EVENT_WRITING,
    EVENT_PHONE_USAGE,
    EVENT_COMPUTER_USAGE,
    EVENT_COMMUNICATION_DISTRACTION,
    EVENT_OTHER_BEHAVIOR,
]

ACTIVE_EVENTS = list(ALL_EVENTS)


def is_output_event_type(event_type: str) -> bool:
    """只有八个正式事件可以进入 Memory 和时间轴。"""
    return event_type in ALL_EVENTS
