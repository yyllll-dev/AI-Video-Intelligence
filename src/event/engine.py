from typing import Dict, List, Optional

from .schemas import Event, TrackingResult
from .event_types import (
    EVENT_ENTER_STUDY,
    EVENT_LEAVE_STUDY,
    EVENT_START_STUDY,
    EVENT_READING,
    EVENT_WRITING,
    EVENT_COMPUTER_USAGE,
    EVENT_PHONE_USAGE,
    EVENT_PICK_UP_PHONE,
    EVENT_PUT_DOWN_PHONE,
    EVENT_SWITCH_TO_PHONE,
    EVENT_RETURN_TO_STUDY,
    EVENT_STUDY_INTERRUPTION,
    EVENT_LONG_PHONE_USAGE,
    EVENT_LONG_ABSENCE,
    EVENT_COMPUTER_LEARNING,
)
from .rules import (
    START_STUDY_DURATION,
    PHONE_USAGE_MIN_DURATION,
    LONG_PHONE_USAGE_DURATION,
    LONG_ABSENCE_DURATION,
)


class EventEngine:
    """
    学习场景事件理解引擎。

    输入：
        连续视频分析得到的 TrackingResult

    输出：
        Event

    核心逻辑：

        Detection
            ↓
        Tracking
            ↓
        EventEngine
            ↓
        Event
    """

    def __init__(self):

        # ==================================================
        # 人物状态
        # ==================================================

        self.person_track_id: Optional[int] = None

        self.person_last_seen: Optional[float] = None

        self.study_entry_time: Optional[float] = None

        # outside / studying / phone
        self.state = "outside"

        # reading / writing / computer_learning
        self.current_activity: Optional[str] = None

        self.activity_start_time: Optional[float] = None

        # 是否已经产生 start_study
        self.study_started = False

        # 是否已经产生 long_absence
        self.long_absence_reported = False

        # ==================================================
        # 手机状态
        # ==================================================

        self.phone_active = False

        self.phone_start_time: Optional[float] = None

        self.phone_last_seen: Optional[float] = None

        self.long_phone_reported = False

    # ======================================================
    # 主入口
    # ======================================================

    def update(
        self,
        results: List[TrackingResult],
    ) -> List[Event]:

        events: List[Event] = []

        if not results:
            return events

        timestamp = max(
            result.timestamp
            for result in results
        )

        # --------------------------------------------------
        # 当前人物
        # --------------------------------------------------

        persons = [
            result
            for result in results
            if result.class_name == "person"
        ]

        # --------------------------------------------------
        # 当前手机
        # --------------------------------------------------

        phones = [
            result
            for result in results
            if result.class_name in {
                "cell phone",
                "phone",
                "mobile phone",
            }
        ]

        # ==================================================
        # 1. 人物状态
        # ==================================================

        if persons:

            person = persons[0]

            events.extend(
                self._handle_person(
                    person,
                    timestamp,
                )
            )

            # ==================================================
            # 2. 学习行为
            # ==================================================

            if self.state == "studying":

                events.extend(
                    self._handle_learning_activity(
                        results,
                        timestamp,
                    )
                )

        else:

            events.extend(
                self._handle_person_absence(
                    timestamp
                )
            )

        # ==================================================
        # 3. 手机状态
        # ==================================================

        if phones:

            events.extend(
                self._handle_phone(
                    phones[0],
                    timestamp,
                )
            )

        else:

            events.extend(
                self._handle_phone_absence(
                    timestamp
                )
            )

        # ==================================================
        # 4. 长时间手机使用
        # ==================================================

        events.extend(
            self._check_long_phone_usage(
                timestamp
            )
        )

        return events

    # ======================================================
    # 人物进入 / 返回
    # ======================================================

    def _handle_person(
        self,
        person: TrackingResult,
        timestamp: float,
    ) -> List[Event]:

        events = []

        track_id = person.track_id

        # --------------------------------------------------
        # 第一次发现人物
        # --------------------------------------------------

        if self.person_track_id is None:

            self.person_track_id = track_id

            self.person_last_seen = timestamp

            self.study_entry_time = timestamp

            self.state = "studying"

            self.study_started = False

            self.long_absence_reported = False

            events.append(
                Event(
                    event_type=EVENT_ENTER_STUDY,
                    start_time=timestamp,
                    end_time=timestamp,
                    track_id=track_id,
                    confidence=person.confidence,
                    description="学生进入学习区域",
                )
            )

            return events

        # --------------------------------------------------
        # 同一个人物重新出现
        # --------------------------------------------------

        if track_id == self.person_track_id:

            was_outside = self.state == "outside"

            self.person_last_seen = timestamp

            self.long_absence_reported = False

            if was_outside:

                self.state = "studying"

                self.study_entry_time = timestamp

                self.study_started = False

                events.append(
                    Event(
                        event_type=EVENT_ENTER_STUDY,
                        start_time=timestamp,
                        end_time=timestamp,
                        track_id=track_id,
                        confidence=person.confidence,
                        description="学生重新进入学习区域",
                    )
                )

                events.append(
                    Event(
                        event_type=EVENT_RETURN_TO_STUDY,
                        start_time=timestamp,
                        end_time=timestamp,
                        track_id=track_id,
                        confidence=person.confidence,
                        description="学生回到学习区域",
                    )
                )

            return events

        # --------------------------------------------------
        # Track ID 改变
        # --------------------------------------------------

        self.person_track_id = track_id

        self.person_last_seen = timestamp

        self.study_entry_time = timestamp

        self.state = "studying"

        self.current_activity = None

        self.activity_start_time = None

        self.study_started = False

        events.append(
            Event(
                event_type=EVENT_ENTER_STUDY,
                start_time=timestamp,
                end_time=timestamp,
                track_id=track_id,
                confidence=person.confidence,
                description="新的学生进入学习区域",
            )
        )

        return events

    # ======================================================
    # 学习行为
    # ======================================================

    def _handle_learning_activity(
        self,
        results: List[TrackingResult],
        timestamp: float,
    ) -> List[Event]:

        events = []

        if self.person_track_id is None:
            return events

        person = next(
            (
                result
                for result in results
                if (
                    result.class_name == "person"
                    and result.track_id == self.person_track_id
                )
            ),
            None,
        )

        if person is None:
            return events

        class_names = {
            result.class_name
            for result in results
        }

        # ==================================================
        # 判断当前学习行为
        # ==================================================

        new_activity = None

        if "book" in class_names:

            new_activity = EVENT_READING

        elif "pen" in class_names:

            new_activity = EVENT_WRITING

        elif (
            "laptop" in class_names
            or "computer" in class_names
        ):

            new_activity = EVENT_COMPUTER_LEARNING

        # 没有明确学习行为
        if new_activity is None:
            return events

        # ==================================================
        # 第一次开始学习
        # ==================================================

        if self.current_activity is None:

            self.current_activity = new_activity

            self.activity_start_time = timestamp

            if not self.study_started:

                self.study_started = True

                events.append(
                    Event(
                        event_type=EVENT_START_STUDY,
                        start_time=timestamp,
                        end_time=timestamp,
                        track_id=self.person_track_id,
                        confidence=person.confidence,
                        description="学生开始学习",
                    )
                )

            events.append(
                Event(
                    event_type=new_activity,
                    start_time=timestamp,
                    end_time=timestamp,
                    track_id=self.person_track_id,
                    confidence=person.confidence,
                    description=self._activity_description(
                        new_activity
                    ),
                )
            )

            return events

        # ==================================================
        # 学习行为发生变化
        # ==================================================

        if new_activity != self.current_activity:

            previous_activity = self.current_activity

            previous_start = self.activity_start_time

            # 结束之前的学习行为
            if previous_start is not None:

                events.append(
                    Event(
                        event_type=previous_activity,
                        start_time=previous_start,
                        end_time=timestamp,
                        track_id=self.person_track_id,
                        confidence=person.confidence,
                        description=(
                            self._activity_description(
                                previous_activity
                            )
                            + "结束"
                        ),
                    )
                )

            # 开始新的学习行为
            self.current_activity = new_activity

            self.activity_start_time = timestamp

            events.append(
                Event(
                    event_type=new_activity,
                    start_time=timestamp,
                    end_time=timestamp,
                    track_id=self.person_track_id,
                    confidence=person.confidence,
                    description=self._activity_description(
                        new_activity
                    ),
                )
            )

        return events

    # ======================================================
    # 手机
    # ======================================================

    def _handle_phone(
        self,
        phone: TrackingResult,
        timestamp: float,
    ) -> List[Event]:

        events = []

        # --------------------------------------------------
        # 手机第一次出现
        # --------------------------------------------------

        if not self.phone_active:

            self.phone_active = True

            self.phone_start_time = timestamp

            self.phone_last_seen = timestamp

            self.long_phone_reported = False

            events.append(
                Event(
                    event_type=EVENT_PICK_UP_PHONE,
                    start_time=timestamp,
                    end_time=timestamp,
                    track_id=self.person_track_id,
                    confidence=phone.confidence,
                    description="学生拿起手机",
                )
            )

            # 如果之前在学习
            if self.state == "studying":

                self.state = "phone"

                # 学习行为被打断
                events.append(
                    Event(
                        event_type=EVENT_STUDY_INTERRUPTION,
                        start_time=timestamp,
                        end_time=timestamp,
                        track_id=self.person_track_id,
                        confidence=phone.confidence,
                        description="学习被手机使用打断",
                    )
                )

                events.append(
                    Event(
                        event_type=EVENT_SWITCH_TO_PHONE,
                        start_time=timestamp,
                        end_time=timestamp,
                        track_id=self.person_track_id,
                        confidence=phone.confidence,
                        description="学生从学习状态切换到使用手机",
                    )
                )

            return events

        # --------------------------------------------------
        # 手机继续出现
        # --------------------------------------------------

        self.phone_last_seen = timestamp

        return events

    # ======================================================
    # 手机消失
    # ======================================================

    def _handle_phone_absence(
        self,
        timestamp: float,
    ) -> List[Event]:

        events = []

        if not self.phone_active:
            return events

        if self.phone_last_seen is None:
            return events

        # --------------------------------------------------
        # 放下手机
        # --------------------------------------------------

        events.append(
            Event(
                event_type=EVENT_PUT_DOWN_PHONE,
                start_time=self.phone_last_seen,
                end_time=timestamp,
                track_id=self.person_track_id,
                confidence=1.0,
                description="学生放下手机",
            )
        )

        # --------------------------------------------------
        # 计算手机使用时间
        # --------------------------------------------------

        if self.phone_start_time is not None:

            duration = (
                self.phone_last_seen
                - self.phone_start_time
            )

            if duration >= PHONE_USAGE_MIN_DURATION:

                events.append(
                    Event(
                        event_type=EVENT_PHONE_USAGE,
                        start_time=self.phone_start_time,
                        end_time=self.phone_last_seen,
                        track_id=self.person_track_id,
                        confidence=1.0,
                        description=(
                            f"学生使用手机 "
                            f"{duration:.1f} 秒"
                        ),
                    )
                )

        # --------------------------------------------------
        # 清除手机状态
        # --------------------------------------------------

        self.phone_active = False

        self.phone_start_time = None

        self.phone_last_seen = None

        self.long_phone_reported = False

        # --------------------------------------------------
        # 回到学习
        # --------------------------------------------------

        if self.state == "phone":

            self.state = "studying"

            self.current_activity = None

            self.activity_start_time = timestamp

            events.append(
                Event(
                    event_type=EVENT_RETURN_TO_STUDY,
                    start_time=timestamp,
                    end_time=timestamp,
                    track_id=self.person_track_id,
                    confidence=1.0,
                    description="学生放下手机并回到学习状态",
                )
            )

        return events

    # ======================================================
    # 长时间使用手机
    # ======================================================

    def _check_long_phone_usage(
        self,
        timestamp: float,
    ) -> List[Event]:

        events = []

        if not self.phone_active:
            return events

        if self.phone_start_time is None:
            return events

        duration = (
            timestamp
            - self.phone_start_time
        )

        if (
            duration >= LONG_PHONE_USAGE_DURATION
            and not self.long_phone_reported
        ):

            events.append(
                Event(
                    event_type=EVENT_LONG_PHONE_USAGE,
                    start_time=self.phone_start_time,
                    end_time=timestamp,
                    track_id=self.person_track_id,
                    confidence=1.0,
                    description=(
                        f"学生长时间使用手机 "
                        f"{duration:.1f} 秒"
                    ),
                )
            )

            self.long_phone_reported = True

        return events

    # ======================================================
    # 人物离开
    # ======================================================

    def _handle_person_absence(
        self,
        timestamp: float,
    ) -> List[Event]:

        events = []

        if self.person_track_id is None:
            return events

        if self.person_last_seen is None:
            return events

        absence_duration = (
            timestamp
            - self.person_last_seen
        )

        # --------------------------------------------------
        # 长时间离开
        # --------------------------------------------------

        if (
            absence_duration >= LONG_ABSENCE_DURATION
            and not self.long_absence_reported
        ):

            events.append(
                Event(
                    event_type=EVENT_LONG_ABSENCE,
                    start_time=self.person_last_seen,
                    end_time=timestamp,
                    track_id=self.person_track_id,
                    confidence=1.0,
                    description=(
                        f"学生长时间离开学习区域 "
                        f"{absence_duration:.1f} 秒"
                    ),
                )
            )

            self.long_absence_reported = True

        # --------------------------------------------------
        # 离开学习区域
        # --------------------------------------------------

        if (
            absence_duration >= LONG_ABSENCE_DURATION
            and self.state != "outside"
        ):

            events.append(
                Event(
                    event_type=EVENT_LEAVE_STUDY,
                    start_time=self.person_last_seen,
                    end_time=timestamp,
                    track_id=self.person_track_id,
                    confidence=1.0,
                    description="学生离开学习区域",
                )
            )

            self.state = "outside"

            self.current_activity = None

            self.activity_start_time = None

        return events

    # ======================================================
    # 描述
    # ======================================================

    @staticmethod
    def _activity_description(
        activity: str,
    ) -> str:

        descriptions = {

            EVENT_READING:
                "学生正在阅读",

            EVENT_WRITING:
                "学生正在书写",

            EVENT_COMPUTER_USAGE:
                "学生正在使用电脑",

            EVENT_COMPUTER_LEARNING:
                "学生正在使用电脑学习",

        }

        return descriptions.get(
            activity,
            "学生正在学习",
        )