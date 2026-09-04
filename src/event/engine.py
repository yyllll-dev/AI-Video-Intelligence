from typing import Dict, List, Optional
from .schemas import Event, TrackingResult
from .event_types import (
    EVENT_SIT_AT_STUDY_POSITION,
    EVENT_LEAVE_STUDY_POSITION,
    EVENT_READING,
    EVENT_WRITING,
    EVENT_PHONE_LEARNING,
    EVENT_COMPUTER_LEARNING,
    EVENT_OTHER_STUDY_BEHAVIOR,
    EVENT_PHONE_DISTRACTION,
    EVENT_COMPUTER_DISTRACTION,
    EVENT_COMMUNICATION_DISTRACTION,
)
from .rules import (
    SIT_AT_STUDY_POSITION_MIN_DURATION,
    LEAVE_STUDY_POSITION_MIN_DURATION,
    STUDY_PREPARATION_MIN_DURATION,
    START_STUDY_MIN_DURATION,
    SEMANTIC_CANDIDATE_WINDOW_DURATION,
    SEMANTIC_CANDIDATE_MIN_DURATION,
)


class EventEngine:
    """
    学习场景事件理解引擎。
    输入：
        Tracker 输出的 TrackingResult 列表。
    输出：
        Event 列表。
    核心职责：
        1. 根据 TrackingResult 判断人物是否处于学习位置。
        2. 维护人物的学习状态。
        3. 生成进入和离开学习位置事件。
        4. 按固定窗口生成宽松活动候选，不提前过滤未知动作。
        5. 最终活动类型由下游 VLM 判断。
    数据流：
        VideoFrame
            ↓
        YOLO Detection
            ↓
        Tracker
            ↓
        TrackingResult
            ↓
        EventEngine
            ↓
        Event
    """

    # ============================================================
    # 初始化
    # ============================================================
    def __init__(
        self,
        semantic_window_duration: float = SEMANTIC_CANDIDATE_WINDOW_DURATION,
    ):
        """
        初始化学习状态。
        """
        # --------------------------------------------------------
        # 当前主要人物
        # --------------------------------------------------------
        self.person_track_id: Optional[int] = None
        # 最近一次看到人物的时间
        self.person_last_seen: Optional[float] = None

        # --------------------------------------------------------
        # 学习位置状态
        #
        # outside
        # sitting
        # preparing
        # studying
        # ending
        # --------------------------------------------------------
        self.state = "outside"
        # 人物开始被认为坐在学习位置的时间
        self.sit_start_time: Optional[float] = None
        # 人物真正离开学习位置的候选开始时间
        self.leave_start_time: Optional[float] = None

        # --------------------------------------------------------
        # 学习流程
        # --------------------------------------------------------
        self.preparation_start_time: Optional[float] = None
        self.study_start_time: Optional[float] = None
        # 是否已达到稳定学习状态，仅用于内部状态判断。
        self.study_started = False

        # --------------------------------------------------------
        # 当前行为
        #
        # reading
        # writing
        # phone_learning
        # computer_learning
        # other_study_behavior
        # phone_distraction
        # computer_distraction
        # communication_distraction
        # --------------------------------------------------------
        self.current_activity: Optional[str] = None
        self.activity_start_time: Optional[float] = None
        self.activity_missing_since: Optional[float] = None
        self.semantic_window_duration = float(semantic_window_duration)
        if self.semantic_window_duration <= 0:
            raise ValueError("semantic_window_duration 必须大于 0")
        self.activity_votes: Dict[str, int] = {}

        # --------------------------------------------------------
        # 学习结束整理
        # --------------------------------------------------------
        self.cleanup_start_time: Optional[float] = None

    # ============================================================
    # 主入口
    # ============================================================
    def update(
        self,
        results: List[TrackingResult],
        timestamp: Optional[float] = None,
        activity_hint: Optional[str] = None,
    ) -> List[Event]:
        """
        根据当前帧的 TrackingResult 更新事件状态。
        参数：
            results:
                Tracker 输出的 TrackingResult。
            timestamp:
                当前视频时间戳。
            activity_hint:
                后续可以由 Qwen‑VL 提供的行为语义结果。
                例如：
                    "reading"
                    "writing"
                    "phone_learning"
                    "phone_distraction"
                    "computer_learning"
                    "computer_distraction"
                    "communication_distraction"
                    "other_study_behavior"
                当前没有 VLM 时可以为 None。
        返回：
            当前时刻新产生的 Event 列表。
        """
        events: List[Event] = []

        # --------------------------------------------------------
        # 确定时间戳
        # --------------------------------------------------------
        if timestamp is None:
            if not results:
                return events
            timestamp = max(
                result.timestamp
                for result in results
            )

        # --------------------------------------------------------
        # 找到当前人物
        # --------------------------------------------------------
        persons = [
            result
            for result in results
            if result.class_name == "person"
        ]
        # 没有检测到人物
        if not persons:
            events.extend(
                self._handle_person_absence(timestamp)
            )
            return events

        # --------------------------------------------------------
        # 当前主要人物
        #
        # 当前版本默认场景中只有一个学习者。
        # 后续如果需要多人学习，可以再扩展。
        # --------------------------------------------------------
        person = self._select_person(persons)

        # --------------------------------------------------------
        # 更新人物状态
        # --------------------------------------------------------
        events.extend(
            self._handle_person(
                person=person,
                results=results,
                timestamp=timestamp,
            )
        )

        # --------------------------------------------------------
        # 如果人物已经坐在学习位置
        # 才继续判断学习流程和学习行为。
        # --------------------------------------------------------
        if self.state in {
            "sitting",
            "preparing",
            "studying",
            "ending",
        }:
            # 判断学习位置
            at_study_position = self._is_at_study_position(
                person,
                results,
            )
            if not at_study_position:
                events.extend(
                    self._handle_leave_candidate(timestamp)
                )
            else:
                # 人物重新回到学习位置
                self.leave_start_time = None
                # 学习准备 / 开始学习
                events.extend(
                    self._handle_study_process(
                        person=person,
                        results=results,
                        timestamp=timestamp,
                    )
                )
                # ------------------------------------------------
                # 行为判断
                # ------------------------------------------------
                if self.state in {"preparing", "studying", "ending"}:
                    activity = self._resolve_activity(
                        results=results,
                        activity_hint=activity_hint,
                    )
                    events.extend(
                        self._handle_activity(
                            activity=activity,
                            person=person,
                            timestamp=timestamp,
                        )
                    )

        return events

    # ============================================================
    # 人物选择
    # ============================================================
    @staticmethod
    def _select_person(
        persons: List[TrackingResult],
    ) -> TrackingResult:
        """
        当前版本默认选择置信度最高的人物。
        后续多人场景可以改成：
            - 指定学习者 track_id
            - 区域过滤
            - 多人独立状态机
        """
        return max(
            persons,
            key=lambda person: person.confidence,
        )

    # ============================================================
    # 人物状态处理
    # ============================================================
    def _handle_person(
        self,
        person: TrackingResult,
        results: List[TrackingResult],
        timestamp: float,
    ) -> List[Event]:
        events: List[Event] = []
        track_id = person.track_id

        # --------------------------------------------------------
        # 第一次发现人物
        # --------------------------------------------------------
        if self.person_track_id is None:
            self.person_track_id = track_id
            self.person_last_seen = timestamp
            # 注意：
            # 第一次看到人不能直接认为“坐到学习位置”。
            # 必须通过家具空间关系判断。
            if self._is_at_study_position(person, results):
                self.sit_start_time = timestamp
                self.state = "sitting"
            return events

        # --------------------------------------------------------
        # 同一个人物
        # --------------------------------------------------------
        if track_id == self.person_track_id:
            self.person_last_seen = timestamp
            if (
                self.state == "outside"
                and self._is_at_study_position(person, results)
            ):
                self.sit_start_time = timestamp
                self.leave_start_time = None
                self.state = "sitting"
            return events

        # --------------------------------------------------------
        # Track ID 发生变化
        #
        # 这里通常意味着 Tracker 换了 ID。
        # 当前版本重新绑定人物。
        # --------------------------------------------------------
        self.person_last_seen = timestamp
        if self.state == "outside":
            self.person_track_id = track_id
            if self._is_at_study_position(person, results):
                self.sit_start_time = timestamp
                self.state = "sitting"
        return events

    # ============================================================
    # 学习位置判断
    # ============================================================
    def _is_at_study_position(
        self,
        person: TrackingResult,
        results: List[TrackingResult],
    ) -> bool:
        """
        判断人物是否位于学习位置。
        当前版本采用“人物 + 学习家具”的空间关系。
        优先寻找：
            chair
            desk
            dining table
        注意：
            COCO 模型没有专门的“学习桌”类别，
            因此 dining table 只能作为桌面目标的基础替代。
        后续如果使用自训练 YOLO，
        可以增加：
            study_desk
            study_chair
        然后在这里直接加入对应类别。
        当前这里只做基础空间判断，
        不依赖固定的像素坐标。
        """
        furniture = [
            result
            for result in results
            if result.class_name in {
                "chair",
                "desk",
                "dining table",
                "study_desk",
                "study_chair",
            }
        ]
        if not furniture:
            return False

        person_center = self._bbox_center(person.bbox)

        # --------------------------------------------------------
        # 如果检测到了 chair
        # 判断人物是否位于椅子附近。
        # --------------------------------------------------------
        chairs = [
            item
            for item in furniture
            if item.class_name in {
                "chair",
                "study_chair",
            }
        ]
        for chair in chairs:
            chair_center = self._bbox_center(chair.bbox)
            if self._nearby(
                person_center,
                chair_center,
                person.bbox,
                chair.bbox,
            ):
                return True

        # --------------------------------------------------------
        # 如果没有椅子，使用桌面作为辅助判断。
        # --------------------------------------------------------
        tables = [
            item
            for item in furniture
            if item.class_name in {
                "desk",
                "dining table",
                "study_desk",
            }
        ]
        for table in tables:
            table_center = self._bbox_center(table.bbox)
            if self._nearby(
                person_center,
                table_center,
                person.bbox,
                table.bbox,
            ):
                return True

        return False

    # ============================================================
    # Bounding Box 工具
    # ============================================================
    @staticmethod
    def _bbox_center(
        bbox: List[float],
    ) -> tuple[float, float]:
        if len(bbox) != 4:
            return 0.0, 0.0
        x1, y1, x2, y2 = bbox
        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )

    @staticmethod
    def _nearby(
        person_center: tuple[float, float],
        object_center: tuple[float, float],
        person_bbox: List[float],
        object_bbox: List[float],
    ) -> bool:
        """
        判断人物和家具是否处于合理空间关系。
        不使用固定像素阈值，
        而是根据人物 bbox 尺寸进行归一化。
        这样不同分辨率下更加稳定。
        """
        if len(person_bbox) != 4:
            return False
        px, py = person_center
        ox, oy = object_center
        person_width = max(
            person_bbox[2] - person_bbox[0],
            1.0,
        )
        person_height = max(
            person_bbox[3] - person_bbox[1],
            1.0,
        )
        dx = abs(px - ox) / person_width
        dy = abs(py - oy) / person_height
        return dx <= 2.5 and dy <= 2.0

    # ============================================================
    # 学习流程
    # ============================================================
    def _handle_study_process(
        self,
        person: TrackingResult,
        results: List[TrackingResult],
        timestamp: float,
    ) -> List[Event]:
        events: List[Event] = []

        # --------------------------------------------------------
        # 第一次确认坐到学习位置
        # --------------------------------------------------------
        if self.sit_start_time is None:
            self.sit_start_time = timestamp
        sit_duration = timestamp - self.sit_start_time

        # --------------------------------------------------------
        # sitting → preparation
        # --------------------------------------------------------
        if (
            self.state == "sitting"
            and sit_duration >= SIT_AT_STUDY_POSITION_MIN_DURATION
        ):
            self.state = "preparing"
            events.append(
                self._create_event(
                    event_type=EVENT_SIT_AT_STUDY_POSITION,
                    start_time=self.sit_start_time,
                    end_time=timestamp,
                    person=person,
                    description="学生坐到学习位置",
                )
            )
            self.preparation_start_time = timestamp

        # --------------------------------------------------------
        # preparation
        # --------------------------------------------------------
        if self.state == "preparing":
            if self.preparation_start_time is None:
                self.preparation_start_time = timestamp
            preparation_duration = timestamp - self.preparation_start_time
            if preparation_duration >= STUDY_PREPARATION_MIN_DURATION:
                self.state = "studying"
                self.study_start_time = timestamp
                self.study_started = False

        # --------------------------------------------------------
        # 正式开始学习
        # --------------------------------------------------------
        if self.state == "studying":
            if self.study_start_time is None:
                self.study_start_time = timestamp
            study_duration = timestamp - self.study_start_time
            if (
                not self.study_started
                and study_duration >= START_STUDY_MIN_DURATION
            ):
                self.study_started = True

        return events

    # ============================================================
    # 行为解析
    # ============================================================
    def _resolve_activity(
        self,
        results: List[TrackingResult],
        activity_hint: Optional[str] = None,
    ) -> Optional[str]:
        """
        确定当前行为。
        优先级：
            1. Qwen‑VL / 上层模块提供的 activity_hint
            2. YOLO 基础目标规则
        这样以后接入 Qwen‑VL 时，
        不需要修改 EventEngine 的整体结构。
        """
        # --------------------------------------------------------
        # VLM 已经判断
        # --------------------------------------------------------
        if activity_hint in {
            EVENT_READING,
            EVENT_WRITING,
            EVENT_PHONE_LEARNING,
            EVENT_COMPUTER_LEARNING,
            EVENT_OTHER_STUDY_BEHAVIOR,
            EVENT_PHONE_DISTRACTION,
            EVENT_COMPUTER_DISTRACTION,
            EVENT_COMMUNICATION_DISTRACTION,
        }:
            return activity_hint

        # --------------------------------------------------------
        # 暂时没有 VLM
        #
        # 使用 YOLO 可直接观察到的目标做基础判断。
        # --------------------------------------------------------
        return self._basic_activity_from_objects(results)

    # ============================================================
    # 基础行为判断
    # ============================================================
    @staticmethod
    def _basic_activity_from_objects(
        results: List[TrackingResult],
    ) -> Optional[str]:
        """
        根据 YOLO 当前能看到的物体，
        给出一个“基础行为候选”。
        注意：
            这里不是最终的行为理解。
        例如：
            book → reading 候选
            cell phone → 无法判断是学习还是分心
            laptop → 无法判断是学习还是分心
        手机和电脑只作为候选提示，最终类型由 Qwen‑VL 纠正。
        """
        class_names = {
            result.class_name
            for result in results
        }

        # --------------------------------------------------------
        # 书本
        # --------------------------------------------------------
        if "book" in class_names:
            return EVENT_READING

        # --------------------------------------------------------
        # 笔
        #
        # 当前 COCO YOLO 通常无法可靠检测 pen，
        # 如果以后自训练模型增加 pen，可以直接使用。
        # --------------------------------------------------------
        if "pen" in class_names:
            return EVENT_WRITING

        # --------------------------------------------------------
        # 手机
        #
        # 不直接判断为 phone_learning 或 distraction。
        # 必须交给 VLM。
        # --------------------------------------------------------
        if (
            "cell phone" in class_names
            or "phone" in class_names
            or "mobile phone" in class_names
        ):
            return EVENT_PHONE_DISTRACTION

        # --------------------------------------------------------
        # 电脑
        #
        # 同样需要 VLM 判断是在学习还是分心。
        # --------------------------------------------------------
        if (
            "laptop" in class_names
            or "computer" in class_names
        ):
            return EVENT_COMPUTER_LEARNING

        # --------------------------------------------------------
        # 没有明确目标
        # --------------------------------------------------------
        return EVENT_OTHER_STUDY_BEHAVIOR

    # ============================================================
    # 学习行为状态处理
    # ============================================================
    def _handle_activity(
        self,
        activity: Optional[str],
        person: TrackingResult,
        timestamp: float,
    ) -> List[Event]:
        events: List[Event] = []

        candidate = activity or EVENT_OTHER_STUDY_BEHAVIOR
        if self.activity_start_time is None:
            self.activity_start_time = timestamp
        self.current_activity = candidate
        self.activity_missing_since = None

        self.activity_votes[candidate] = self.activity_votes.get(candidate, 0) + 1
        if timestamp - self.activity_start_time < self.semantic_window_duration:
            return events

        event = self._finish_activity_window(timestamp, person)
        if event is not None:
            events.append(event)
        return events

    def _finish_activity_window(
        self,
        end_time: float,
        person: Optional[TrackingResult] = None,
    ) -> Optional[Event]:
        if self.activity_start_time is None:
            return None
        start_time = self.activity_start_time
        duration = end_time - start_time
        if duration < SEMANTIC_CANDIDATE_MIN_DURATION:
            return None

        event_type = max(
            self.activity_votes,
            key=self.activity_votes.get,
            default=EVENT_OTHER_STUDY_BEHAVIOR,
        )
        event = Event(
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            track_id=self.person_track_id,
            confidence=person.confidence if person is not None else 1.0,
            description="待 VLM 识别的学习行为候选",
        )
        self.activity_start_time = end_time
        self.current_activity = None
        self.activity_missing_since = None
        self.activity_votes = {}
        return event

    # ============================================================
    # 离开学习位置
    # ============================================================
    def _handle_leave_candidate(
        self,
        timestamp: float,
    ) -> List[Event]:
        events: List[Event] = []

        # --------------------------------------------------------
        # 第一次发现可能离开
        # --------------------------------------------------------
        if self.leave_start_time is None:
            self.leave_start_time = timestamp
            return events

        leave_duration = timestamp - self.leave_start_time

        # --------------------------------------------------------
        # 持续离开达到阈值
        # --------------------------------------------------------
        if (
            leave_duration >= LEAVE_STUDY_POSITION_MIN_DURATION
            and self.state != "outside"
        ):
            # ----------------------------------------------------
            # 如果之前正在学习，先结束当前学习行为。
            # ----------------------------------------------------
            activity_event = self._finish_activity_window(self.leave_start_time)
            if activity_event is not None:
                events.append(activity_event)

            # ----------------------------------------------------
            # 产生“离开学习位置”
            # ----------------------------------------------------
            events.append(
                Event(
                    event_type=EVENT_LEAVE_STUDY_POSITION,
                    start_time=self.leave_start_time,
                    end_time=timestamp,
                    track_id=self.person_track_id,
                    confidence=1.0,
                    description="学生离开学习位置",
                )
            )

            self.state = "outside"
            self._reset_learning_state(keep_person=True)

        return events

    # ============================================================
    # 人物消失
    # ============================================================
    def _handle_person_absence(
        self,
        timestamp: float,
    ) -> List[Event]:
        events: List[Event] = []
        if self.person_track_id is None:
            return events
        if self.person_last_seen is None:
            return events

        # --------------------------------------------------------
        # 注意：
        # 人物只消失一两帧不能马上认为离开。
        #
        # Tracker 可能暂时丢失目标。
        # 因此这里只记录离开候选。
        # --------------------------------------------------------
        if self.leave_start_time is None:
            self.leave_start_time = self.person_last_seen

        absence_duration = timestamp - self.leave_start_time
        if (
            absence_duration >= LEAVE_STUDY_POSITION_MIN_DURATION
        ):
            if self.state != "outside":
                activity_event = self._finish_activity_window(self.leave_start_time)
                if activity_event is not None:
                    events.append(activity_event)

                events.append(
                    Event(
                        event_type=EVENT_LEAVE_STUDY_POSITION,
                        start_time=self.leave_start_time,
                        end_time=timestamp,
                        track_id=self.person_track_id,
                        confidence=1.0,
                        description="学生离开学习位置",
                    )
                )

                self.state = "outside"
                self._reset_learning_state(keep_person=True)

        return events

    # ============================================================
    # 状态清理
    # ============================================================
    def _reset_learning_state(
        self,
        keep_person: bool = True,
    ) -> None:
        if not keep_person:
            self.person_track_id = None
            self.person_last_seen = None
        self.sit_start_time = None
        self.leave_start_time = None
        self.preparation_start_time = None
        self.study_start_time = None
        self.study_started = False
        self.current_activity = None
        self.activity_start_time = None
        self.activity_missing_since = None
        self.activity_votes = {}
        self.cleanup_start_time = None

    # ============================================================
    # Event 创建
    # ============================================================
    def _create_event(
        self,
        event_type: str,
        start_time: float,
        end_time: float,
        person: TrackingResult,
        description: str,
    ) -> Event:
        return Event(
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            track_id=(
                self.person_track_id
                if self.person_track_id is not None
                else person.track_id
            ),
            confidence=person.confidence,
            description=description,
        )

    def finalize(self, timestamp: float) -> List[Event]:
        """在视频结束或实时分析停止时，以最后一帧时间关闭活动。"""
        events: List[Event] = []
        timestamp = float(timestamp)
        activity_event = self._finish_activity_window(timestamp)
        if activity_event is not None:
            events.append(activity_event)

        self._reset_learning_state(keep_person=False)
        self.state = "outside"
        return events

    # ============================================================
    # 行为描述
    # ============================================================
    @staticmethod
    def _activity_description(
        activity: str,
    ) -> str:
        descriptions = {
            EVENT_READING: "学生正在阅读",
            EVENT_WRITING: "学生正在书写",
            EVENT_PHONE_LEARNING: "学生正在使用手机学习",
            EVENT_COMPUTER_LEARNING: "学生正在使用电脑学习",
            EVENT_OTHER_STUDY_BEHAVIOR: "学生正在进行其他学习行为",
            EVENT_PHONE_DISTRACTION: "学生正在使用手机分心",
            EVENT_COMPUTER_DISTRACTION: "学生正在使用电脑分心",
            EVENT_COMMUNICATION_DISTRACTION: "学生正在进行交流分心",
        }
        return descriptions.get(
            activity,
            "学生正在进行学习相关行为",
        )
