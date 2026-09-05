from typing import Dict, List, Optional
from .schemas import Event, TrackingResult
from .event_types import (
    EVENT_SIT_AT_STUDY_POSITION,
    EVENT_LEAVE_STUDY_POSITION,
    EVENT_READING,
    EVENT_WRITING,
    EVENT_PHONE_USAGE,
    EVENT_COMPUTER_USAGE,
    EVENT_OTHER_BEHAVIOR,
    EVENT_COMMUNICATION_DISTRACTION,
)
from .rules import (
    SIT_AT_STUDY_POSITION_MIN_DURATION,
    LEAVE_STUDY_POSITION_MIN_DURATION,
    STUDY_PREPARATION_MIN_DURATION,
    START_STUDY_MIN_DURATION,
    SEMANTIC_CANDIDATE_WINDOW_DURATION,
    SEMANTIC_CANDIDATE_MIN_DURATION,
    VISIBLE_AWAY_MIN_DURATION,
    UNKNOWN_POSITION_ENTRY_MIN_DURATION,
    LEAVE_MIN_OBSERVATIONS,
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
        self.person_last_bbox: Optional[List[float]] = None
        self.person_presence_start_time: Optional[float] = None
        self.last_position_evidence: Optional[bool] = None
        self.last_reset_reason: Optional[str] = None
        self.leave_evidence_count = 0

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
        # phone_usage
        # computer_usage
        # other_behavior
        # communication_distraction
        # --------------------------------------------------------
        self.current_activity: Optional[str] = None
        self.activity_start_time: Optional[float] = None
        self.semantic_window_duration = float(semantic_window_duration)
        if self.semantic_window_duration <= 0:
            raise ValueError("semantic_window_duration 必须大于 0")
        # 一个检测帧可以同时给多个物体候选各投一票，避免 if/return 优先级。
        self.activity_votes: Dict[str, float] = {}
        self.activity_observation_count = 0
        self.activity_unclassified_count = 0

    def debug_state(self) -> dict:
        """返回状态机快照，供逐帧链路日志和故障定位使用。"""
        return {
            "state": self.state,
            "person_track_id": self.person_track_id,
            "person_last_seen": self.person_last_seen,
            "person_last_bbox": self.person_last_bbox,
            "person_presence_start_time": self.person_presence_start_time,
            "last_position_evidence": self.last_position_evidence,
            "last_reset_reason": self.last_reset_reason,
            "leave_evidence_count": self.leave_evidence_count,
            "sit_start_time": self.sit_start_time,
            "leave_start_time": self.leave_start_time,
            "preparation_start_time": self.preparation_start_time,
            "study_start_time": self.study_start_time,
            "study_started": self.study_started,
            "current_activity": self.current_activity,
            "activity_start_time": self.activity_start_time,
            "activity_votes": dict(self.activity_votes),
            "activity_observation_count": self.activity_observation_count,
            "activity_unclassified_count": self.activity_unclassified_count,
        }

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
                    "phone_usage"
                    "computer_usage"
                    "communication_distraction"
                    "other_behavior"
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
        if person is None:
            return self._handle_person_absence(timestamp)

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
            position_evidence = self._study_position_evidence(
                person,
                results,
            )
            self.last_position_evidence = position_evidence
            # 只有“检测到了家具且人物明确远离”才是负证据。
            # 家具漏检返回 None，继续当前会话，避免姿态/遮挡造成假离开。
            if position_evidence is False:
                events.extend(
                    self._handle_leave_candidate(
                        timestamp,
                        min_duration=VISIBLE_AWAY_MIN_DURATION,
                        reason="学生持续远离学习位置",
                    )
                )
            else:
                # 正证据或证据未知，都取消可见人物的离开候选。
                self.leave_start_time = None
                self.leave_evidence_count = 0
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
    def _select_person(
        self,
        persons: List[TrackingResult],
    ) -> Optional[TrackingResult]:
        """
        当前版本默认选择置信度最高的人物。
        后续多人场景可以改成：
            - 指定学习者 track_id
            - 区域过滤
            - 多人独立状态机
        """
        if self.person_track_id is not None:
            current = next(
                (person for person in persons if person.track_id == self.person_track_id),
                None,
            )
            if current is not None:
                return current
            # ID 改变时只允许绑定到上一位置附近的人，避免多人场景串人。
            if self.state != "outside" and self.person_last_bbox is not None:
                nearby = [
                    person for person in persons
                    if self._same_person_area(self.person_last_bbox, person.bbox)
                ]
                if not nearby:
                    return None
                return max(nearby, key=lambda person: person.confidence)
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
            self.person_last_bbox = list(person.bbox)
            self.person_presence_start_time = timestamp
            # 注意：
            # 第一次看到人不能直接认为“坐到学习位置”。
            # 必须通过家具空间关系判断。
            self._maybe_start_sitting(person, results, timestamp)
            return events

        # --------------------------------------------------------
        # 同一个人物
        # --------------------------------------------------------
        if track_id == self.person_track_id:
            self.person_last_seen = timestamp
            self.person_last_bbox = list(person.bbox)
            if self.state == "outside":
                self._maybe_start_sitting(person, results, timestamp)
            return events

        # --------------------------------------------------------
        # Track ID 发生变化
        #
        # 这里通常意味着 Tracker 换了 ID。
        # 当前版本重新绑定人物。
        # --------------------------------------------------------
        self.person_last_seen = timestamp
        self.person_last_bbox = list(person.bbox)
        # Tracker 短暂丢失后可能给同一人物分配新 ID。既然当前帧已选中
        # 新人物，就必须同步重绑，否则后续 Event 会一直携带过期 track_id。
        self.person_track_id = track_id
        if self.state == "outside":
            self.person_presence_start_time = timestamp
            self._maybe_start_sitting(person, results, timestamp)
        return events

    def _maybe_start_sitting(
        self,
        person: TrackingResult,
        results: List[TrackingResult],
        timestamp: float,
    ) -> None:
        evidence = self._study_position_evidence(person, results)
        self.last_position_evidence = evidence
        if self.person_presence_start_time is None:
            self.person_presence_start_time = timestamp
        if evidence is True:
            self.sit_start_time = timestamp
            self.leave_start_time = None
            self.leave_evidence_count = 0
            self.state = "sitting"
        elif evidence is False:
            # 明确远离桌椅，不累计“稳定在学习区域”的兜底时间。
            self.person_presence_start_time = timestamp
        elif timestamp - self.person_presence_start_time >= UNKNOWN_POSITION_ENTRY_MIN_DURATION:
            self.sit_start_time = self.person_presence_start_time
            self.leave_start_time = None
            self.leave_evidence_count = 0
            self.state = "sitting"

    @staticmethod
    def _same_person_area(previous_bbox: List[float], current_bbox: List[float]) -> bool:
        if len(previous_bbox) != 4 or len(current_bbox) != 4:
            return False
        px = (previous_bbox[0] + previous_bbox[2]) / 2.0
        py = (previous_bbox[1] + previous_bbox[3]) / 2.0
        cx = (current_bbox[0] + current_bbox[2]) / 2.0
        cy = (current_bbox[1] + current_bbox[3]) / 2.0
        width = max(previous_bbox[2] - previous_bbox[0], 1.0)
        height = max(previous_bbox[3] - previous_bbox[1], 1.0)
        return abs(cx - px) / width <= 1.5 and abs(cy - py) / height <= 1.5

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
        return self._study_position_evidence(person, results) is True

    def _study_position_evidence(
        self,
        person: TrackingResult,
        results: List[TrackingResult],
    ) -> Optional[bool]:
        """返回 True/False/None：在位置/明确远离/家具证据缺失。"""
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
            return None

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
    ) -> Dict[str, float]:
        """
        收集当前帧的全部平等行为候选。这里仅表达“相关物体出现”，
        不把物体存在直接当成最终动作。
        """
        # --------------------------------------------------------
        # VLM 已经判断
        # --------------------------------------------------------
        normalized_hint = activity_hint
        if normalized_hint in {
            EVENT_READING,
            EVENT_WRITING,
            EVENT_PHONE_USAGE,
            EVENT_COMPUTER_USAGE,
            EVENT_OTHER_BEHAVIOR,
            EVENT_COMMUNICATION_DISTRACTION,
        }:
            return {normalized_hint: 1.0}

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
    ) -> Dict[str, float]:
        """
        根据 YOLO 当前能看到的物体，返回全部基础行为候选。
        注意：
            这里不是最终的行为理解。
        例如：
            book → reading 的弱候选证据
            cell phone → phone_usage 的弱候选证据
            laptop → computer_usage 的弱候选证据
        所有出现的相关物体各贡献一次，不设置类别优先级。最终动作由 VLM
        和跨窗口时序层判断。
        """
        class_names = {
            result.class_name
            for result in results
        }

        candidates: Dict[str, float] = {}
        if "book" in class_names:
            candidates[EVENT_READING] = 1.0

        # --------------------------------------------------------
        # 笔
        #
        # 当前 COCO YOLO 通常无法可靠检测 pen，
        # 如果以后自训练模型增加 pen，可以直接使用。
        # --------------------------------------------------------
        if "pen" in class_names:
            candidates[EVENT_WRITING] = 1.0

        # --------------------------------------------------------
        # 手机
        #
        # 只表示手机出现在画面里，不区分学习或分心。
        # --------------------------------------------------------
        if (
            "cell phone" in class_names
            or "phone" in class_names
            or "mobile phone" in class_names
        ):
            candidates[EVENT_PHONE_USAGE] = 1.0

        # --------------------------------------------------------
        # 电脑
        #
        # 只表示电脑出现在画面里。
        # --------------------------------------------------------
        if (
            "laptop" in class_names
            or "computer" in class_names
            or "keyboard" in class_names
            or "mouse" in class_names
        ):
            candidates[EVENT_COMPUTER_USAGE] = 1.0

        return candidates

    # ============================================================
    # 学习行为状态处理
    # ============================================================
    def _handle_activity(
        self,
        activity: Dict[str, float],
        person: TrackingResult,
        timestamp: float,
    ) -> List[Event]:
        events: List[Event] = []

        # 空候选只表示本帧没有具体物体证据，不能给“其他”投票。
        candidates = activity or {}
        if self.activity_start_time is None:
            self.activity_start_time = timestamp
        self.activity_observation_count += 1
        if not candidates:
            self.activity_unclassified_count += 1
        # current_activity 只供关键帧采集命名；并列时使用中性 other，真正的
        # 多候选证据完整保存在 activity_votes 中。
        top_score = max(candidates.values(), default=0.0)
        top_candidates = [
            name for name, score in candidates.items() if score == top_score
        ]
        self.current_activity = (
            top_candidates[0]
            if len(top_candidates) == 1
            else EVENT_OTHER_BEHAVIOR
        )

        for candidate, score in candidates.items():
            self.activity_votes[candidate] = (
                self.activity_votes.get(candidate, 0.0) + max(0.0, float(score))
            )
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

        # 用全部分析帧作为分母，避免少量偶发物体被“只在有效票之间归一化”
        # 后伪装成接近 100% 的强证据。例如 30 帧中仅 2 帧出现 laptop，
        # 现在显示约 6.7%，而不是 computer_usage=1.0。
        observation_total = self.activity_observation_count
        candidate_scores = {
            name: round(score / observation_total, 4)
            for name, score in self.activity_votes.items()
        } if observation_total > 0 else {}
        top_score = max(candidate_scores.values(), default=0.0)
        top_candidates = [
            name for name, score in candidate_scores.items()
            if abs(score - top_score) < 1e-9
        ]
        # other_behavior 在这里是“等待 VLM 分类”的入口，不是参与投票后
        # 获胜的类别。真正的其他事件只能由后续兜底产生。
        event_type = top_candidates[0] if len(top_candidates) == 1 else EVENT_OTHER_BEHAVIOR
        unclassified_ratio = (
            self.activity_unclassified_count / self.activity_observation_count
            if self.activity_observation_count > 0
            else 1.0
        )
        event = Event(
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            track_id=self.person_track_id,
            confidence=person.confidence if person is not None else 1.0,
            description="待 VLM 识别的学习行为候选",
            candidate_scores=candidate_scores,
            unclassified_ratio=round(unclassified_ratio, 4),
        )
        self.activity_start_time = end_time
        self.current_activity = None
        self.activity_votes = {}
        self.activity_observation_count = 0
        self.activity_unclassified_count = 0
        return event

    # ============================================================
    # 离开学习位置
    # ============================================================
    def _handle_leave_candidate(
        self,
        timestamp: float,
        *,
        min_duration: float = LEAVE_STUDY_POSITION_MIN_DURATION,
        reason: str = "学生离开学习位置",
    ) -> List[Event]:
        events: List[Event] = []

        # --------------------------------------------------------
        # 第一次发现可能离开
        # --------------------------------------------------------
        if self.leave_start_time is None:
            self.leave_start_time = timestamp
            self.leave_evidence_count = 1
        else:
            self.leave_evidence_count += 1

        leave_duration = timestamp - self.leave_start_time

        # --------------------------------------------------------
        # 持续离开达到阈值
        # --------------------------------------------------------
        if (
            leave_duration >= min_duration
            and self.leave_evidence_count >= LEAVE_MIN_OBSERVATIONS
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
                    description=reason,
                )
            )

            self.state = "outside"
            self._reset_learning_state(keep_person=True, reason=reason)

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
        if self.state == "outside":
            self.person_presence_start_time = None
            self.leave_evidence_count = 0
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
            self.leave_evidence_count = 1
        else:
            self.leave_evidence_count += 1

        absence_duration = timestamp - self.leave_start_time
        if (
            absence_duration >= LEAVE_STUDY_POSITION_MIN_DURATION
            and self.leave_evidence_count >= LEAVE_MIN_OBSERVATIONS
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
                self._reset_learning_state(keep_person=True, reason="person_absent")

        return events

    # ============================================================
    # 状态清理
    # ============================================================
    def _reset_learning_state(
        self,
        keep_person: bool = True,
        reason: str = "manual_reset",
    ) -> None:
        self.last_reset_reason = reason
        if not keep_person:
            self.person_track_id = None
            self.person_last_seen = None
            self.person_last_bbox = None
        self.sit_start_time = None
        self.leave_start_time = None
        self.leave_evidence_count = 0
        self.preparation_start_time = None
        self.study_start_time = None
        self.study_started = False
        self.current_activity = None
        self.activity_start_time = None
        self.activity_votes = {}
        self.activity_observation_count = 0
        self.activity_unclassified_count = 0
        self.person_presence_start_time = None

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
        # 若结束时正在等待“人物缺失”确认，动作只能截止到最后可见时刻，
        # 不能把没有人物证据的尾部区间算进上一动作。
        activity_end = (
            min(timestamp, self.leave_start_time)
            if self.leave_start_time is not None
            else timestamp
        )
        activity_event = self._finish_activity_window(activity_end)
        if activity_event is not None:
            events.append(activity_event)

        self._reset_learning_state(keep_person=False, reason="finalize")
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
            EVENT_PHONE_USAGE: "学生正在使用手机",
            EVENT_COMPUTER_USAGE: "学生正在使用电脑",
            EVENT_OTHER_BEHAVIOR: "无法判断人物正在进行的具体行为",
            EVENT_COMMUNICATION_DISTRACTION: "学生正在进行交流分心",
        }
        return descriptions.get(
            activity,
            "学生正在进行学习相关行为",
        )
