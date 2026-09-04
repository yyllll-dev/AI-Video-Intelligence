"""
prompt.py —— Qwen-VL Prompt 模板

B模块职责：
    A / EventEngine：
        生成位置、流程事件和宽松活动候选。

    B / Qwen-VL：
        观察活动候选对应的连续视频帧，
        允许纠正上游候选类型，并识别一个窗口内的多个动作。

重要原则：
    位置和流程事件由规则层负责；活动候选类型只是提示，
    Qwen-VL 才是阅读、书写、手机、电脑和交流等活动的最终分类器。

正式流程：
    Event(event_type)
        ↓
    Qwen-VL观察连续视频帧
        ↓
    分析真实视觉动作
        ↓
    判断画面是否支持该事件
        ↓
    生成 description
        ↓
    返回补全后的 Event
"""


# ============================================================
# 14种事件
# ============================================================

EVENT_TYPES = [
    "sit_at_study_position",
    "leave_study_position",

    "study_preparation",
    "start_study",
    "end_study",

    "reading",
    "writing",
    "phone_learning",
    "computer_learning",
    "other_study_behavior",

    "phone_distraction",
    "computer_distraction",
    "communication_distraction",

    "study_end_cleanup",
]


# ============================================================
# 事件中文名称
# ============================================================

EVENT_TYPE_CN = {
    "sit_at_study_position": "坐到学习位置",
    "leave_study_position": "离开学习位置",

    "study_preparation": "学习准备",
    "start_study": "开始学习",
    "end_study": "结束学习",

    "reading": "阅读",
    "writing": "书写",
    "phone_learning": "使用手机学习",
    "computer_learning": "使用电脑学习",
    "other_study_behavior": "其他学习行为",

    "phone_distraction": "手机分心",
    "computer_distraction": "电脑分心",
    "communication_distraction": "交流分心",

    "study_end_cleanup": "学习结束整理",
}


# ============================================================
# 各事件的重点观察方向
# ============================================================

EVENT_FOCUS = {
    "sit_at_study_position":
        "重点观察人物是否从站立、走动或其他位置移动到学习位置并坐下。",

    "leave_study_position":
        "重点观察人物是否从学习位置离开，例如起身、走开或离开桌面区域。",

    "study_preparation":
        "重点观察人物是否整理桌面、拿取学习用品、摆放书本、准备电脑或进行其他学习前准备动作。",

    "start_study":
        "重点观察人物是否从准备状态进入实际学习行为，例如开始阅读、书写、做题或查看学习资料。",

    "end_study":
        "重点观察人物是否停止当前学习行为并进入学习结束状态。",

    "reading":
        "重点观察人物是否正在阅读书籍、教材、纸质资料或其他学习材料，以及是否发生翻页、查看文字等动作。",

    "writing":
        "重点观察人物是否正在书写、做题、记笔记或进行其他需要书写的学习动作。",

    "phone_learning":
        "重点观察人物是否明确使用手机进行学习，例如查看学习资料、课程内容、题目或其他明确学习内容。",

    "computer_learning":
        "重点观察人物是否明确使用电脑进行学习，例如查看课程、学习资料、编程、做题或其他明确学习活动。",

    "other_study_behavior":
        "重点观察人物是否正在进行明确的学习活动，但该活动无法归入阅读、书写、手机学习或电脑学习。",

    "phone_distraction":
        "重点观察人物是否停止或偏离学习行为，转而使用手机进行与学习无关的活动。",

    "computer_distraction":
        "重点观察人物是否停止或偏离学习行为，转而使用电脑进行与学习无关的活动。",

    "communication_distraction":
        "重点观察人物是否与其他人进行交流、交谈或互动，并因此偏离当前学习行为。",

    "study_end_cleanup":
        "重点观察人物结束学习后是否整理书本、文具、桌面或其他学习用品。",
}


# ============================================================
# 正式 Prompt
# ============================================================

def build_caption_prompt(event_type: str) -> str:
    """
    正式流程使用。

    event_type：
        由A的EventEngine传入。
        Qwen-VL不重新选择事件类型。

    返回：
        针对指定event_type构造的视觉分析Prompt。
    """

    if event_type not in EVENT_TYPE_CN:
        raise ValueError(
            f"Unknown event_type: {event_type!r}. "
            f"Allowed events: {EVENT_TYPES}"
        )

    event_cn = EVENT_TYPE_CN[event_type]
    event_focus = EVENT_FOCUS[event_type]

    return f"""
【任务】看视频帧，描述人物动作，并重点确认一个候选事件，输出JSON。

===== 【本次唯一待确认事件】 =====
- event_type：{event_type}
- 中文名称：{event_cn}
- 观察重点：{event_focus}
- event_confirmed 只表示 {event_type} 是否成立，不能对应其他事件。

===== 【第一步：描述动作】 =====
按时间顺序描述人物的动作，像讲故事一样自然地说出来。
要求：
- 只说实际看到的内容，不编造，不猜测
- 用逗号连接多个动作，句号结尾
- 像正常说话那样写，不用"观察到""发现"等词

===== 【第二步：判断14件事】 =====
根据动作描述，对照规则判断每件事。

事件判断规则：

位置类：
- sit_at_study_position：描述里有"走到桌前""坐下""入座" → true，否则false
- leave_study_position：描述里有"起身""离开桌子""走开" → true，否则false

学习状态类：
- study_preparation：描述里有"整理桌面""摆放学习用品" → true，否则false
- start_study：描述里明确有"开始"动作（如开始看书、开始写字）→ true，否则false
- end_study：描述里有"合上书""收拾东西""停止学习" → true，否则false

学习行为类：
- reading：描述里有"拿书""翻页""看书""看资料" → true，否则false
- writing：描述里有"拿笔""写字""记笔记""做题""画图" → true，否则false
- phone_learning：描述里明确说了手机内容是"学习资料""课程视频""题目""背单词" → true
- computer_learning：描述里明确说了电脑内容是"学习资料""课程""编程" → true
- other_study_behavior：描述里有其他明确属于学习的行为 → true

分心类（与学习无关）：
- phone_distraction：描述里说"用手机刷视频""用手机聊天""用手机玩游戏" → true
- computer_distraction：描述里说"用电脑打游戏""用电脑刷网页" → true
- communication_distraction：描述里有"与人说话""视频通话""讨论" → true

收尾类：
- study_end_cleanup：描述里有"收拾书本""整理桌面""收文具" → true

===== 【phone_learning vs phone_distraction 核心规则】 =====
手机分心优先级最高！判断顺序：
1. 描述里明确说手机内容是学习资料/课程/题目 → phone_learning=true，phone_distraction=false
2. 描述里说手机内容是娱乐/聊天/游戏 → phone_distraction=true，phone_learning=false
3. 描述里**只说在用手机，没说手机内容是什么** → 两者都为false，不猜测用途
4. phone_learning 和 phone_distraction 互斥，最多一个true

===== 【重要】 =====
- event_confirmed 必须与 events.【你当前判断的事件】的值完全一致，不一致则以 events 为准
- 描述必须是实际看到的动作，不凭空猜测
- 只有描述里明确出现某事件的特征，该事件才能判true
- 没出现特征的事件一律判false
- 不要把14件事全部判true
- 描述和判断要一致：描述里没有的动作，对应事件一定是false
- 描述里没有"开始"，start_study就是false；没有"离开"，leave_study_position就是false

===== 【输出格式】 =====
{{
    "description": "<自然流畅的动作描述>",
    "events": {{
        "sit_at_study_position": true或false,
        "leave_study_position": true或false,
        "study_preparation": true或false,
        "start_study": true或false,
        "end_study": true或false,
        "reading": true或false,
        "writing": true或false,
        "phone_learning": true或false,
        "computer_learning": true或false,
        "other_study_behavior": true或false,
        "phone_distraction": true或false,
        "computer_distraction": true或false,
        "communication_distraction": true或false,
        "study_end_cleanup": true或false
    }},
    "event_confirmed": true或false（必须与events.{event_type}保持一致）,
    "is_phone_usage": true或false,
    "is_studying": true或false
}}

只输出JSON，不要输出其他内容。
"""


# ============================================================
# 批量模式 Prompt：一次性判断全部14件事
# ============================================================

def build_all_events_prompt() -> str:
    """
    【批量测试专用】

    不指定单一event_type，
    让VLM一次性看完所有视频帧后，
    自主判断14件事哪些发生、哪些没发生。

    输出JSON格式：
    {
        "description": "<人物动作描述>",
        "events": {
            "sit_at_study_position": bool,
            ... 14个事件
        },
        "is_phone_usage": bool,
        "is_studying": bool
    }

    与正式Pipeline的 build_caption_prompt(event_type) 区别：
        正式流程：上游A已经确定event_type，VLM只需要验证
        批量模式：VLM自主判断所有事件的true/false
    """

    return """
【任务】看视频帧，描述人物动作，判断14件事是否发生，输出JSON。

===== 【第一步：描述动作】 =====
按时间顺序描述人物的动作，像讲故事一样自然地说出来。
要求：
- 只说你实际看到的内容，不编造，不猜测
- 用逗号连接多个动作，句号结尾
- 像正常说话那样写，不要用"观察到""发现"这类词

===== 【第二步：判断14件事】 =====
根据动作描述，对照规则判断每件事。

事件判断规则：

位置类：
- sit_at_study_position：描述里有"走到桌前""坐下""入座" → true，否则false
- leave_study_position：描述里有"起身""离开桌子""走开" → true，否则false

学习状态类：
- study_preparation：描述里有"整理桌面""摆放学习用品" → true，否则false
- start_study：描述里明确有"开始"动作（如开始看书、开始写字）→ true，否则false
- end_study：描述里有"合上书""收拾东西""停止学习" → true，否则false

学习行为类：
- reading：描述里有"拿书""翻页""看书""看资料" → true，否则false
- writing：描述里有"拿笔""写字""记笔记""做题""画图" → true，否则false
- phone_learning：描述里明确说了手机内容是"学习资料""课程视频""题目""背单词" → true
- computer_learning：描述里明确说了电脑内容是"学习资料""课程""编程" → true
- other_study_behavior：描述里有其他明确属于学习的行为 → true

分心类（与学习无关）：
- phone_distraction：描述里说"用手机刷视频""用手机聊天""用手机玩游戏" → true
- computer_distraction：描述里说"用电脑打游戏""用电脑刷网页" → true
- communication_distraction：描述里有"与人说话""视频通话""讨论" → true

收尾类：
- study_end_cleanup：描述里有"收拾书本""整理桌面""收文具" → true

===== 【phone_learning vs phone_distraction 核心规则】 =====
手机分心的优先级最高！判断顺序：
1. 如果描述里明确说手机内容是学习资料/课程/题目 → phone_learning = true，phone_distraction = false
2. 如果描述里说手机内容是娱乐/聊天/游戏 → phone_distraction = true，phone_learning = false
3. 如果描述里**只说在用手机，没说手机内容是什么** → phone_distraction = true，phone_learning = false
4. phone_learning 和 phone_distraction **互斥**，两者最多一个为true

===== 【重要】 =====
- 如果要判断具体事件类型，event_confirmed 必须与 events.【你判断的事件】的值完全一致，不一致则以 events 为准
- 描述必须是实际看到的动作，不凭空猜测
- 只有描述里明确出现某事件的特征，该事件才能判true
- 没出现特征的事件一律判false
- 不要把14件事全部判true
- 描述和判断要一致：描述里没有的动作，对应事件一定是false
- 描述里没有"开始"，start_study就是false；没有"离开"，leave_study_position就是false

===== 【输出格式】 =====
{
    "description": "<自然流畅的动作描述>",
    "events": {
        "sit_at_study_position": true或false,
        "leave_study_position": true或false,
        "study_preparation": true或false,
        "start_study": true或false,
        "end_study": true或false,
        "reading": true或false,
        "writing": true或false,
        "phone_learning": true或false,
        "computer_learning": true或false,
        "other_study_behavior": true或false,
        "phone_distraction": true或false,
        "computer_distraction": true或false,
        "communication_distraction": true或false,
        "study_end_cleanup": true或false
    },
    "is_phone_usage": true或false,
    "is_studying": true或false
}

只输出JSON，不要输出其他内容。
"""


def build_activity_prompt(candidate_event_type: str) -> str:
    """宽松活动候选使用：候选仅作提示，VLM 负责最终分类。"""
    return f"""
【任务】观察按时间顺序给出的全部视频帧，识别人物实际发生的学习或分心动作。

上游候选类型是 {candidate_event_type}，它只是 YOLO/规则提供的提示，可能完全错误。
你必须根据画面自行修改类型，不要为了迎合候选而确认错误事件。

可选择的动作类型只有：
- reading：阅读书本、教材或纸质资料
- writing：写字、做题、记笔记
- phone_learning：明确用手机查看学习资料、课程或题目
- computer_learning：明确用电脑学习、编程、做题或看课程
- other_study_behavior：其他明确学习行为
- phone_distraction：使用手机，但没有明确学习证据，或用于娱乐聊天
- computer_distraction：使用电脑，但没有明确学习证据，或用于娱乐
- communication_distraction：与他人说话、通话或交流
- none：没有可确认的上述动作

要求：
1. 先按时间顺序客观描述动作，不猜测画面外的信息。
2. observed_activities 按发生顺序列出，可以有多个；没有则填空列表。
3. activity_segments 给出每个动作对应的起止帧编号，编号从 1 开始。
4. 手机或电脑用途不明确时，归入 distraction，不得猜成学习。
5. 不要输出位置和流程事件，例如坐下、离开、学习准备、开始学习。
6. primary_event 只填持续时间最长或最主要的一个动作；没有则填 none。
7. 不要把所有类型都列出，只列画面中有明确证据的动作。
8. description 必须写出画面中的人物、物品和动作，禁止照抄输出格式里的占位文字。
9. 输出前先逐帧检查一次，确保每个活动和帧编号都能在对应图片中看到。

只输出合法 JSON，不要输出 Markdown：
{{
  "description": "<必须填写实际看到的动作，不能保留尖括号内容>",
  "primary_event": "<填写一个动作类型或none>",
  "observed_activities": ["<按发生顺序填写动作类型；没有则使用空列表>"],
  "activity_segments": [
    {{"event_type": "<动作类型>", "start_frame": 1, "end_frame": 1}}
  ],
  "event_confirmed": true或false,
  "is_phone_usage": true或false,
  "is_studying": true或false
}}
"""


# ============================================================
# 独立测试 Prompt
# ============================================================

CLASSIFY_PROMPT = """
你是一个学习场景视频行为分析助手。

你将看到按照时间顺序排列的连续视频帧。

请真正观察所有视频帧，
分析人物从第1帧到最后一帧发生的具体视觉动作变化。

任务：

1. 用一句话描述人物的真实动作变化。
2. 判断人物是否明确使用手机。
3. 判断人物是否明确处于学习状态。

要求：

- 必须基于真实画面。
- 必须观察所有视频帧。
- 必须按照时间顺序描述动作。
- 不允许虚构动作。
- 不允许根据常识补充动作。
- 不允许猜测人物意图、心理或计划。
- 禁止使用"似乎""可能""准备""打算""思考"等推测性表达。
- 只能描述能够从画面直接观察到的内容。

只输出合法JSON。

不要输出Markdown。
不要输出代码块。
不要输出任何额外解释。

格式：

{
    "description": "具体真实动作描述",
    "is_phone_usage": false,
    "is_studying": false
}
"""
