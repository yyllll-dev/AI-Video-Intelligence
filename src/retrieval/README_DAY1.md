# 高嘉沐 Day 1 提交说明

## 1. 今日任务

按照任务书和队长接口，完成 `Video Memory + Embedding + Retrieval` 的第一天框架。
本次只修改本人负责的 `src/retrieval/`，没有修改 `src/event/`、`src/pipeline/`、
`src/vlm/`、`src/detection/`、`src/tracking/` 或 `demo/`。

## 2. 已完成内容

1. 定义 `EventLike` 接口，兼容队长提供的 `Event`：
   `event_type`、`start_time`、`end_time`、`track_id`、`confidence`、`description`。
2. 定义完整 `MemoryRecord`：事件、时间、Caption、截图、视频片段、Embedding等字段。
3. 定义 `MemoryStore` 接口，并提供Day 1可运行的 `InMemoryStore`。
4. 定义 `EmbeddingProvider` 接口，并提供无第三方依赖的Day 1占位实现。
5. 定义 `QueryProcessor` 接口，为后续Qwen查询改写预留位置。
6. 实现 `VideoMemoryService`，跑通“事件保存 -> 向量生成 -> 自然语言检索”。
7. 提供独立Demo和4项自动测试。

## 3. 接口兼容说明

队长当前 `Event` 使用 `description`，任务书后续又使用 `caption`。
本模块接收 `description`，保存时统一写入 `MemoryRecord.caption`；Day 4接入Qwen-VL后，
也可以通过 `caption=` 参数覆盖为更完整的VLM描述。

`timestamp` 当前统一取 `start_time`，同时完整保存 `start_time` 和 `end_time`。

## 4. VS Code运行方法

在VS Code中打开项目根目录，打开终端后依次运行：

```powershell
conda activate multimodal
python --version
python -m src.retrieval.demo_day1
python -m unittest discover -s src/retrieval/tests -v
```

Python版本应为 `3.11.x`。

Demo预期核心输出：

```text
用户问题：什么时候使用了手机？
检索事件：phone_usage
事件时间：632.5s
事件描述：学生暂停学习并使用手机
视频片段：data/clips/event_003.mp4
```

测试预期显示4项测试全部为 `ok`，最后显示 `OK`。

## 5. 交给A、B、E的接口

### A/B写入事件

调用 `VideoMemoryService.remember_event(event, caption=..., video_path=...)`。

### A/E进行查询

调用 `VideoMemoryService.search(query, top_k=3)`。

返回的每个 `SearchResult` 包含：

- `record.event_type`
- `record.timestamp`
- `record.start_time`
- `record.end_time`
- `record.caption`
- `record.screenshot_path`
- `record.video_path`
- `similarity_score`

也可以调用 `SearchResult.to_dict()`，直接得到便于UI或JSON传递的字典。

## 6. 后续替换点

- Day 2：新增 `SQLiteMemoryStore`，替换当前退出即丢失的内存存储。
- Day 3：实现真实视频片段保存和截图保存。
- Day 4：用正式中文Embedding模型替换 `HashingEmbedder`。
- Day 5：接入正式向量索引并确定Top-K和阈值。
- Day 6：与A的Event、B的Caption和E的UI正式联调。

`HashingEmbedder` 只是为了Day 1零依赖验证接口，不应在最终版本中冒充正式语义模型。
