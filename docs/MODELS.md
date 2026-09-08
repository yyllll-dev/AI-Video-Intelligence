# VisionOracle AI 模型及来源说明

## 1. 文档目的

本文说明 VisionOracle 当前实际使用的 AI 模型、模型来源、版本与文件、在系统中的职责、加载方式、许可证注意事项及已知限制。本文只描述代码中已经接入的能力，不把规则算法、目标跟踪或哈希检索包装成额外 AI 模型。

## 2. 模型总览

| 模型 | 类型 | 当前用途 | 是否微调 | 是否本地推理 |
|---|---|---|---|---|
| Ultralytics YOLO11n | 目标检测模型 | 检测人物及学习场景相关物体，为事件引擎提供视觉线索 | 否，使用 COCO 预训练权重 | 是 |
| Qwen2-VL-2B-Instruct | 视觉语言模型 | 理解事件关键帧，确认行为类别、边界和转场 | 否，使用官方指令模型 | 是 |

系统没有调用云端模型 API。网络只在本地缺少 Qwen 权重时用于通过 ModelScope 下载模型；权重准备完成后可以离线推理。

## 3. Ultralytics YOLO11n

### 3.1 基本信息

| 项目 | 内容 |
|---|---|
| 模型名称 | YOLO11n Detect |
| 开发者 | Ultralytics |
| 模型任务 | 通用目标检测 |
| 预训练数据 | COCO |
| 项目权重路径 | `models/yolo11n.pt` |
| 当前权重大小 | 5,613,764 字节，约 5.35 MiB |
| 当前文件 SHA-256 | `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1` |
| 项目中使用的库 | `ultralytics==8.4.140` |

Ultralytics 官方 YOLO11 文档将 `yolo11n.pt` 列为目标检测权重。官方 COCO 参考表给出的 YOLO11n 规模约为 2.6M 参数、6.5B FLOPs；这些是官方通用基准信息，不是 VisionOracle 的应用性能测试结果。

### 3.2 在本项目中的职责

YOLO11n 对送入分析链路的视频帧执行目标检测。当前只保留以下 COCO 类别：

- `person`
- `chair`
- `dining table`
- `book`
- `cell phone`
- `laptop`
- `keyboard`
- `mouse`

检测结果包含类别、置信度和边界框，并传给目标跟踪和事件引擎。YOLO 只负责提供物体证据，不直接判断“阅读”“书写”“使用手机”等行为。例如，画面中出现手机只表示存在手机，不等于人物正在持续使用手机。

### 3.3 选择理由

- `n` 版本体积小、加载快，适合 AI PC 和现场演示。
- 能在 CPU 上运行，也可以通过 Ultralytics 使用 NVIDIA GPU。
- COCO 类别已覆盖人物、书本、手机、电脑和桌椅等基础视觉对象。
- 与较大的 YOLO11 版本相比，更适合把计算资源留给视觉语言模型。

### 3.4 来源

- 官方模型文档：<https://docs.ultralytics.com/models/yolo11/>
- Ultralytics 官方仓库：<https://github.com/ultralytics/ultralytics>
- 官方权重资源仓库：<https://github.com/ultralytics/assets/>
- 本项目采用的权重下载地址：<https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt>

### 3.5 许可证注意事项

Ultralytics 官方说明提供 AGPL-3.0 和 Enterprise 两种许可路径。当前项目用于公开参赛、教学演示或后续商业化时，都应根据实际发布方式核对 Ultralytics 最新条款：

- AGPL-3.0 路径通常要求满足相应开源义务。
- 不希望承担 AGPL 开源义务的商业或闭源场景，应向 Ultralytics 核实 Enterprise License。
- 本说明不是法律意见，最终合规结论应以官方许可证原文和实际使用方式为准。

官方许可说明：<https://docs.ultralytics.com/>

## 4. Qwen2-VL-2B-Instruct

### 4.1 基本信息

| 项目 | 内容 |
|---|---|
| 模型名称 | Qwen2-VL-2B-Instruct |
| 开发者 | Qwen Team / Alibaba Cloud |
| 模型类型 | 视觉语言指令模型 |
| ModelScope 标识 | `Qwen/Qwen2-VL-2B-Instruct` |
| 参数规模 | 模型页标注约 2.21B |
| 模型文件规模 | ModelScope 模型页标注约 4.43 GB |
| 项目加载接口 | Hugging Face Transformers |
| 当前相关库 | `transformers==5.16.1`、`modelscope==1.39.1`、`qwen-vl-utils==0.0.14` |

### 4.2 在本项目中的职责

Qwen2-VL 接收事件时间窗中抽取的多张关键帧和约束提示词，完成以下任务：

1. 对 Event Engine 给出的候选事件进行主要语义确认。
2. 判断一个时间窗内实际出现了哪些正式事件。
3. 对疑似行为切换窗口进行逐帧或分段边界复核。
4. 复核人物坐下、离开等位置变化。
5. 定位两种行为之间首次发生切换的帧。
6. 生成受结构化字段约束的客观事件描述。

所有复核入口复用同一个模型和 Processor，不会为每次事件重复加载权重。Web 页面打开后会预热模型；页面显示“模型已就绪”后才允许开始分析。

### 4.3 明确不由 Qwen2-VL 完成的部分

- 视频读取、时间戳生成和摄像头录像不由 Qwen 完成。
- YOLO 目标检测和目标跟踪不由 Qwen 完成。
- 当前全事件总结默认根据已确认事件按确定性规则组织，不默认调用 Qwen 自由生成。
- 自然语言搜索当前通过事件词典和本地哈希向量完成，不是开放域 Qwen 视频问答。
- Qwen 输出不能替代人工审核，也不能作为高风险教育评价的唯一依据。

### 4.4 选择理由

- 2B 版本相较 7B、72B 权重更小，更符合本地 AI PC 部署约束。
- 支持多图、视频相关视觉理解和中文指令。
- 官方采用动态分辨率与多模态位置编码设计，适合不同分辨率的关键帧输入。
- Apache-2.0 许可路径相对清晰，便于公开研究与演示，但仍需遵循模型卡和依赖许可证。

### 4.5 来源

- ModelScope 官方模型页：<https://www.modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct/summary>
- Qwen2-VL 官方介绍：<https://qwenlm.github.io/blog/qwen2-vl/>
- Qwen2-VL 论文：<https://arxiv.org/abs/2409.12191>
- 论文给出的代码地址：<https://github.com/QwenLM/Qwen2-VL>

### 4.6 下载与加载方式

推荐把 Qwen 权重放在项目目录之外：

```powershell
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2-VL-2B-Instruct', local_dir=r'D:\AIModels\Qwen2-VL-2B-Instruct'))"
$env:QWEN_VL_MODEL_PATH = "D:\AIModels\Qwen2-VL-2B-Instruct"
```

代码读取顺序如下：

1. 优先使用函数参数或环境变量 `QWEN_VL_MODEL_PATH`。
2. 未配置本地目录时，调用 ModelScope 的 `snapshot_download`。
3. 使用 `Qwen2VLForConditionalGeneration.from_pretrained` 和 `AutoProcessor.from_pretrained` 加载。
4. 使用 `device_map="auto"` 让 Transformers 根据当前 PyTorch 环境分配设备。
5. 使用进程级缓存复用已经加载的模型实例。

### 4.7 许可证

Qwen 官方介绍和 ModelScope 模型页均将开源的 Qwen2-VL-2B 标注为 Apache License 2.0。再分发权重或构建公开产品前，应同时检查：

- 模型页的最新许可证与模型卡。
- Transformers、ModelScope、PyTorch 等运行依赖的许可证。
- 输入视频、数据集和输出内容是否具有合法使用授权。

## 5. 非模型组件说明

为避免模型数量和 AI 能力表述失真，以下组件不计作第三个 AI 模型：

| 组件 | 实现 | 作用 |
|---|---|---|
| 目标跟踪 | `SimpleTracker` | 根据检测框和时间关联连续帧中的目标 |
| 事件引擎 | `EventEngine` | 根据目标、持续时间和状态机产生候选事件 |
| 检索向量 | `HashingEmbedder` | 将中文字符和相邻字符稳定映射为 256 维本地向量 |
| 查询理解 | 事件名称与中文别名词典 | 将“手机”等明确查询映射到 `phone_usage` |
| 全事件总结 | 结构化规则生成 | 按已确认事件顺序生成客观过程描述 |
| 扇形图 | 时长统计 | 按事件实际起止时间计算有效时长和占比 |

`HashingEmbedder` 是确定性检索实现，不具备通用语义理解能力。它的优点是完全离线、无额外模型、结果稳定；局限是对词典外表达和复杂语义的泛化能力较弱。

## 6. 模型数据流与责任边界

```text
视频帧
  ↓
YOLO11n：检测人物和物体
  ↓
SimpleTracker：关联目标
  ↓
EventEngine：产生弱候选与时间窗
  ↓
关键帧提取：固化事件证据
  ↓
Qwen2-VL：确认事件、边界和客观描述
  ↓
结构化事件记忆：总结、统计、检索和回放
```

这种组合把高频、轻量的目标检测与低频、计算量较大的视觉语言判断分开。摄像头采集和录像也与语义分析解耦，避免 Qwen 推理速度直接改变录像时长。

## 7. 复现与完整性检查

### 7.1 YOLO 权重校验

在 PowerShell 中执行：

```powershell
Get-FileHash .\models\yolo11n.pt -Algorithm SHA256
```

当前提交文件应得到：

```text
0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1
```

### 7.2 Qwen 模型校验

Qwen 权重未直接放入项目压缩包，原因是体积较大。提交材料应保留：

- 模型标识 `Qwen/Qwen2-VL-2B-Instruct`。
- 实际下载来源。
- 本地模型目录或缓存位置。
- ModelScope 下载生成的文件清单及版本信息。
- 性能测试时使用的 PyTorch、Transformers 和 CUDA 版本。

## 8. 声明

本文记录的是当前代码实际模型配置，不等同于对模型准确性、安全性或特定用途适用性的保证。VisionOracle 输出应作为视频回顾辅助信息，由用户结合原始视频进行人工确认。