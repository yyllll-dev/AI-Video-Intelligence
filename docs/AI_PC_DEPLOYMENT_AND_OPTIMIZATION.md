# VisionOracle AI PC 本地部署及优化方案说明

## 1. 方案目标

本方案用于把 VisionOracle 的视频检测、事件判断、多模态理解、统计、检索与回放完整部署在 AI PC 本地。核心目标是：

1. 原始视频不依赖云端推理服务，模型准备完成后可离线分析。
2. 让轻量检测模型运行在适合的硬件加速后端，把视觉语言模型与逐帧检测解耦。
3. 上传视频和实时摄像头共用同一事件协议、时间轴和结果界面。
4. 所有性能结论均能追溯到真实设备和原始测试文件，不把未使用的设备包装成推理资源。

## 2. 已实现的本地架构

```text
上传视频 / 实时摄像头
        ↓
统一帧流、采样与真实时间戳
        ↓
YOLO11n + OpenVINO FP16（Intel GPU）
        ↓
SimpleTracker + Event Engine 候选事件
        ↓
关键帧缓冲与时序证据
        ↓
Qwen2-VL-2B-Instruct（本次测试为 CPU）
        ↓
事件记忆、客观总结、有效时长、自然语言检索和本地回放
```

系统不把每一帧都交给 VLM。YOLO 负责高频物体线索，规则和跟踪负责产生候选时间窗，Qwen2-VL 只处理被筛选出的关键帧与边界证据。两类模型在同一 Python 进程中缓存复用，避免对每个事件重复加载权重。

## 3. 模型、框架与精度

| 组件 | 版本/模型 | 推理框架 | 精度 | 本地职责 |
|---|---|---|---|---|
| 目标检测 | Ultralytics YOLO11n | Ultralytics + OpenVINO | FP16（Intel 路径） | 人物、书本、手机、电脑、桌椅等视觉线索 |
| 视觉语言模型 | Qwen2-VL-2B-Instruct | PyTorch + Transformers | bfloat16（实测） | 候选事件确认、位置与转场复核、结构化描述 |
| 目标跟踪 | `SimpleTracker` | 项目本地代码 | 不适用 | 跨帧关联目标 |
| 事件引擎 | `EventEngine` | 项目本地规则与状态机 | 不适用 | 产生候选事件与时间边界 |
| 检索 | 事件词典 + `HashingEmbedder` | 项目本地代码 | 不适用 | 明确事件词搜索与本地结果排序 |

模型来源、许可证和文件校验信息见 `docs/MODELS.md`。系统没有把规则、哈希检索或图表统计包装成额外 AI 模型。

## 4. Intel AI PC 部署步骤

### 4.1 获取代码并创建环境

```powershell
git clone <项目仓库地址>
cd AI-Video-Intelligence
conda create -n visionoracle python=3.11 -y
conda activate visionoracle
python -m pip install --upgrade pip
python -m pip install -r requirements-intel.txt
```

`requirements-intel.txt` 在普通依赖基础上增加 OpenVINO，用于 Intel GPU 的 YOLO 推理。NVIDIA 或纯 CPU 环境仍按 `requirements.txt` 和原有设备参数运行，Intel 路径不会改写上传视频或摄像头的业务逻辑。

### 4.2 准备模型

项目使用 `models/yolo11n.pt` 作为源权重。在 Intel AI PC 上导出 FP16 OpenVINO IR，并完成真实推理检查：

```powershell
python tools/intel/prepare_yolo_openvino.py --device intel:gpu --precision fp16
```

只有看到以下提示，才把 YOLO 设备记录为 Intel GPU：

```text
PASS: inference completed on the requested Intel device
```

提前下载 Qwen 权重并设置本地路径：

```powershell
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2-VL-2B-Instruct', local_dir=r'D:\AIModels\Qwen2-VL-2B-Instruct'))"
$env:QWEN_VL_MODEL_PATH = "D:\AIModels\Qwen2-VL-2B-Instruct"
```

### 4.3 启动与检查

先执行只读环境检查：

```powershell
.\run.ps1 -CheckOnly -QwenModelPath "D:\AIModels\Qwen2-VL-2B-Instruct" -YoloDevice intel:gpu
```

再启动 Web 应用：

```powershell
.\run.ps1 -QwenModelPath "D:\AIModels\Qwen2-VL-2B-Instruct" -YoloDevice intel:gpu
```

访问 `http://127.0.0.1:7860`。页面启动后先加载并缓存模型，显示“模型已就绪”后才允许开始分析；模型加载阶段不会提前录制摄像头。

### 4.4 设备降级策略

- Intel GPU/OpenVINO 检查通过：使用 `intel:gpu`。
- Intel GPU 驱动或 OpenVINO 插件不可用：明确切换到 `cpu`，不得把 CPU 回退结果写成 GPU 测试。
- NVIDIA 环境：继续使用 `0` 或 `cuda:0`，不经过 OpenVINO 导出目录。
- Qwen 设备由 PyTorch/Transformers 实际能力决定，并与 YOLO 设备分别记录。

## 5. 已落地的优化

### 5.1 轻量模型与异构执行

- 使用 YOLO11n 而不是更大检测模型，降低逐帧检测负担。
- Intel 路径把 YOLO 导出为 OpenVINO IR，以 FP16 在 Intel GPU 上执行。
- Qwen2-VL 使用 2B 版本，在保证多图与中文理解能力的同时控制本地权重与内存规模。

### 5.2 分层采样

- 源视频可保持 24 FPS 播放与归档，分析链路按 4 FPS 采样。
- 缓冲区按 4 FPS 保存时序证据，每次向 VLM 组织 9 张关键帧。
- 通过“检测 → 跟踪/规则 → 候选事件 → VLM 复核”减少无意义的大模型调用。

### 5.3 模型复用与生命周期管理

- Web 页面打开后预热模型。
- YOLO、Qwen 模型和 Processor 使用进程级缓存，上传视频与摄像头复用同一实例。
- 模型加载时间与端到端业务时间分开记录，便于区分冷启动和稳定运行性能。

### 5.4 录像与分析解耦

- 摄像头录像从第一帧实际写入时开始建立时间轴。
- 录像按实际采集帧率编码，避免分析速度慢导致成片时长缩水。
- 点击停止后先终止采集并合并完整 MP4，剩余事件语义分析可继续收尾。
- 上传视频和摄像头分析使用统一事件时间范围，保证事件回放与源视频对齐。

### 5.5 本地数据闭环

关键帧、录像、回放、结构化事件和搜索结果均生成在本机。模型准备完成后，核心分析不依赖云端 API；运行数据默认被 `.gitignore` 排除，避免个人视频误提交到公开仓库。

## 6. Intel AI PC 实测基线

在 Intel Core i9-14900HX、31.71 GiB 内存、Windows 11 的测试机上，YOLO 使用 OpenVINO FP16 `intel:gpu`，Qwen 使用 CPU。15.042 秒、1280×720、24 FPS 视频的一次完整运行结果如下：

| 指标 | 实测值 |
|---|---:|
| E2E | 1194.002 秒 |
| 视频处理 FPS | 0.302 FPS |
| Pipeline 分析吞吐 | 0.051 帧/秒 |
| YOLO 平均 / 中位数 / P95 | 25.165 / 17.067 / 21.945 毫秒 |
| 系统 CPU 平均 / 峰值 | 73.658% / 96.700% |
| 进程内存平均 / 峰值 | 5.102 / 5.430 GiB |
| 事件数 / 错误数 | 6 / 0 |

完整口径、原始证据与限制见 `docs/PERFORMANCE_TEST.md`。

## 7. 后续优化路线

以下内容是工程方案，不是已经取得的性能提升，实施后必须重新以同一视频、同一参数测量。

| 优先级 | 优化项 | 实施方式 | 验证指标 |
|---|---|---|---|
| P0 | 增加 VLM 分阶段计时 | 在主判断、位置、窗口标签和转场调用外层统一计时 | VLM 平均、P50、P95、各阶段耗时占比 |
| P0 | 降低重复 VLM 调用 | 基于相邻窗口相似度、规则置信度和事件状态缓存跳过重复复核 | VLM 调用数、E2E、事件一致性 |
| P1 | 验证 Qwen 加速后端 | 在支持的 Intel 平台上评估 OpenVINO/ONNX 或厂商支持路径 | VLM 延迟、内存、准确性回归 |
| P1 | 增加 Intel GPU 遥测 | 采集 GPU 利用率、显存和明确的编译设备全名 | GPU 利用率、显存、设备证据 |
| P1 | 固定基准环境 | 接通电源并使用最佳性能模式，关闭高负载后台程序 | 多轮 E2E 极差与变异系数 |
| P2 | 评估 YOLO INT8 | 使用代表性校准集量化，保持检测结果回归门槛 | YOLO 延迟、召回率、事件一致性 |
| P2 | 长视频稳定性 | 用固定长视频重复运行并监控内存趋势 | 错误率、内存增长、长时间吞吐 |

优化验收必须同时检查性能和事件输出，不能通过关闭 VLM、降低关键功能或更换简单视频来声称同条件提升。

## 8. 风险控制与回退

| 风险 | 控制措施 |
|---|---|
| OpenVINO 未真正使用 Intel GPU | 导出脚本必须完成指定设备推理；结果同时记录后端和可用设备 |
| Qwen 首次下载失败 | 比赛前在稳定网络环境下载到独立目录，并设置 `QWEN_VL_MODEL_PATH` |
| ModelScope SSL 或网络问题 | 使用已校验的本地模型目录离线运行，不在比赛现场临时下载 |
| 视频与事件时间轴偏离 | 源 FPS/实际采集 FPS 驱动录像与事件时间，不用分析速度代替真实时间 |
| 运行数据泄露 | `data/`、日志和本地模型缓存不提交 Git，演示后按授权要求清理 |
| 模型误判 | 时间线与搜索结果始终提供原视频回放，保留人工复核权 |

## 9. NPU 与设备声明

本次测试机不声明 NPU 参与推理。原始采集器曾把 `Microsoft Input Configuration Device` 中的 `INPUT` 误匹配为 `NPU`，该问题已通过严格边界匹配修正。比赛材料只说明真实参与推理的设备：YOLO 为 OpenVINO Intel GPU 路径，Qwen 为 CPU。

这符合“核心视频分析与 AI 推理在 Intel AI PC 本地运行”的部署方向，同时避免把未使用或未验证的 NPU 写入性能成果。

## 10. 交付与验收清单

- 项目代码、`requirements.txt`、`requirements-intel.txt`、`run.ps1` 齐全。
- YOLO11n、Qwen2-VL-2B-Instruct 的来源、精度、职责和许可证已说明。
- Intel GPU 导出与推理检查脚本可复现。
- 性能原始 JSON、CSV、环境快照、摘要和哈希文件完整保留。
- 正式性能结果至少覆盖比赛要求中的四项指标，当前覆盖六项。
- 不提交测试视频、摄像头录像、关键帧、事件回放、调试日志和测试代码。
- 最终项目说明书明确区分“已实现优化”“真实测试结果”和“后续优化建议”。
