# VisionOracle Intel AI PC 性能测试结果

## 1. 测试结论

VisionOracle 已在一台搭载 Intel Core i9-14900HX 与 Intel UHD Graphics 的 Windows 11 笔记本上完成一次端到端本地测试。固定测试视频为 1280×720、24 FPS、15.042 秒；YOLO11n 通过 OpenVINO FP16 路径运行，Qwen2-VL-2B-Instruct 通过 PyTorch/Transformers 在 CPU 上运行。

本次完整 Pipeline 成功结束，产生 6 个事件和 6 条事件记忆，错误数为 0。端到端耗时 1194.002 秒，视频处理速度为 0.302 FPS，YOLO 单次推理平均延迟为 25.165 毫秒。测试真实覆盖比赛要求中的 6 项指标：视频处理 FPS、End-to-End Latency、单模型推理 Latency、视频 Pipeline Throughput、CPU 利用率和内存占用。

结果同时表明：YOLO 的 Intel GPU/OpenVINO 路径延迟较低，而完整系统处理 15 秒视频仍需约 19 分 54 秒。结合 Qwen 在 CPU 上运行且共调用 16 次，可以判断当前主要优化方向是视觉语言模型推理链路；由于脚本尚未记录每次 VLM 延迟，该结论是基于设备分配、调用次数与端到端耗时作出的工程判断，不把它表述为已经精确分解的耗时占比。

## 2. 测试范围与证据

| 项目 | 内容 |
|---|---|
| 测试状态 | 成功 |
| 测试协议 | 模型先加载，再计一次完整视频 Pipeline；不含整段视频预热 |
| 测试开始时间 | 2026-09-08 19:45:26 UTC |
| 测试完成时间 | 2026-09-08 20:05:25 UTC |
| 项目 Git 提交 | `c608aae52dad81ecdbee5e7fcda14e6291396054` |
| 原始结果目录 | `docs/performance/intel_ai_pc_20260909_034520/` |
| 权威结果源 | `performance_results.json` |
| 表格结果源 | `performance_runs.csv` |
| 环境快照 | `environment.json` |
| 测试摘要 | `test_notes.md` |

原始证据文件保留测试机当时的输出，不为美化报告而改写。`hashes.sha256` 用于校验四份原始文件的完整性。

## 3. Intel AI PC 测试环境

| 类别 | 实际配置 |
|---|---|
| CPU | Intel Core i9-14900HX，24 个物理核心、32 个逻辑线程 |
| GPU | Intel UHD Graphics；机器同时存在 NVIDIA GeForce RTX 4060 Laptop GPU |
| NPU | 本次不声明存在或使用 NPU；原始采集结果属于字符串误匹配，详见 9.2 |
| 系统内存 | 31.71 GiB |
| 操作系统 | Windows 11 家庭中文版，10.0.26200，64 位 |
| 电源计划 | 平衡 |
| Python | 3.11.16 |
| PyTorch | 2.11.0 |
| Transformers | 5.16.1 |
| Ultralytics | 8.4.140 |
| OpenCV | 5.0.0.93 |
| OpenVINO | 2026.3.1 |
| Gradio | 6.26.0 |
| YOLO | YOLO11n，OpenVINO，FP16，请求并记录为 `intel:gpu` |
| VLM | Qwen2-VL-2B-Instruct，bfloat16，CPU |

OpenVINO 在测试时返回 `CPU`、`GPU.0`、`GPU.1` 三个可用设备 ID。现有脚本记录了 YOLO 的请求设备和已校验设备，但没有把编译后设备 ID 反查成显卡完整名称，因此本文不把某个 `GPU.x` 进一步武断映射为具体显卡型号。

## 4. 测试视频与 Pipeline 参数

| 参数 | 数值 |
|---|---:|
| 文件名 | `radio.mp4` |
| SHA-256 | `d799965a279e29f25324d36d2c4d366a6876603ae646992623d293d950bae44f` |
| 文件大小 | 1,373,841 字节 |
| 编码 | H.264 |
| 分辨率 | 1280×720 |
| 原始帧率 | 24.000 FPS |
| 总帧数 | 361 |
| 视频时长 | 15.042 秒 |
| 分析采样率 | 4.0 FPS |
| 缓冲采样率 | 4.0 FPS |
| 每次关键帧数 | 9 |
| YOLO 置信度阈值 | 0.35 |
| 资源采样间隔 | 0.5 秒 |
| 完整测量次数 | 1 |

帧数与帧率换算得到 361 ÷ 24 = 15.042 秒，与记录的视频时长一致。

## 5. 指标口径

| 指标 | 口径 |
|---|---|
| End-to-End Latency | 模型加载完成后，从完整 Pipeline 开始到分析结果返回的墙钟时间 |
| 视频处理 FPS | 源视频总帧数 ÷ E2E 时间 |
| Pipeline 分析吞吐 | 实际进入 YOLO 分析的帧数 ÷ E2E 时间 |
| 实时倍速 | 源视频时长 ÷ E2E 时间；1.0× 表示与视频等速 |
| YOLO 延迟 | 每次目标检测调用的独立计时统计 |
| 系统 CPU | `psutil` 对整机 CPU 的周期采样 |
| 进程内存 | Python 进程常驻集 RSS 的周期采样 |

模型加载时间未计入 E2E：YOLO 0.827 秒、Qwen 3.695 秒，合计 4.522 秒。总墙钟时间 1198.528 秒与“E2E + 模型加载”基本吻合。

## 6. 核心性能结果

| 指标 | 结果 | 判断 |
|---|---:|---|
| End-to-End Latency | 1194.002 秒 | 处理完整 15.042 秒视频约需 19 分 54 秒 |
| 视频处理 FPS | 0.302 FPS | 以 361 个源帧除以 E2E 时间 |
| Pipeline 分析吞吐 | 0.051 帧/秒 | 61 个分析帧除以 E2E 时间 |
| 实时倍速 | 0.0126× | 未达到实时 |
| 每处理 1 秒视频所需时间 | 79.380 秒 | 当前完整功能基线 |
| 已处理源帧 | 361 帧 | 与视频总帧数一致 |
| 实际分析帧 | 61 帧 | 与 4 FPS 采样设置一致 |
| 错误数 | 0 | 本轮完整执行成功 |

## 7. 模型推理与应用结果

### 7.1 YOLO11n 延迟

| 统计项 | 结果 |
|---|---:|
| 调用次数 | 61 |
| 平均延迟 | 25.165 毫秒 |
| 中位数 | 17.067 毫秒 |
| P95 | 21.945 毫秒 |
| 最大值 | 506.493 毫秒 |

最大值明显高于中位数和 P95，说明少量启动或调度抖动抬高了平均值；单次测试不足以判断其长期分布。

### 7.2 Qwen2-VL 调用

| 调用阶段 | 次数 |
|---|---:|
| 主判断 | 3 |
| 位置复核 | 3 |
| 窗口标签 | 6 |
| 转场帧定位 | 4 |
| 边界复核 | 0 |
| 总结 | 0 |
| 合计 | 16 |

当前脚本只记录调用次数，没有记录 VLM 的平均、中位数或 P95 延迟，因此本文不提供虚构的 VLM 延迟数值。

### 7.3 应用输出

| 指标 | 结果 |
|---|---:|
| 识别事件数 | 6 |
| 保存事件记忆数 | 6 |
| Pipeline 错误数 | 0 |

## 8. 资源占用

| 指标 | 平均值 | 峰值 |
|---|---:|---:|
| 系统 CPU 利用率 | 73.658% | 96.700% |
| 进程 CPU（按 32 线程归一化） | 67.609% | 77.441% |
| 进程内存 RSS | 5.102 GiB | 5.430 GiB |

资源监控共采样 2276 次，采样间隔 0.5 秒。原始 JSON 中的进程 CPU 为 2163.471% / 2478.100%，这是 `psutil` 把多核并行累加后的口径，可超过 100%；为便于阅读，表中同时给出除以 32 个逻辑线程后的归一化值。系统 CPU 指标不需要归一化。

本轮没有可靠采集 GPU 利用率、显存占用、功耗和温度，因此这些指标不进入正式结果，也不以 0 代替缺失值。

## 9. 数据质量与限制

### 9.1 可用性判断

- 状态为 `success`，错误数为 0。
- 视频元数据、处理帧数、分析帧数和调用次数能够交叉核对。
- 四份原始文件的 SHA-256 与 `hashes.sha256` 一致。
- 测试完整保留 Qwen2-VL，没有通过关闭核心功能来获得更好结果。

### 9.2 NPU 误识别校正

原始 `environment.json` 将 `Microsoft Input Configuration Device` 识别为 NPU。原因是旧采集规则用 `NPU|VPU` 匹配整个设备实例字符串，英文 `INPUT` 中恰好包含连续的 `NPU`。这不是实际 NPU 证据。

本报告据此排除该字段，不声明 NPU 存在、参与推理或具有任何利用率。Intel 官方说明，Windows 中的 Intel NPU 通常以 **Intel AI Boost** 标识；Intel Core i9-14900HX 的官方规格页也未列出 NPU。项目中的采集正则已改为严格单词/分隔符边界，避免再次把 `INPUT` 误识别为 NPU。

- Intel Core i9-14900HX 官方规格：<https://www.intel.com/content/www/us/en/products/sku/235995/intel-core-i9-processor-14900hx-36m-cache-up-to-5-80-ghz/specifications.html>
- Intel NPU 识别说明：<https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html>

### 9.3 结论边界

- 仅完成 1 次完整运行，不能据此评价多轮波动或长期稳定性。
- 测试视频只有 15.042 秒，适合快速基线，不代表所有视频长度与内容。
- 电源计划为“平衡”，不是最佳性能模式。
- VLM 没有单独计时，无法严格拆分 E2E 中各阶段占比。
- 本结果没有可用于比较的优化前数据，不构造“优化前后提升百分比”。

## 10. 比赛性能要求覆盖情况

| 比赛候选指标 | 是否提供 | 本次证据 |
|---|---|---|
| 视频处理 FPS | 是 | 0.302 FPS |
| End-to-End Latency | 是 | 1194.002 秒 |
| 单模型推理 Latency | 是 | YOLO 平均 25.165 毫秒，中位数 17.067 毫秒，P95 21.945 毫秒 |
| 视频 Pipeline Throughput | 是 | 0.051 分析帧/秒 |
| CPU 利用率 | 是 | 系统平均 73.658%，峰值 96.700% |
| 内存占用 | 是 | 平均 5.102 GiB，峰值 5.430 GiB |

本次 Intel AI PC 性能测试已提供比赛要求中的至少四项性能指标，并完整说明硬件配置、模型、推理框架、模型精度、视频分辨率和视频帧率。

## 11. 复现命令

在完成 Intel GPU OpenVINO 导出检查并设置 Qwen 本地模型路径后执行：

```powershell
python tools/performance/run_intel_benchmark.py `
  --source "D:\Videos\radio.mp4" `
  --qwen-model-path "D:\AIModels\Qwen2-VL-2B-Instruct" `
  --yolo-device intel:gpu `
  --label intel_ai_pc
```

视频路径和 Qwen 路径必须替换为测试机上的真实位置；重复测试必须保持视频哈希和 Pipeline 参数一致。
