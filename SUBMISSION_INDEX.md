# VisionOracle 评审材料总入口

本文件是比赛评委阅读本项目的第一入口。所有正式提交材料、可运行代码和原始性能证据均保存在当前仓库中。

## 一、提交材料导航

| 比赛要求 | 对应材料 | 建议阅读顺序 |
|---|---|---:|
| 最终项目说明书 PDF | `output/pdf/VisionOracle_Project_Manual.pdf` | 1 |
| 可运行项目代码与完整运行说明 | `README.md`、`requirements.txt`、`requirements-intel.txt`、`run.ps1` | 2 |
| 使用的 AI 模型及模型来源说明 | `docs/MODELS.md` | 3 |
| 应用场景及用户价值说明 | `docs/SCENARIOS_AND_VALUE.md` | 4 |
| AI PC 本地部署及优化方案说明 | `docs/AI_PC_DEPLOYMENT_AND_OPTIMIZATION.md` | 5 |
| Intel AI PC 性能测试结果 | `docs/PERFORMANCE_TEST.md` | 6 |
| 性能测试原始证据 | `docs/performance/intel_ai_pc_20260909_034520/` | 7 |
| 千问协助答题部分沟通截图 | `docs/evidence/qianwen/` | 8 |

## 二、源码架构

```text
AI-Video-Intelligence/
├── SUBMISSION_INDEX.md         # 评审材料总入口
├── README.md                   # 安装、模型准备、运行和使用说明
├── requirements*.txt           # 通用依赖与 Intel OpenVINO 可选依赖
├── run.ps1                     # Windows 一键检查与启动
├── demo/                       # Gradio Web 应用与界面状态管理
├── src/
│   ├── detection/              # 视频源、YOLO 检测与设备选择
│   ├── tracking/               # 目标跨帧关联
│   ├── event/                  # 候选事件、规则和状态机
│   ├── vlm/                    # Qwen2-VL 加载、提示词与结构化解析
│   ├── pipeline/               # 端到端调度、缓存、录像和回放
│   └── retrieval/              # 事件记忆、展示和自然语言检索
├── tools/
│   ├── intel/                  # YOLO OpenVINO 导出与 Intel GPU 校验
│   └── performance/            # 性能测试和硬件环境采集
├── docs/                       # 比赛专项说明与真实性能证据
└── output/pdf/                 # 最终项目说明书
```

## 三、核心处理链路

```text
上传视频 / 实时摄像头
        ↓
统一帧流、采样与真实时间戳
        ↓
YOLO11n 目标检测
        ↓
SimpleTracker + Event Engine
        ↓
关键帧缓冲与时序证据
        ↓
Qwen2-VL 语义确认
        ↓
事件记忆、总结、有效时长、检索与回放
```

## 四、快速运行

普通 Windows / NVIDIA 环境按 `README.md` 安装后运行：

```powershell
.\run.ps1 -QwenModelPath "D:\AIModels\Qwen2-VL-2B-Instruct" -YoloDevice 0
```

Intel AI PC 在完成 OpenVINO 导出与真实设备检查后运行：

```powershell
python tools/intel/prepare_yolo_openvino.py --device intel:gpu --precision fp16
.\run.ps1 -QwenModelPath "D:\AIModels\Qwen2-VL-2B-Instruct" -YoloDevice intel:gpu
```

## 五、材料真实性说明

- Intel AI PC 性能结果来自一次真实完整运行，原始 JSON、CSV、环境快照、测试摘要和 SHA-256 清单均已保留。
- 测试没有关闭 Qwen2-VL，也没有通过删除核心功能获得更好成绩。
- 缺失的 GPU 利用率、功耗和 VLM 单次延迟没有填 0 或虚构。
- 千问截图仅为项目筹备过程中部分沟通记录，最终架构、代码、测试口径和提交材料由团队结合实际项目核验后完成。
