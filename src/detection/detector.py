"""YOLO 目标检测 —— 逐帧处理连续视频

本文件：src/detection/detector.py —— YOLO 检测。

设计原则：
    - 输入统一 VideoFrame（来自 video_source.py），输出统一 DetectionResult 列表。
    - 摄像头与视频文件都调用同一个 detect()，逐帧处理连续视频。
    - 结果字段：frame_id / timestamp / class_name / confidence / bbox([x1,y1,x2,y2])。
    - 阈值与类别白名单从 config.py 读取。
    - 模型文件不存在时给出清晰错误，不自动下载/替换模型。
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.detection.config import (
    ALLOWED_CLASSES,
    CONF_THRESHOLD,
    DEVICE,
    IOU_THRESHOLD,
    MODEL_DOWNLOAD_HINT,
    MODEL_PATH,
    OPENVINO_MODEL_PATH,
)
from src.detection.video_source import VideoFrame


# ============ 统一检测结果对象 ============

@dataclass
class DetectionResult:
    """单个目标的检测结果，字段风格与 TrackingResult 一致。"""
    frame_id: int          # 帧序号
    timestamp: float       # 秒
    class_name: str        # 类别名，如 person / book / cell phone
    confidence: float      # 置信度，0~1
    bbox: List[float]      # [x1, y1, x2, y2]，全项目统一


# ============ YOLO 检测器 ============

class YoloDetector:
    """YOLO 检测器：加载一次模型，对每一帧调用 detect()。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        allowed_classes: Optional[set] = None,
        device: Optional[str] = None,
    ):
        self._device = str(device or DEVICE).strip()
        self.backend = "pytorch"
        self.execution_device = self._device
        self.precision: Optional[str] = None
        self.available_openvino_devices: List[str] = []

        # Intel 显卡不是 CUDA 设备。Ultralytics 的 OpenVINO 后端使用
        # intel:gpu / intel:cpu / intel:npu 设备名，并加载导出的 IR 目录。
        if self._device.lower().startswith("intel:"):
            self.backend = "openvino"
            requested_kind = self._device.split(":", 1)[1].strip().upper()
            if requested_kind not in {"GPU", "CPU", "NPU"}:
                raise ValueError(
                    f"不支持的 Intel YOLO 设备: {self._device!r}。"
                    "请使用 intel:gpu、intel:cpu 或 intel:npu。"
                )

            configured_openvino_path = os.getenv("YOLO_OPENVINO_MODEL_PATH", "").strip()
            self._model_path = Path(
                model_path or configured_openvino_path or OPENVINO_MODEL_PATH
            ).expanduser()
            self._validate_openvino_runtime(requested_kind)
            self.execution_device = f"intel:{requested_kind.lower()}"
            self._load_openvino_manifest()
        else:
            # 原有 PyTorch / CUDA / CPU 路径保持不变。
            self._model_path = Path(model_path).expanduser() if model_path else MODEL_PATH

        if not self._model_path.exists():
            if self.backend == "openvino":
                raise FileNotFoundError(self._openvino_model_hint())
            raise FileNotFoundError(MODEL_DOWNLOAD_HINT)

        if self.backend == "openvino":
            xml_files = list(self._model_path.glob("*.xml")) if self._model_path.is_dir() else []
            bin_files = list(self._model_path.glob("*.bin")) if self._model_path.is_dir() else []
            metadata_path = self._model_path / "metadata.yaml"
            if not xml_files or not bin_files or not metadata_path.is_file():
                raise FileNotFoundError(self._openvino_model_hint())

        # 阈值与类别白名单（默认取 config 中的配置）
        self._conf = conf_threshold if conf_threshold is not None else CONF_THRESHOLD
        self._iou = iou_threshold if iou_threshold is not None else IOU_THRESHOLD
        self._allowed_classes = set(allowed_classes) if allowed_classes else set(ALLOWED_CLASSES)
        # 延迟导入 ultralytics：模型文件缺失时无需先装包也能给出清晰报错
        from ultralytics import YOLO

        self._model = YOLO(str(self._model_path))
        self._names = self._load_class_names()

        # 类别白名单 → 模型类别索引。
        # 如果配置中的类别不在当前 YOLO 模型中，则自动跳过。
        self._class_ids: Optional[List[int]] = None
        if self._allowed_classes:
            self._class_ids = [
                idx for idx, name in self._names.items()
                if name in self._allowed_classes
            ]
            if not self._class_ids:
                raise ValueError(
                    f"允许类别 {sorted(self._allowed_classes)} 均不在模型类别中，"
                    f"模型类别示例: {list(self._names.values())[:10]}"
                )

    @property
    def model_path(self) -> Path:
        """实际加载的 YOLO 模型路径，供性能报告记录。"""
        return self._model_path

    def _openvino_model_hint(self) -> str:
        return (
            f"未找到有效的 OpenVINO YOLO 模型目录: {self._model_path}\n"
            "请先安装 Intel 依赖并导出模型：\n"
            "  python -m pip install -r requirements-intel.txt\n"
            "  python tools/intel/prepare_yolo_openvino.py --device intel:gpu\n"
            "目录中必须包含 .xml、.bin 和 metadata.yaml 文件。"
        )

    def _validate_openvino_runtime(self, requested_kind: str) -> None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "使用 Intel GPU 运行 YOLO 需要 OpenVINO。"
                "请执行: python -m pip install -r requirements-intel.txt"
            ) from exc

        try:
            self.available_openvino_devices = [
                str(item) for item in ov.Core().available_devices
            ]
        except Exception as exc:
            raise RuntimeError(f"OpenVINO 无法读取本机推理设备: {exc}") from exc

        device_found = any(
            item.upper() == requested_kind
            or item.upper().startswith(f"{requested_kind}.")
            for item in self.available_openvino_devices
        )
        if not device_found:
            available = ", ".join(self.available_openvino_devices) or "无"
            raise RuntimeError(
                f"OpenVINO 未识别到请求的 {requested_kind} 设备；"
                f"当前可用设备: {available}。请安装或更新 Intel 显卡驱动。"
            )

    def _load_openvino_manifest(self) -> None:
        manifest_path = self._model_path / "visionoracle_export.json"
        if not manifest_path.is_file():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            precision = manifest.get("precision")
            if isinstance(precision, str) and precision:
                self.precision = precision.lower()
        except (OSError, ValueError, TypeError):
            # 该清单只用于报告溯源，损坏时不阻止有效 IR 模型运行。
            self.precision = None

    def _load_class_names(self) -> dict[int, str]:
        if self.backend != "openvino":
            return {int(index): str(name) for index, name in self._model.names.items()}

        # 直接读取导出元数据，避免访问 YOLO.names 时先用默认设备临时加载
        # 一次 OpenVINO 模型；真正加载只发生在带 intel:* 参数的 predict()。
        from ultralytics.utils import YAML

        metadata = YAML.load(self._model_path / "metadata.yaml")
        raw_names = metadata.get("names", {}) if isinstance(metadata, dict) else {}
        if isinstance(raw_names, list):
            names = {index: str(name) for index, name in enumerate(raw_names)}
        elif isinstance(raw_names, dict):
            names = {int(index): str(name) for index, name in raw_names.items()}
        else:
            names = {}
        if not names:
            raise ValueError(
                f"OpenVINO 模型元数据中缺少类别名称: {self._model_path / 'metadata.yaml'}"
            )
        return names

    def detect(self, video_frame: VideoFrame) -> List[DetectionResult]:
        """对单帧执行检测，返回该帧所有目标的 DetectionResult 列表。"""
        results = self._model.predict(
            source=video_frame.frame,
            conf=self._conf,
            iou=self._iou,
            classes=self._class_ids,
            device=self._device,
            verbose=False,
        )
        detections: List[DetectionResult] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(DetectionResult(
                frame_id=video_frame.frame_id,
                timestamp=video_frame.timestamp,
                class_name=self._names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                bbox=[round(v, 2) for v in (x1, y1, x2, y2)],
            ))
        return detections

    def __call__(self, video_frame: VideoFrame) -> List[DetectionResult]:
        """允许检测器直接作为 Pipeline 的逐帧回调使用。"""
        return self.detect(video_frame)


# ============ 统一创建入口 ============

def create_detector(**kwargs) -> YoloDetector:
    """统一创建检测器；摄像头与视频文件共用同一个实例。"""
    return YoloDetector(**kwargs)
