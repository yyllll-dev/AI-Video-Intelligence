"""YOLO 目标检测 —— 逐帧处理连续视频

本文件：src/detection/detector.py —— YOLO 检测。

设计原则：
    - 输入统一 VideoFrame（来自 video_source.py），输出统一 DetectionResult 列表。
    - 摄像头与视频文件都调用同一个 detect()，逐帧处理连续视频。
    - 结果字段：frame_id / timestamp / class_name / confidence / bbox([x1,y1,x2,y2])。
    - 阈值与类别白名单从 config.py 读取。
    - 模型文件不存在时给出清晰错误，不自动下载/替换模型。
"""

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
        # 模型权重路径（默认取 config.MODEL_PATH）
        self._model_path = Path(model_path) if model_path else MODEL_PATH
        if not self._model_path.exists():
            raise FileNotFoundError(MODEL_DOWNLOAD_HINT)

        # 阈值与类别白名单（默认取 config 中的配置）
        self._conf = conf_threshold if conf_threshold is not None else CONF_THRESHOLD
        self._iou = iou_threshold if iou_threshold is not None else IOU_THRESHOLD
        self._allowed_classes = set(allowed_classes) if allowed_classes else set(ALLOWED_CLASSES)
        self._device = device or DEVICE

        # 延迟导入 ultralytics：模型文件缺失时无需先装包也能给出清晰报错
        from ultralytics import YOLO

        self._model = YOLO(str(self._model_path))

        # 类别白名单 → 模型类别索引。
        # 如果配置中的类别不在当前 YOLO 模型中，则自动跳过。
        self._class_ids: Optional[List[int]] = None
        if self._allowed_classes:
            self._class_ids = [
                idx for idx, name in self._model.names.items()
                if name in self._allowed_classes
            ]
            if not self._class_ids:
                raise ValueError(
                    f"允许类别 {sorted(self._allowed_classes)} 均不在模型类别中，"
                    f"模型类别示例: {list(self._model.names.values())[:10]}"
                )

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
                class_name=self._model.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                bbox=[round(v, 2) for v in (x1, y1, x2, y2)],
            ))
        return detections


# ============ 统一创建入口 ============

def create_detector(**kwargs) -> YoloDetector:
    """统一创建检测器；摄像头与视频文件共用同一个实例。"""
    return YoloDetector(**kwargs)
