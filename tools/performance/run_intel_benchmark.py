"""Run one complete VisionOracle performance test on an Intel AI PC.

The benchmark deliberately performs one measured run only. Models are loaded
before the E2E timer, but no full-video warm-up pass is performed. The measured
pipeline still includes YOLO, tracking, EventEngine, Qwen2-VL, memory creation,
and final event processing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cv2
    import psutil
    import torch
except ImportError as exc:  # pragma: no cover - preflight error on target PC
    raise SystemExit(
        f"Missing benchmark dependency: {exc}. "
        "Run: python -m pip install -r requirements.txt"
    ) from exc

from src.detection.detector import YoloDetector
from src.pipeline.runtime import EndToEndRunner
from src.vlm.qwen_vlm import MODEL_NAME, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real Intel AI PC VisionOracle benchmark."
    )
    parser.add_argument("--source", required=True, help="Path to the test video")
    parser.add_argument(
        "--qwen-model-path",
        default=os.getenv("QWEN_VL_MODEL_PATH", ""),
        help="Local Qwen model directory; defaults to QWEN_VL_MODEL_PATH",
    )
    parser.add_argument("--yolo-device", default="cpu")
    parser.add_argument("--yolo-confidence", type=float, default=0.35)
    parser.add_argument("--analysis-fps", type=float, default=4.0)
    parser.add_argument("--buffer-fps", type=float, default=4.0)
    parser.add_argument("--keyframe-count", type=int, default=9)
    parser.add_argument("--resource-interval", type=float, default=0.5)
    parser.add_argument("--label", default="intel_ai_pc")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "docs" / "performance"),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_label(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return cleaned.strip("_") or "intel_ai_pc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize_seconds(values: list[float]) -> dict[str, float | int | None]:
    return {
        "calls": len(values),
        "mean_seconds": statistics.fmean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "p95_seconds": percentile(values, 0.95),
        "max_seconds": max(values) if values else None,
    }


def probe_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open test video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    finally:
        capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "file_name": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "codec": codec.strip("\x00 "),
    }


def model_dtype(model: Any) -> str | None:
    try:
        return str(next(model.parameters()).dtype).replace("torch.", "")
    except (AttributeError, StopIteration):
        return None


def model_devices(model: Any) -> list[str]:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        return sorted({str(value) for value in device_map.values()})
    try:
        return [str(next(model.parameters()).device)]
    except (AttributeError, StopIteration):
        return []


class TimedDetector:
    """Transparent detector proxy that records real YOLO call latency."""

    def __init__(self, detector: YoloDetector) -> None:
        self.detector = detector
        self.latencies: list[float] = []

    def __call__(self, frame: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.detector(frame)
        finally:
            self.latencies.append(time.perf_counter() - started)


class ResourceSampler:
    def __init__(self, interval: float) -> None:
        self.interval = max(0.1, float(interval))
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples: list[dict[str, float]] = []

    def start(self) -> None:
        self.process.cpu_percent(None)
        psutil.cpu_percent(None)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                memory = self.process.memory_info()
                self.samples.append(
                    {
                        "process_cpu_percent": self.process.cpu_percent(None),
                        "system_cpu_percent": psutil.cpu_percent(None),
                        "process_rss_bytes": float(memory.rss),
                    }
                )
            except (psutil.Error, OSError):
                continue

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, self.interval * 3))
        if not self.samples:
            try:
                memory = self.process.memory_info()
                self.samples.append(
                    {
                        "process_cpu_percent": self.process.cpu_percent(None),
                        "system_cpu_percent": psutil.cpu_percent(None),
                        "process_rss_bytes": float(memory.rss),
                    }
                )
            except (psutil.Error, OSError):
                pass

    def summary(self) -> dict[str, float | int | None]:
        process_cpu = [item["process_cpu_percent"] for item in self.samples]
        system_cpu = [item["system_cpu_percent"] for item in self.samples]
        rss = [item["process_rss_bytes"] for item in self.samples]
        gib = 1024 ** 3
        return {
            "sample_count": len(self.samples),
            "sample_interval_seconds": self.interval,
            "process_cpu_mean_percent": statistics.fmean(process_cpu) if process_cpu else None,
            "process_cpu_peak_percent": max(process_cpu) if process_cpu else None,
            "system_cpu_mean_percent": statistics.fmean(system_cpu) if system_cpu else None,
            "system_cpu_peak_percent": max(system_cpu) if system_cpu else None,
            "process_memory_mean_gib": statistics.fmean(rss) / gib if rss else None,
            "process_memory_peak_gib": max(rss) / gib if rss else None,
        }


def collect_windows_environment(output_path: Path) -> dict[str, Any]:
    script = Path(__file__).with_name("collect_intel_environment.ps1")
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        return {"collection_error": "PowerShell was not found"}
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-OutputPath",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {"collection_error": completed.stderr.strip() or completed.stdout.strip()}
    try:
        return json.loads(output_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"collection_error": str(exc)}


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def write_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not collected"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_notes(
    path: Path,
    *,
    video: dict[str, Any],
    environment: dict[str, Any],
    metrics: dict[str, Any],
    pipeline: dict[str, Any],
    status: str,
) -> None:
    cpu_names = environment.get("cpu", {}).get("names", [])
    lines = [
        "# Intel AI PC single-run performance test",
        "",
        f"- Status: `{status}`",
        "- Protocol: models loaded before timing; one complete measured run; no full-video warm-up",
        f"- CPU: {', '.join(cpu_names) if cpu_names else 'not collected'}",
        f"- Video: `{video['file_name']}`",
        f"- Resolution: {video['resolution']}",
        f"- Source frame rate: {format_value(video['fps'])} FPS",
        f"- Source duration: {format_value(video['duration_seconds'])} s",
        f"- Analysis FPS setting: {pipeline['analysis_fps']}",
        f"- YOLO requested device: `{pipeline['yolo_requested_device']}`",
        f"- YOLO actual device: `{pipeline['yolo_device']}`",
        f"- YOLO backend: `{pipeline['yolo_backend']}`",
        f"- YOLO precision: `{pipeline['yolo_dtype'] or 'not collected'}`",
        f"- Qwen devices: `{', '.join(pipeline['qwen_devices']) or 'not collected'}`",
        "",
        "## Required metrics",
        "",
        "| Metric | Value | Definition |",
        "|---|---:|---|",
        f"| End-to-End Latency | {format_value(metrics['e2e_latency_seconds'])} s | Complete measured pipeline wall time |",
        f"| Video processing FPS | {format_value(metrics['video_processing_fps'])} FPS | Decoded frames divided by E2E time |",
        f"| Pipeline analysis throughput | {format_value(metrics['pipeline_analysis_throughput_fps'])} frames/s | Actual YOLO calls divided by E2E time |",
        f"| Process CPU mean / peak | {format_value(metrics['resources']['process_cpu_mean_percent'])}% / {format_value(metrics['resources']['process_cpu_peak_percent'])}% | psutil process sampling; may exceed 100% on multicore CPUs |",
        f"| Process memory mean / peak | {format_value(metrics['resources']['process_memory_mean_gib'])} / {format_value(metrics['resources']['process_memory_peak_gib'])} GiB | Process RSS |",
        f"| Real-time factor | {format_value(metrics['realtime_factor'])}x | Source duration divided by E2E time |",
        "",
        "This short-video result is a basic local performance measurement. It is not a long-duration stability test.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hashes(path: Path, video: Path, generated_files: list[Path]) -> None:
    rows = [f"{sha256_file(video)}  TEST_VIDEO:{video.name}"]
    rows.extend(f"{sha256_file(item)}  {item.name}" for item in generated_files)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Test video not found: {source}")
    if args.qwen_model_path:
        qwen_path = Path(args.qwen_model_path).expanduser().resolve()
        if not qwen_path.is_dir():
            raise SystemExit(f"Qwen model directory not found: {qwen_path}")
        qwen_model_path: str | None = str(qwen_path)
    else:
        qwen_model_path = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).expanduser().resolve() / f"{safe_label(args.label)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    environment_path = output_dir / "environment.json"
    results_path = output_dir / "performance_results.json"
    csv_path = output_dir / "performance_runs.csv"
    notes_path = output_dir / "test_notes.md"
    hashes_path = output_dir / "hashes.sha256"

    video = probe_video(source)
    environment = collect_windows_environment(environment_path)
    environment.update(
        {
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
            },
            "software": {
                name: package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "ultralytics",
                    "opencv-python",
                    "gradio",
                    "psutil",
                    "openvino",
                )
            },
            "git_commit": git_commit(),
        }
    )
    atomic_json(environment_path, environment)

    runtime_dir = PROJECT_ROOT / "data" / "performance_runtime" / output_dir.name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    sampler: ResourceSampler | None = None
    result: dict[str, Any] | None = None
    benchmark: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": started_at,
        "protocol": {
            "measured_runs": 1,
            "full_video_warmup_runs": 0,
            "models_loaded_before_e2e_timer": True,
        },
        "video": video,
    }
    atomic_json(results_path, benchmark)

    try:
        print("[Benchmark] Loading YOLO before the E2E timer...")
        load_started = time.perf_counter()
        base_detector = YoloDetector(
            device=args.yolo_device,
            conf_threshold=args.yolo_confidence,
        )
        yolo_load_seconds = time.perf_counter() - load_started

        print("[Benchmark] Loading Qwen2-VL before the E2E timer...")
        load_started = time.perf_counter()
        qwen_model, _ = load_model(model_path=qwen_model_path)
        qwen_load_seconds = time.perf_counter() - load_started

        yolo_core_model = getattr(getattr(base_detector, "_model", None), "model", None)
        detected_yolo_dtype = getattr(base_detector, "precision", None) or model_dtype(
            yolo_core_model
        )
        pipeline_config = {
            "analysis_fps": args.analysis_fps,
            "buffer_fps": args.buffer_fps,
            "keyframe_count": args.keyframe_count,
            "yolo_confidence": args.yolo_confidence,
            "yolo_requested_device": args.yolo_device,
            "yolo_device": getattr(base_detector, "execution_device", args.yolo_device),
            "yolo_backend": getattr(base_detector, "backend", "unknown"),
            "yolo_model_path": str(getattr(base_detector, "model_path", "")),
            "yolo_dtype": detected_yolo_dtype,
            "openvino_available_devices": getattr(
                base_detector, "available_openvino_devices", []
            ),
            "qwen_model": MODEL_NAME,
            "qwen_dtype": model_dtype(qwen_model),
            "qwen_devices": model_devices(qwen_model),
            "torch_cuda_available": torch.cuda.is_available(),
        }
        environment["inference"] = pipeline_config
        atomic_json(environment_path, environment)

        timed_detector = TimedDetector(base_detector)
        runner = EndToEndRunner(
            source=str(source),
            detector=timed_detector,
            use_vlm=True,
            qwen_model_path=qwen_model_path,
            yolo_device=args.yolo_device,
            yolo_confidence=args.yolo_confidence,
            analysis_fps=args.analysis_fps,
            buffer_fps=args.buffer_fps,
            keyframe_count=args.keyframe_count,
            clips_dir=runtime_dir / "clips",
            recordings_dir=runtime_dir / "recordings",
            replays_dir=runtime_dir / "replays",
        )

        print("[Benchmark] Starting the single measured run...")
        sampler = ResourceSampler(args.resource_interval)
        sampler.start()
        e2e_started = time.perf_counter()
        try:
            result = runner.run()
        finally:
            e2e_seconds = time.perf_counter() - e2e_started
            sampler.stop()

        processed_frames = float(result.get("stream", {}).get("frames", 0.0))
        analysis_frames = len(timed_detector.latencies)
        duration_seconds = float(video.get("duration_seconds", 0.0))
        errors = list(result.get("errors", []))
        vlm_calls = dict(result.get("vlm_calls", {}))
        valid = not errors and int(vlm_calls.get("total", 0)) > 0
        status = "success" if valid else "invalid"

        metrics = {
            "e2e_latency_seconds": e2e_seconds,
            "video_processing_fps": processed_frames / e2e_seconds if e2e_seconds > 0 else None,
            "pipeline_analysis_throughput_fps": analysis_frames / e2e_seconds if e2e_seconds > 0 else None,
            "realtime_factor": duration_seconds / e2e_seconds if e2e_seconds > 0 else None,
            "seconds_per_video_second": e2e_seconds / duration_seconds if duration_seconds > 0 else None,
            "processed_frames": processed_frames,
            "analysis_frames": analysis_frames,
            "yolo_latency": summarize_seconds(timed_detector.latencies),
            "resources": sampler.summary(),
        }
        benchmark.update(
            {
                "status": status,
                "completed_at_utc": utc_now(),
                "model_load": {
                    "yolo_seconds": yolo_load_seconds,
                    "qwen_seconds": qwen_load_seconds,
                    "included_in_e2e": False,
                },
                "environment": environment,
                "pipeline": pipeline_config,
                "metrics": metrics,
                "application_result": {
                    "events_detected": result.get("events_detected"),
                    "memories_saved": result.get("memories_saved"),
                    "vlm_calls": vlm_calls,
                    "errors": errors,
                },
            }
        )
        atomic_json(results_path, benchmark)

        csv_row = {
            "status": status,
            "video_file": video["file_name"],
            "video_duration_seconds": duration_seconds,
            "video_resolution": video["resolution"],
            "video_source_fps": video["fps"],
            "yolo_backend": pipeline_config["yolo_backend"],
            "yolo_device": pipeline_config["yolo_device"],
            "yolo_precision": pipeline_config["yolo_dtype"],
            "e2e_latency_seconds": metrics["e2e_latency_seconds"],
            "video_processing_fps": metrics["video_processing_fps"],
            "pipeline_analysis_throughput_fps": metrics["pipeline_analysis_throughput_fps"],
            "realtime_factor": metrics["realtime_factor"],
            "process_cpu_mean_percent": metrics["resources"]["process_cpu_mean_percent"],
            "process_cpu_peak_percent": metrics["resources"]["process_cpu_peak_percent"],
            "process_memory_mean_gib": metrics["resources"]["process_memory_mean_gib"],
            "process_memory_peak_gib": metrics["resources"]["process_memory_peak_gib"],
            "analysis_frames": analysis_frames,
            "vlm_calls_total": vlm_calls.get("total", 0),
            "errors_count": len(errors),
        }
        write_csv(csv_path, csv_row)
        write_notes(
            notes_path,
            video=video,
            environment=environment,
            metrics=metrics,
            pipeline=pipeline_config,
            status=status,
        )
        write_hashes(
            hashes_path,
            source,
            [environment_path, results_path, csv_path, notes_path],
        )

        print(f"[Benchmark] Status: {status}")
        print(f"[Benchmark] Results: {output_dir}")
        if errors:
            print(f"[Benchmark] Application errors: {len(errors)}")
        if int(vlm_calls.get("total", 0)) <= 0:
            print("[Benchmark] Invalid: Qwen VLM was not called.")
        return 0 if valid else 2
    except Exception as exc:
        if sampler is not None:
            sampler.stop()
        benchmark.update(
            {
                "status": "failed",
                "completed_at_utc": utc_now(),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        atomic_json(results_path, benchmark)
        write_hashes(hashes_path, source, [environment_path, results_path])
        print(f"[Benchmark] Failed: {exc}", file=sys.stderr)
        print(f"[Benchmark] Partial results: {output_dir}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
