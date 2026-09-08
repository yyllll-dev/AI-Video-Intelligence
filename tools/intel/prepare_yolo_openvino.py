"""Export YOLO11n to OpenVINO and verify inference on an Intel device.

Run from the project root after installing ``requirements-intel.txt``::

    python tools/intel/prepare_yolo_openvino.py --device intel:gpu

The generated OpenVINO directory is a machine-local build artifact and is not
committed to Git. Both uploaded videos and the live camera automatically use it
when ``YOLO_DEVICE=intel:gpu``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "models" / "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO11n to OpenVINO and smoke-test an Intel device."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="PyTorch YOLO weight file (default: models/yolo11n.pt)",
    )
    parser.add_argument(
        "--device",
        choices=("intel:gpu", "intel:cpu", "intel:npu"),
        default="intel:gpu",
        help="OpenVINO device used for the verification inference",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help="Export precision (default: fp16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Static inference image size (default: 640)",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def openvino_files(model_dir: Path) -> tuple[list[Path], list[Path]]:
    if not model_dir.is_dir():
        return [], []
    return list(model_dir.glob("*.xml")), list(model_dir.glob("*.bin"))


def require_device(available_devices: list[str], requested_device: str) -> None:
    requested_kind = requested_device.split(":", 1)[1].upper()
    found = any(
        item.upper() == requested_kind
        or item.upper().startswith(f"{requested_kind}.")
        for item in available_devices
    )
    if not found:
        available = ", ".join(available_devices) or "none"
        raise RuntimeError(
            f"OpenVINO did not detect {requested_kind}. Available devices: {available}. "
            "Install/update the Intel graphics driver before benchmarking."
        )


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"YOLO source weights not found: {source}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero")

    try:
        import numpy as np
        import openvino as ov
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Intel YOLO dependencies are missing. Run: "
            "python -m pip install -r requirements-intel.txt"
        ) from exc

    available_devices = [str(item) for item in ov.Core().available_devices]
    require_device(available_devices, args.device)
    print(f"[Intel YOLO] OpenVINO devices: {', '.join(available_devices)}")

    expected_dir = source.with_name(f"{source.stem}_openvino_model")
    xml_files, bin_files = openvino_files(expected_dir)
    exported_now = False
    if not xml_files and not bin_files:
        print(
            f"[Intel YOLO] Exporting {source.name} as {args.precision.upper()} "
            f"OpenVINO IR ({args.imgsz}x{args.imgsz})..."
        )
        exported = Path(
            YOLO(str(source)).export(
                format="openvino",
                imgsz=args.imgsz,
                batch=1,
                dynamic=False,
                device="cpu",
                quantize=16 if args.precision == "fp16" else 32,
            )
        ).resolve()
        if exported != expected_dir.resolve():
            raise RuntimeError(
                f"Ultralytics exported to an unexpected directory: {exported}; "
                f"expected: {expected_dir.resolve()}"
            )
        exported_now = True
        xml_files, bin_files = openvino_files(expected_dir)
    elif not xml_files or not bin_files:
        raise RuntimeError(
            f"Incomplete OpenVINO model directory: {expected_dir}. "
            "Remove that generated directory and run this command again."
        )
    else:
        print(f"[Intel YOLO] Reusing existing model: {expected_dir}")

    metadata_path = expected_dir / "metadata.yaml"
    if not metadata_path.is_file():
        raise RuntimeError(f"OpenVINO metadata is missing: {metadata_path}")

    manifest_path = expected_dir / "visionoracle_export.json"
    manifest: dict[str, object] | None = None
    if exported_now:
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_model": source.name,
            "source_sha256": sha256(source),
            "precision": args.precision,
            "image_size": args.imgsz,
            "ultralytics_version": version("ultralytics"),
            "openvino_version": version("openvino"),
        }
    elif manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_precision = str(manifest.get("precision", "")).lower()
        if existing_precision and existing_precision != args.precision:
            raise RuntimeError(
                f"Existing model precision is {existing_precision}, not {args.precision}. "
                "Use the matching --precision value or rebuild the generated directory."
            )
    else:
        raise RuntimeError(
            f"Existing OpenVINO model has no export manifest: {manifest_path}. "
            "Remove that generated directory and rerun this command so its "
            "precision and source hash can be certified."
        )

    # A real inference is essential: device enumeration alone does not prove that
    # the exported graph can compile and execute on this Intel device.
    print(f"[Intel YOLO] Running smoke test on {args.device}...")
    test_frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    results = YOLO(str(expected_dir)).predict(
        source=test_frame,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    if not results:
        raise RuntimeError("OpenVINO smoke test returned no result object")

    # 只有真实推理成功后才写入设备验证信息，避免失败的导出被性能报告
    # 当作已验证 Intel GPU 模型。
    if manifest is not None:
        manifest.update(
            {
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "verified_device": args.device,
                "available_openvino_devices": available_devices,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("[Intel YOLO] PASS: inference completed on the requested Intel device.")
    print(f"[Intel YOLO] Model directory: {expected_dir}")
    print(f"[Intel YOLO] Set YOLO_DEVICE={args.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
