import argparse
from pathlib import Path

from src.pipeline.runtime import EndToEndRunner, save_run_report


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(description="AI Video Intelligence 端到端运行器")
    parser.add_argument("--source", required=True, help="视频路径，或摄像头编号 0")
    parser.add_argument("--qwen-model-path", help="本地 Qwen2/Qwen2.5-VL 模型目录")
    parser.add_argument("--yolo-device", default="cpu", help="cpu、0 或 cuda:0")
    parser.add_argument("--yolo-confidence", type=float, default=0.5)
    parser.add_argument("--analysis-fps", type=float, default=2.0)
    parser.add_argument("--buffer-fps", type=float, default=2.0)
    parser.add_argument("--buffer-duration", type=float, default=30.0)
    parser.add_argument("--recording-segment-seconds", type=float, default=60.0)
    parser.add_argument("--replay-max-seconds", type=float, default=60.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-duration", type=float, help="摄像头最多运行秒数")
    parser.add_argument("--query", default="刚才发生了什么？")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="逐帧打印 YOLO、Tracker、Event、VLM 和 Memory 的完整输入输出",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="只调试前半链路，不加载 Qwen-VL 及其依赖",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    runner = EndToEndRunner(
        source=parse_source(args.source),
        use_vlm=not args.no_vlm,
        qwen_model_path=args.qwen_model_path,
        yolo_device=args.yolo_device,
        yolo_confidence=args.yolo_confidence,
        analysis_fps=args.analysis_fps,
        buffer_fps=args.buffer_fps,
        buffer_duration=args.buffer_duration,
        recording_segment_seconds=args.recording_segment_seconds,
        replay_max_seconds=args.replay_max_seconds,
        trace=args.trace,
        clips_dir=project_root / "data" / "clips",
        recordings_dir=project_root / "data" / "recordings",
        replays_dir=project_root / "data" / "replays",
    )
    result = runner.run(max_frames=args.max_frames, max_duration=args.max_duration)
    search_results = runner.search(args.query) if result["memories_saved"] else []
    report_path = save_run_report(
        result,
        search_results,
        project_root / "data" / "outputs",
    )

    print("\n=== 运行完成 ===")
    print(f"处理帧数: {int(result['stream']['frames'])}")
    print(f"检测事件: {result['events_detected']}")
    print(f"写入记忆: {result['memories_saved']}")
    print(f"错误数量: {len(result['errors'])}")
    print(f"结果文件: {report_path}")


if __name__ == "__main__":
    main()
