"""环境版本检查脚本：你和队友各跑一次，对比输出即可。"""
import sys
import platform
import importlib.metadata as metadata

PACKAGES = [
    "torch",
    "transformers",
    "ultralytics",
    "opencv-python",
    "numpy",
    "pillow",
    "modelscope",
    "qwen-vl-utils",
    "accelerate",
    "gradio",
    "pytest",
]


def main() -> None:
    lines = []
    lines.append(f"Python: {sys.version.split()[0]}")
    lines.append(f"OS: {platform.platform()}")
    lines.append("")

    installed = {
        dist.metadata["Name"].lower(): dist.version
        for dist in metadata.distributions()
    }
    for pkg in PACKAGES:
        version = installed.get(pkg.lower(), "NOT INSTALLED")
        lines.append(f"{pkg}: {version}")

    lines.append("")
    lines.append("--- torch / CUDA ---")
    try:
        import torch

        lines.append(f"torch: {torch.__version__}")
        lines.append(f"CUDA: {torch.version.cuda}")
        lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            lines.append("GPU: N/A (running on CPU)")
    except Exception as exc:
        lines.append(f"torch import failed: {exc}")

    output = "\n".join(lines)
    print(output)

    with open("env_check.txt", "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print("\n结果已保存到 env_check.txt")


if __name__ == "__main__":
    main()
