"""Print a snapshot of the runtime environment (Python, PyTorch, accelerator).

Usage: uv run python scripts/env_report.py
"""

import platform
import sys

import torch


def main() -> None:
    print(f"python : {sys.version.split()[0]} ({platform.platform()})")
    print(f"torch  : {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gib = props.total_memory / 2**30
        print(f"cuda   : yes — {props.name} ({vram_gib:.1f} GiB, sm_{props.major}{props.minor})")
        print(f"bf16   : {torch.cuda.is_bf16_supported()}")
    else:
        print("cuda   : not available (CPU-only build or no NVIDIA GPU)")


if __name__ == "__main__":
    main()
