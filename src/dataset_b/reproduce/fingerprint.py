"""Print one environment fingerprint as JSON; runs inside the profile's interpreter."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from typing import Any


def _torch_fingerprint() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    result: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_index": None,
        "gpu_name": None,
        "vram_bytes": None,
        "cudnn_version": torch.backends.cudnn.version(),
        "finite_cuda_operation": False,
    }
    if result["cuda_available"]:
        total = torch.ones(8, device="cuda:0").sum()
        result.update(
            device_index=0,
            gpu_name=torch.cuda.get_device_name(0),
            vram_bytes=torch.cuda.get_device_properties(0).total_memory,
            finite_cuda_operation=bool(torch.isfinite(total).item()),
        )
    return result


def _driver_version() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip().splitlines()[0] if output.strip() else None


def fingerprint(smoke_imports: list[str]) -> dict[str, Any]:
    import_errors = {}
    for module in smoke_imports:
        try:
            importlib.import_module(module)
        except Exception as exc:  # A smoke import may fail with any error type.
            import_errors[module] = f"{type(exc).__name__}: {exc}"
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
    }
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": dict(sorted(packages.items())),
        "smoke_imports": smoke_imports,
        "import_errors": import_errors,
        "torch": _torch_fingerprint(),
        "nvidia_driver": _driver_version(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imports", default="")
    args = parser.parse_args()
    modules = [module for module in args.imports.split(",") if module]
    print(json.dumps(fingerprint(modules), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
