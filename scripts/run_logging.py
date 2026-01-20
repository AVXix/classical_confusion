from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import os
import platform
import sys


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize(s: str) -> str:
    s = s.strip().replace(" ", "_")
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
    out = "".join(keep)
    return out[:80] if out else "run"


def _try_version(mod_name: str) -> str | None:
    try:
        mod = __import__(mod_name)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def collect_environment_info() -> dict[str, Any]:
    versions = {
        name: _try_version(name)
        for name in [
            "numpy",
            "librosa",
            "torch",
            "sklearn",
            "kaggle",
            "joblib",
        ]
    }
    versions = {k: v for k, v in versions.items() if v is not None}

    return {
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "env": {
            "KAGGLE_CONFIG_DIR": os.environ.get("KAGGLE_CONFIG_DIR"),
            "KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME"),
        },
        "package_versions": versions,
    }


def create_run_dir(*, base_dir: Path, script_name: str, run_name: str | None = None) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{_timestamp()}_{_sanitize(script_name)}"
    if run_name:
        stem += f"_{_sanitize(run_name)}"

    run_dir = base_dir / stem
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    # Rare collision: add numeric suffix.
    for i in range(2, 10_000):
        candidate = base_dir / f"{stem}_{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate

    raise RuntimeError("Could not create a unique run dir")


def _format_kv_block(title: str, d: Mapping[str, Any]) -> str:
    lines = [f"== {title} =="]
    for k in sorted(d.keys(), key=lambda x: str(x)):
        v = d[k]
        lines.append(f"{k}: {v}")
    lines.append("")
    return "\n".join(lines)


def write_run_info_txt(
    *,
    run_dir: Path,
    script_name: str,
    argv: list[str],
    args: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)

    args_dict: dict[str, Any] | None
    if args is None:
        args_dict = None
    elif is_dataclass(args):
        args_dict = asdict(args)
    elif isinstance(args, Mapping):
        args_dict = dict(args)
    else:
        # argparse.Namespace or any object with __dict__.
        args_dict = getattr(args, "__dict__", None)
        if args_dict is not None:
            args_dict = dict(args_dict)

    env = collect_environment_info()

    parts: list[str] = []
    parts.append("== Run ==\n")
    parts.append(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
    parts.append(f"script: {script_name}\n")
    parts.append(f"command: {' '.join(argv)}\n")
    parts.append("\n")

    parts.append(_format_kv_block("Environment", env))

    if args_dict is not None:
        parts.append(_format_kv_block("Hyperparameters / Args", args_dict))

    if extra:
        parts.append(_format_kv_block("Extra", dict(extra)))

    out_path = run_dir / "run_info.txt"
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path
