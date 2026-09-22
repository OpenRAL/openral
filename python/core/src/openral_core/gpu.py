"""Torch-free GPU VRAM probe shared across layers.

Both ``openral deploy sim`` (``python/cli``) and the reasoner node
(``packages/openral_reasoner_ros``) needed the same one-shot ``nvidia-smi``
query for GPU 0's VRAM in GB — duplicated because the CLI cannot import the
ROS package. Lifted here (``openral_core``, importable from every layer)
rather than re-solved twice.
"""

from __future__ import annotations

import subprocess

__all__ = ["detect_gpu_vram_gb"]


def detect_gpu_vram_gb(field: str) -> float:
    """One ``nvidia-smi --query-gpu=<field>`` value for GPU 0 in GB, or ``0.0``.

    Deliberately torch-free — callers must not pull in torch just to size the
    GPU. Any failure (no ``nvidia-smi``, no GPU, parse error) returns ``0.0``;
    callers skip their check rather than blocking on a host where the value
    can't be read. ``field`` is an ``nvidia-smi --query-gpu`` column, e.g.
    ``memory.total`` or ``memory.free``.

    Example:
        >>> detect_gpu_vram_gb("memory.total") >= 0.0
        True
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    lines = out.stdout.strip().splitlines()
    if not lines:
        return 0.0
    try:
        return float(lines[0].strip()) / 1024.0  # MiB → GiB
    except ValueError:
        return 0.0
