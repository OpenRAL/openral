"""``tools/_robometer_scorer.Scorer`` refuses a non-NF4 checkpoint with a typed error.

A bf16 (not pre-quantized) ``model.safetensors`` is a bad-weights configuration,
so it raises ``ROSConfigError`` (CLAUDE.md §5), not a bare ``RuntimeError``.
The scorer module flips process-global torch determinism flags on import, so it
is exercised in a child interpreter to keep them out of this test process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def test_non_prequantized_checkpoint_raises_ros_config_error(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    pytest.importorskip("torch")
    import torch
    from safetensors.torch import save_file

    save_file(
        {"lm_head.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}, tmp_path / "model.safetensors"
    )
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(_REPO / 'tools')!r})\n"
        "import _robometer_scorer as s\n"
        "from openral_core.exceptions import ROSConfigError\n"
        "try:\n"
        f"    s.Scorer({str(tmp_path)!r}, device='cpu')\n"
        "except ROSConfigError as exc:\n"
        "    print('ROSConfigError:', exc)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=True
    )
    assert "ROSConfigError:" in out.stdout
    assert "not an NF4 pre-quantized checkpoint" in out.stdout


def test_warm_up_loads_the_model_at_configure_not_at_the_first_score(tmp_path: Path) -> None:
    """Spark R219: the lazy load ate ~40 s of the goal. ``warm_up`` loads it, typed errors too."""
    pytest.importorskip("safetensors")
    pytest.importorskip("torch")
    import torch
    from safetensors.torch import save_file

    save_file(
        {"lm_head.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}, tmp_path / "model.safetensors"
    )
    script = (
        "from openral_runner.backends.reward.robometer_reward import RobometerInProcessReward\n"
        "from openral_core.exceptions import ROSConfigError\n"
        f"m = RobometerInProcessReward(model_id='robometer', weights_source={str(tmp_path)!r},"
        " device='cpu')\n"
        "try:\n"
        "    m.warm_up()\n"
        "except ROSConfigError as exc:\n"
        "    print('ROSConfigError:', exc)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=True
    )
    assert "ROSConfigError:" in out.stdout, out.stderr
