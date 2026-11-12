"""The installers fetch pinned releases and verify them; nothing is piped into a shell."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_INSTALLERS = (
    "scripts/install.sh",
    "scripts/bootstrap_ollama.sh",
    "Dockerfile.dev",
    "docker/inference/Dockerfile.x86",
    "python/cli/src/openral_cli/bootstrap/bootstrap_ubuntu.sh",
    "python/cli/src/openral_cli/bootstrap/bootstrap_macos.sh",
)


@pytest.mark.parametrize("rel", _INSTALLERS)
def test_no_remote_script_is_piped_into_a_shell(rel: str) -> None:
    # Backslash continuations joined, so `curl … \` / `| sudo bash` is one command.
    text = re.sub(r"\\\n\s*", " ", (_ROOT / rel).read_text(encoding="utf-8"))
    code = [
        line
        for line in text.splitlines()
        if not line.lstrip().startswith("#")  # install.sh's own usage line is curl | bash
    ]
    fetch = r"(?:curl|wget)\b"
    patterns = (
        rf"{fetch}[^|]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b",  # curl … | sh
        rf"(?:ba|z)?sh\s+<\(\s*{fetch}",  # bash <(curl …)
        rf"(?:ba|z)?sh\s+-c\s+[\"']?\$\(\s*{fetch}",  # sh -c "$(curl …)"
    )
    piped = [line for line in code if any(re.search(p, line) for p in patterns)]
    assert not piped, (rel, piped)


def test_the_pipe_guard_sees_a_continued_line() -> None:
    """The pre-pin bootstrap_ubuntu.sh text, split over two lines, is caught."""
    text = "curl -sSf https://just.systems/install.sh \\\n  | sudo bash -s -- --to /x\n"
    joined = re.sub(r"\\\n\s*", " ", text)
    assert re.search(r"(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b", joined)


def test_every_ollama_artifact_is_sha256_pinned_and_verified_before_extraction() -> None:
    """Each tarball the bootstrap installs is pinned and checked before tar sees it (#559)."""
    text = (_ROOT / "scripts/bootstrap_ollama.sh").read_text(encoding="utf-8")
    pins = dict(re.findall(r"^\s*(ollama-linux-[\w-]+)\) echo (\S+) ;;$", text, flags=re.M))
    assert set(pins) == {
        "ollama-linux-amd64",
        "ollama-linux-arm64",
        "ollama-linux-arm64-jetpack5",
        "ollama-linux-arm64-jetpack6",
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in pins.values()), pins
    # Every download is verified before the first extraction, and nothing runs upstream's
    # unverified install.sh any more.
    assert text.index("sha256sum -c") < text.index("tar -xf")
    assert "scripts/install.sh" not in text


def test_bootstrap_offers_exactly_the_curated_ollama_models() -> None:
    """The bootstrap reads its model tags from REASONER_MODELS, not a copy of them (#559)."""
    from openral_core import OLLAMA_BASE_URL, REASONER_MODELS

    out = subprocess.run(
        ["bash", str(_ROOT / "scripts/bootstrap_ollama.sh"), "--help"],
        env={**os.environ, "OPENRAL_PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    curated = sorted(
        (m for m in REASONER_MODELS.values() if m.default_endpoint == OLLAMA_BASE_URL),
        key=lambda m: m.min_gpu_vram_gb or 0.0,
    )
    assert curated
    for m in curated:
        assert f"  {m.served_model_id} {m.id}\n" in out
    # The default is the curated model with the smallest GPU floor.
    assert f"(default: {curated[0].served_model_id})" in out
    assert f"OPENRAL_REASONER_MODEL={curated[0].id}\n" in out
