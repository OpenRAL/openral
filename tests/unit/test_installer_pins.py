"""The installers fetch pinned releases and verify them; nothing is piped into a shell."""

from __future__ import annotations

import re
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


def test_ollama_installer_pin_is_a_sha256() -> None:
    """An empty pin made `just bootstrap-ollama` refuse on every Linux host."""
    text = (_ROOT / "scripts/bootstrap_ollama.sh").read_text(encoding="utf-8")
    pin = re.search(r'OLLAMA_INSTALL_SH_SHA256="\$\{OLLAMA_INSTALL_SH_SHA256:-([^}]*)\}"', text)
    assert pin is not None
    assert re.fullmatch(r"[0-9a-f]{64}", pin[1])
