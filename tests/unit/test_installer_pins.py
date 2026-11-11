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
    code = [
        line
        for line in (_ROOT / rel).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")  # install.sh's own usage line is curl | bash
    ]
    piped = [line for line in code if re.search(r"curl[^|]*\|\s*(sudo\s+)?(ba|z)?sh\b", line)]
    assert not piped, (rel, piped)


def test_ollama_installer_pin_is_a_sha256() -> None:
    """An empty pin made `just bootstrap-ollama` refuse on every Linux host."""
    text = (_ROOT / "scripts/bootstrap_ollama.sh").read_text(encoding="utf-8")
    pin = re.search(r'OLLAMA_INSTALL_SH_SHA256="\$\{OLLAMA_INSTALL_SH_SHA256:-([^}]*)\}"', text)
    assert pin is not None
    assert re.fullmatch(r"[0-9a-f]{64}", pin[1])
