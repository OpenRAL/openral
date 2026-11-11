"""Sidecar supply-chain controls: pinned upstream clones and an allowlisted env.

``ensure_source`` fetches a third-party repo at a pinned commit and refuses to
hand back a checkout whose ``HEAD`` is anything else; ``sidecar_env`` keeps
operator credentials (reasoner API keys) away from remote-code sidecars. Real
``git`` against a local repo, no network.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from openral_sim._deps import _RLDX_SIDECAR_REPO_SHA, _rldx_sidecar_setup_plan
from openral_sim._sidecar_common import (
    checkout_head,
    ensure_source,
    make_isolated_env,
    sidecar_env,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def _git(*args: str, cwd: Path) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def _upstream(tmp_path: Path) -> tuple[str, str, str]:
    """A two-commit upstream repo; returns ``(url, first_sha, tip_sha)``."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    for i in (1, 2):
        (repo / "f.txt").write_text(str(i))
        _git("add", "f.txt", cwd=repo)
        _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", f"c{i}", cwd=repo)
    # Let a shallow fetch ask for a non-tip commit by id, as GitHub allows.
    _git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=repo)
    return repo.as_uri(), _git("rev-parse", "HEAD~1", cwd=repo), _git("rev-parse", "HEAD", cwd=repo)


def test_fetches_exactly_the_pinned_commit_not_head(tmp_path: Path) -> None:
    url, first, tip = _upstream(tmp_path)
    source = ensure_source("t", tmp_path / "work", url, first)
    assert checkout_head(source) == first != tip
    # Reuse of a checkout at the pin is fine.
    assert ensure_source("t", tmp_path / "work", url, first) == source


def test_refuses_an_existing_checkout_at_another_commit(tmp_path: Path) -> None:
    url, first, tip = _upstream(tmp_path)
    ensure_source("t", tmp_path / "work", url, tip)
    with pytest.raises(SystemExit, match="not the pinned"):
        ensure_source("t", tmp_path / "work", url, first)


def test_a_failed_fetch_leaves_nothing_that_looks_like_a_checkout(tmp_path: Path) -> None:
    """An interrupted first fetch must not block the next run (no half-made source/)."""
    url, first, _ = _upstream(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        ensure_source("t", tmp_path / "work", url, "0" * 40)  # not in the repo
    assert not (tmp_path / "work" / "source").exists()
    assert checkout_head(ensure_source("t", tmp_path / "work", url, first)) == first


@pytest.mark.parametrize("pin", ["", "main", "HEAD", "cf67c31f", "Z" * 40])
def test_refuses_a_pin_that_is_not_a_full_sha(tmp_path: Path, pin: str) -> None:
    url, _, _ = _upstream(tmp_path)
    with pytest.raises(SystemExit, match="not a full SHA"):
        ensure_source("t", tmp_path / "work", url, pin)
    assert not (tmp_path / "work" / "source").exists()


def test_rldx_install_plan_fetches_the_pin_and_verifies_head() -> None:
    clone = next(s for s in _rldx_sidecar_setup_plan().steps if "fetch" in s.description)
    script = clone.argv[-1]
    assert f"fetch -q --depth 1 origin {_RLDX_SIDECAR_REPO_SHA}" in script
    assert f'rev-parse HEAD)" = {_RLDX_SIDECAR_REPO_SHA} ]' in script
    assert " clone " not in script


def test_sidecar_env_drops_credentials_keeps_runtime() -> None:
    environ = {
        "PATH": "/usr/bin",
        "HOME": "/home/op",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "HF_HOME": "/data/hf",
        "HF_HUB_OFFLINE": "1",
        "HF_TOKEN": "hf_x",
        "OPENRAL_ALLOW_REMOTE_CODE": "1",
        "ROS_DOMAIN_ID": "7",
        "OPENRAL_REASONER_API_KEY": "sk-openral",
        "ANTHROPIC_API_KEY": "sk-ant",
        "OPENAI_API_KEY": "sk-oai",
        "GH_TOKEN": "ghp_x",
        "AWS_SECRET_ACCESS_KEY": "aws",
        "OPENRAL_ROBOT_UI_TOKEN": "tok",
        "SSH_AUTH_SOCK": "/run/agent",
        "RANDOM_TOOL_VAR": "x",
    }
    env = sidecar_env(environ)
    for kept in (
        "PATH", "HOME", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
        "LD_LIBRARY_PATH", "HF_HOME", "HF_HUB_OFFLINE", "HF_TOKEN", "OPENRAL_ALLOW_REMOTE_CODE",
        "ROS_DOMAIN_ID",
    ):  # fmt: skip
        assert env[kept] == environ[kept], kept
    for dropped in (
        "OPENRAL_REASONER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GH_TOKEN",
        "AWS_SECRET_ACCESS_KEY", "OPENRAL_ROBOT_UI_TOKEN", "SSH_AUTH_SOCK", "RANDOM_TOOL_VAR",
    ):  # fmt: skip
        assert dropped not in env, dropped
    assert sidecar_env(environ, extra=["RANDOM_TOOL_VAR"])["RANDOM_TOOL_VAR"] == "x"


def test_sidecar_env_keeps_simulator_dds_and_threading_config() -> None:
    """What the sim sidecars read beyond the basics survives the allowlist."""
    environ = {
        "LD_PRELOAD": "/usr/lib/aarch64-linux-gnu/libgomp.so.1",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "OMP_NUM_THREADS": "8",
        "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
        "CYCLONEDDS_URI": "file:///etc/cyclone.xml",
        "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4",
        "MS_SKIP_ASSET_DOWNLOAD_PROMPT": "1",
        "LIBERO_CONFIG_PATH": "/data/libero",
        "VLABENCH_ROOT": "/data/vlabench",
        "PRIVACY_CONSENT": "Y",
        "CURL_CA_BUNDLE": "/etc/ssl/ca.pem",
    }
    assert sidecar_env(environ) == environ


def test_credential_words_match_whole_words_only() -> None:
    """A *_TOKENS budget is config, not a secret; a *_TOKEN still is."""
    env = sidecar_env(
        {
            "OPENRAL_REASONER_MAX_TOKENS": "4096",
            "OPENRAL_HUB_TOKEN": "t",
            "OPENRAL_SECRET_KEY": "s",
            "OPENRAL_APIKEY": "k",
        }
    )
    assert env == {"OPENRAL_REASONER_MAX_TOKENS": "4096"}


def test_isolated_env_never_carries_the_reasoner_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-openral")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    env = make_isolated_env(Path("/tmp/sidecar/.venv"))
    assert "OPENRAL_REASONER_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env["PATH"].startswith("/tmp/sidecar/.venv/bin")


_TOOLS = Path(__file__).resolve().parents[2] / "tools"
_OPERATOR_SECRETS = {
    "OPENRAL_REASONER_API_KEY": "sk-openral",
    "ANTHROPIC_API_KEY": "sk-ant",
    "OPENAI_API_KEY": "sk-oai",
    "GH_TOKEN": "ghp_x",
    "VLLM_API_KEY": "sk-vllm",
}


@pytest.mark.parametrize(
    ("tool", "reads", "keeps_hf_token"),
    [
        ("da3_depth_sidecar", {"OPENRAL_DA3_DEPTH_SIDECAR_HOME": "/h"}, False),
        ("qwen_vlm_sidecar", {"OPENRAL_QWEN_VLM_SIDECAR_VENV": "/v"}, True),
        (
            "locateanything_sidecar",
            {"OPENRAL_LOCATEANYTHING_SIDECAR_HOME": "/h", "OPENRAL_ALLOW_REMOTE_CODE": "1"},
            True,
        ),
        (
            "cosmos3_reasoner_sidecar",
            {
                "OPENRAL_COSMOS3_MAX_MODEL_LEN": "8192",
                "OPENRAL_COSMOS3_SIDECAR_HOME": "/h",
                "VLLM_USE_FLASHINFER_SAMPLER": "1",
            },
            True,
        ),
    ],
)
def test_boot_helper_scrubs_environ_before_anything(
    tool: str,
    reads: dict[str, str],
    keeps_hf_token: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The env each sidecar's installs and server inherit: what it reads, no API key.

    ``--help`` exits right after argument parsing, i.e. after the scrub and before
    any venv install or exec, so the patched ``os.environ`` is exactly what the
    rest of the boot (and the exec'd server) would see.
    """
    import importlib.util
    import os
    import sys

    environ = {"PATH": "/usr/bin", "HF_HOME": "/data/hf", "HF_TOKEN": "hf_x"}
    environ |= reads | _OPERATOR_SECRETS
    spec = importlib.util.spec_from_file_location(f"_scrub_{tool}", _TOOLS / f"{tool}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(os, "environ", dict(environ))
    monkeypatch.setattr(sys, "argv", [tool, "--help"])
    with pytest.raises(SystemExit):
        module.main()
    capsys.readouterr()
    expected = {"PATH": "/usr/bin", "HF_HOME": "/data/hf", **reads}
    if keeps_hf_token:
        expected["HF_TOKEN"] = "hf_x"
    assert dict(os.environ) == expected
