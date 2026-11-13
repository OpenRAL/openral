"""Sidecar supply-chain controls: pinned upstream clones and an allowlisted env.

``ensure_source`` fetches a third-party repo at a pinned commit and refuses to
hand back a checkout whose ``HEAD`` is anything else; ``sidecar_env`` keeps
operator credentials (reasoner API keys) away from remote-code sidecars. Real
``git`` against a local repo, no network.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
from openral_core.exceptions import ROSConfigError
from openral_sim._deps import _RLDX_SIDECAR_REPO_SHA, _rldx_sidecar_setup_plan
from openral_sim._sidecar_common import (
    checkout_head,
    ensure_source,
    make_isolated_env,
    require_pinned_head,
    run_sidecar,
    sidecar_env,
)
from openral_sim.backends import robotwin
from openral_sim.backends.robotwin import _ensure_robotwin_checkout

# Only the fetch tests need git; the environment-policy tests run everywhere.
_needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


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


@_needs_git
def test_fetches_exactly_the_pinned_commit_not_head(tmp_path: Path) -> None:
    url, first, tip = _upstream(tmp_path)
    source = ensure_source("t", tmp_path / "work", url, first)
    assert checkout_head(source) == first != tip
    # Reuse of a checkout at the pin is fine.
    assert ensure_source("t", tmp_path / "work", url, first) == source


@_needs_git
def test_refuses_an_existing_checkout_at_another_commit(tmp_path: Path) -> None:
    url, first, tip = _upstream(tmp_path)
    ensure_source("t", tmp_path / "work", url, tip)
    with pytest.raises(SystemExit, match="not the pinned"):
        ensure_source("t", tmp_path / "work", url, first)


@_needs_git
def test_a_failed_fetch_leaves_nothing_that_looks_like_a_checkout(tmp_path: Path) -> None:
    """An interrupted first fetch must not block the next run (no half-made source/)."""
    url, first, _ = _upstream(tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        ensure_source("t", tmp_path / "work", url, "0" * 40)  # not in the repo
    assert not (tmp_path / "work" / "source").exists()
    assert checkout_head(ensure_source("t", tmp_path / "work", url, first)) == first


@_needs_git
@pytest.mark.parametrize("pin", ["", "main", "HEAD", "cf67c31f", "Z" * 40])
def test_refuses_a_pin_that_is_not_a_full_sha(tmp_path: Path, pin: str) -> None:
    url, _, _ = _upstream(tmp_path)
    with pytest.raises(SystemExit, match="not a full SHA"):
        ensure_source("t", tmp_path / "work", url, pin)
    assert not (tmp_path / "work" / "source").exists()


@_needs_git
def test_require_pinned_head_refuses_a_directory_that_is_not_a_checkout(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no git checkout"):
        require_pinned_head("t", tmp_path, "0" * 40)


@_needs_git
def test_robotwin_checkout_is_verified_on_reuse_not_only_on_clone(tmp_path: Path) -> None:
    """``ensure_source`` hands RoboTwin over and it is renamed out of ``source/``:
    the pin must still hold on every later provisioning run (#560)."""
    url, first, tip = _upstream(tmp_path)
    home = tmp_path / "home"
    root = _ensure_robotwin_checkout(home, url, first)
    assert root == home / "RoboTwin" and checkout_head(root) == first
    assert _ensure_robotwin_checkout(home, url, first) == root  # reuse at the pin
    # The provisioned root moved off the pin (a pull, an operator's checkout): refused.
    _git("fetch", "-q", "--depth", "1", "origin", tip, cwd=root)
    _git("checkout", "-q", "FETCH_HEAD", cwd=root)
    with pytest.raises(ROSConfigError, match="not the pinned"):
        _ensure_robotwin_checkout(home, url, first)
    # Not a git checkout at all (a copied tree): refused too.
    shutil.rmtree(root / ".git")
    with pytest.raises(ROSConfigError, match="no git checkout"):
        _ensure_robotwin_checkout(home, url, first)


@_needs_git
def test_lingbot_checkout_is_verified_on_reuse_not_only_on_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LingBot-VLA2 boot helper executes its ``<home>/source`` checkout: a reused
    checkout moved off the pin is refused like a fresh one (#560 for LingBot)."""
    lingbot = _tool("lingbot_vla2_sidecar")
    monkeypatch.delenv(lingbot._REPO_ENV, raising=False)
    url, first, tip = _upstream(tmp_path)
    home = tmp_path / "home"
    source = lingbot._ensure_source(home, url=url, sha=first)
    assert source == home / "source" and checkout_head(source) == first
    assert lingbot._ensure_source(home, url=url, sha=first) == source  # reuse at the pin
    _git("fetch", "-q", "--depth", "1", "origin", tip, cwd=source)
    _git("checkout", "-q", "FETCH_HEAD", cwd=source)
    with pytest.raises(SystemExit, match="not the pinned"):
        lingbot._ensure_source(home, url=url, sha=first)


@_needs_git
def test_every_launch_verifies_the_provisioned_robotwin_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_robotwin_root`` hands the checkout to the sidecar, which imports its code:
    the provisioned root is pin-checked there too, also when named through
    OPENRAL_ROBOTWIN_ROOT; an operator's own checkout elsewhere is theirs."""
    url, first, tip = _upstream(tmp_path)
    home = tmp_path / "home"
    root = _ensure_robotwin_checkout(home, url, tip)  # provisioned, but not at the pin
    (root / "assets" / "objects" / "objaverse").mkdir(parents=True)
    (root / "assets" / "objects" / "objaverse" / "list.json").write_text("[]")
    monkeypatch.setattr(robotwin, "_ROBOTWIN_SIDECAR_HOME", home)
    monkeypatch.setattr(robotwin, "_ROBOTWIN_SHA", first)
    monkeypatch.delenv(robotwin._ROBOTWIN_ROOT_ENV, raising=False)
    with pytest.raises(ROSConfigError, match="not the pinned"):
        robotwin._robotwin_root()
    monkeypatch.setenv(robotwin._ROBOTWIN_ROOT_ENV, str(root))
    with pytest.raises(ROSConfigError, match="not the pinned"):
        robotwin._robotwin_root()
    own = tmp_path / "operator"
    shutil.copytree(root, own)
    monkeypatch.setenv(robotwin._ROBOTWIN_ROOT_ENV, str(own))
    assert robotwin._robotwin_root() == own.resolve()


def test_rldx_install_plan_fetches_the_pin_and_verifies_head() -> None:
    clone = next(s for s in _rldx_sidecar_setup_plan().steps if "fetch" in s.description)
    script = clone.argv[-1]
    assert f"fetch -q --depth 1 origin {_RLDX_SIDECAR_REPO_SHA}" in script
    assert f'rev-parse HEAD)" = {_RLDX_SIDECAR_REPO_SHA} ]' in script
    assert " clone " not in script
    # Fetched beside the destination and moved in only when complete.
    assert 'part="$src.partial"' in script and 'mv "$part" "$src"' in script


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


def test_sidecar_env_drops_any_key_and_keeps_server_overrides() -> None:
    """OPENRAL_ACTOR_KEY is the compliance log's HMAC key; RLDX_ATTN_IMPL a documented opt-out."""
    env = sidecar_env(
        {"OPENRAL_ACTOR_KEY": "k", "RLDX_ATTN_IMPL": "sdpa", "TRANSFORMERS_OFFLINE": "1"}
    )
    assert env == {"RLDX_ATTN_IMPL": "sdpa", "TRANSFORMERS_OFFLINE": "1"}


@_needs_git
def test_run_sidecar_scrubs_before_the_clone_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RLDX / InternVLA boot through run_sidecar: the clone's `uv sync` sees no API key."""
    import argparse
    import os

    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    url, _, tip = _upstream(tmp_path)
    monkeypatch.setattr(os, "environ", {**os.environ, **_OPERATOR_SECRETS})
    seen: dict[str, str] = {}

    def install_deps(**_: object) -> Path:
        seen.update(os.environ)
        raise SystemExit("stop before exec")

    args = argparse.Namespace(home=tmp_path / "work", quantization="bf16")
    with pytest.raises(SystemExit, match="stop before exec"):
        run_sidecar(
            label="t",
            family="rldx",
            repo_url=url,
            repo_sha=tip,
            args=args,
            install_deps=install_deps,
            make_wrapper=lambda **_: tmp_path,
        )
    assert seen and not set(_OPERATOR_SECRETS) & set(seen)


def test_isolated_env_never_carries_the_reasoner_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_REASONER_API_KEY", "sk-openral")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    env = make_isolated_env(Path("/tmp/sidecar/.venv"))
    assert "OPENRAL_REASONER_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env["PATH"].startswith("/tmp/sidecar/.venv/bin")


_TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _tool(name: str) -> ModuleType:
    """Import ``tools/<name>.py`` (a script, not a package) as a fresh module."""
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", _TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    import os
    import sys

    environ = {"PATH": "/usr/bin", "HF_HOME": "/data/hf", "HF_TOKEN": "hf_x"}
    environ |= reads | _OPERATOR_SECRETS
    module = _tool(tool)
    monkeypatch.setattr(os, "environ", dict(environ))
    monkeypatch.setattr(sys, "argv", [tool, "--help"])
    with pytest.raises(SystemExit):
        module.main()
    capsys.readouterr()
    expected = {"PATH": "/usr/bin", "HF_HOME": "/data/hf", **reads}
    if keeps_hf_token:
        expected["HF_TOKEN"] = "hf_x"
    assert dict(os.environ) == expected


def test_sidecar_env_drops_urls_carrying_credentials() -> None:
    """A private index's user:password rides in the URL, not in a credential-named variable."""
    env = sidecar_env(
        {
            "UV_INDEX_URL": "https://bot:s3cret@pypi.internal.example/simple",
            "PIP_INDEX_URL": "https://pypi.org/simple",
            "HTTPS_PROXY": "http://user:pw@proxy.example:3128",
        }
    )
    assert env == {"PIP_INDEX_URL": "https://pypi.org/simple"}
