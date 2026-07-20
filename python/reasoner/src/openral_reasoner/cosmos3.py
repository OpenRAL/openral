"""NVIDIA Cosmos 3 Edge reasoner backend (``OPENRAL_REASONER_LLM_PROVIDER=cosmos``).

Cosmos 3 (released 2026; Edge tier 2026-07-20) is NVIDIA's omnimodal world-model
family built on a Mixture-of-Transformers architecture with two towers: an
autoregressive **reasoner** (text/image/video in → text out, physical reasoning,
task planning, 2D/3D grounding) and a diffusion **generator** (video/action out).
OpenRAL uses only the reasoner tower, served behind an OpenAI-compatible
chat-completions API — the exact surface :class:`OpenAICompatibleToolUseClient`
already speaks — so the S2 tool-call contract (provider tool-use API, no
free-form JSON, CLAUDE.md §3) is preserved while the planner itself becomes a
*physical-AI-native VLM running on-robot* (Jetson Thor / RTX), with no cloud
round-trip and no per-token cost.

Serving paths (any OpenAI-compatible endpoint works; the first is managed):

* **Managed local vLLM** (default) — :class:`Cosmos3ToolUseClient` probes
  ``OPENRAL_REASONER_LLM_BASE_URL`` (default ``http://127.0.0.1:8901/v1``) and,
  when the endpoint is loopback and down, auto-starts
  ``tools/cosmos3_reasoner_sidecar.py`` (uv-provisioned isolated venv, then
  ``vllm serve nvidia/Cosmos3-Edge`` with tool calling enabled). Same
  lazy-spawn/teardown lifecycle as the Qwen scene-VLM sidecar.
* **Self-managed vLLM / NIM** — point ``OPENRAL_REASONER_LLM_BASE_URL`` at an
  already-running ``vllm serve`` or a Cosmos 3 Reasoner NIM container and set
  ``OPENRAL_COSMOS3_AUTOSTART=0`` (autostart also disengages automatically for
  non-loopback URLs).

License: the Cosmos 3 model family ships under the Linux Foundation
**OpenMDW-1.1** license — commercial and non-commercial use permitted — so no
noncommercial guard (``OPENRAL_ALLOW_NONCOMMERCIAL``) applies here. This is a
*weights* license fact recorded per CLAUDE.md §1.9; OpenRAL's own code stays
Apache-2.0.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from openral_core.exceptions import ROSConfigError

from openral_reasoner.tool_use import (
    DEFAULT_SYSTEM_PROMPT,
    OpenAICompatibleToolUseClient,
)

if TYPE_CHECKING:
    from openral_core import ReasonerToolCall

    from openral_reasoner.palette import ToolPalette

__all__ = [
    "AUTOSTART_ENV",
    "BOOT_TIMEOUT_ENV",
    "COSMOS3_BASE_URL",
    "DEFAULT_BOOT_TIMEOUT_S",
    "DEFAULT_COSMOS3_MODEL",
    "SIDECAR_SCRIPT_ENV",
    "Cosmos3ToolUseClient",
    "find_cosmos3_sidecar_script",
]

# Managed local endpoint for the Cosmos 3 reasoner. A dedicated port (not
# vLLM's :8000 default) so the managed sidecar never collides with a generic
# user-run `vllm serve` covered by the `vllm` provider preset.
COSMOS3_BASE_URL: str = "http://127.0.0.1:8901/v1"

# Default checkpoint: the 4B Edge tier — the on-device (Jetson Thor / RTX)
# member of the Cosmos 3 family and the reason this provider exists. Any
# Cosmos 3 tier works via OPENRAL_REASONER_LLM_MODEL (`nvidia/Cosmos3-Nano`
# for workstation-grade quality, `nvidia/Cosmos3-Super` for datacenter).
DEFAULT_COSMOS3_MODEL: str = "nvidia/Cosmos3-Edge"

# Env knobs (all optional). AUTOSTART defaults on for loopback endpoints;
# BOOT_TIMEOUT covers the first-boot worst case: venv provisioning (vLLM +
# CUDA wheels) plus the ~8 GB BF16 weight download from HF Hub.
AUTOSTART_ENV: str = "OPENRAL_COSMOS3_AUTOSTART"
BOOT_TIMEOUT_ENV: str = "OPENRAL_COSMOS3_BOOT_TIMEOUT_S"
SIDECAR_SCRIPT_ENV: str = "OPENRAL_COSMOS3_SIDECAR"
DEFAULT_BOOT_TIMEOUT_S: float = 1800.0

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def find_cosmos3_sidecar_script() -> Path:
    """Locate ``tools/cosmos3_reasoner_sidecar.py`` (env override or repo walk).

    Mirrors the Qwen scene-VLM backend's resolution: an explicit
    ``OPENRAL_COSMOS3_SIDECAR`` wins; otherwise walk this file's parents
    looking for the in-tree script.

    Raises:
        ROSConfigError: When the script cannot be found (e.g. an installed
            wheel without the repo checkout) and no override is set.
    """
    override = os.environ.get(SIDECAR_SCRIPT_ENV)
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        cand = parent / "tools" / "cosmos3_reasoner_sidecar.py"
        if cand.exists():
            return cand
    raise ROSConfigError(
        f"could not locate tools/cosmos3_reasoner_sidecar.py; set {SIDECAR_SCRIPT_ENV} to its path"
    )


def _base_url_is_loopback(base_url: str) -> bool:
    """True when ``base_url``'s host is a loopback name (autostart territory)."""
    host = urllib.parse.urlparse(base_url).hostname or ""
    return host in _LOOPBACK_HOSTS


def _endpoint_is_up(base_url: str, *, timeout_s: float = 2.0) -> bool:
    """Probe ``GET <base_url>/models`` — any HTTP response means the server is up.

    vLLM binds its port only after "Application startup complete" (model
    loaded), so a successful HTTP round-trip implies readiness, not just a
    listening socket. An HTTP error status (401 from ``--api-key``, 404 from
    a shim) still proves a live server, so it counts as up.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        with contextlib.closing(
            urllib.request.urlopen(url, timeout=timeout_s)
        ):  # reason: loopback/deployment-configured HTTP endpoint probe, no user-supplied scheme
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


class Cosmos3ToolUseClient(OpenAICompatibleToolUseClient):
    """:class:`ToolUseClient` for the Cosmos 3 reasoner tower with managed serving.

    The wire path is exactly :class:`OpenAICompatibleToolUseClient` (openai SDK
    ``tools`` / ``tool_calls``, ``tool_choice="required"``, image ``data:`` URIs
    for :meth:`describe_image`); what this subclass adds is the **managed local
    server lifecycle**: before the first call it probes the endpoint and, when
    the endpoint is loopback, down, and ``auto_start`` is on, spawns
    ``tools/cosmos3_reasoner_sidecar.py`` and waits for readiness. A child we
    spawned is terminated at interpreter exit (never a server someone else
    started — same contract as the Qwen scene-VLM sidecar client).

    Args:
        model_id: Cosmos 3 checkpoint id served by the endpoint. Defaults to
            :data:`DEFAULT_COSMOS3_MODEL` (``nvidia/Cosmos3-Edge``).
        api_key: Only needed for an endpoint fronted by auth
            (``vllm serve --api-key`` / a gateway); the managed local server
            enforces none.
        base_url: OpenAI-compatible endpoint. Defaults to
            :data:`COSMOS3_BASE_URL` (``http://127.0.0.1:8901/v1``).
        timeout_s: Per-call wall-clock timeout. Default 120 s — an on-device
            4B VLM's first inference after boot compiles kernels and is much
            slower than steady state.
        max_tokens: Optional completion-token cap (see base class).
        auto_start: Spawn the managed sidecar when the loopback endpoint is
            down. Defaults ``True``; forced off for non-loopback URLs.
        boot_timeout_s: How long to wait for the spawned server to become
            ready. First boot provisions the venv and downloads weights —
            keep this generous (default :data:`DEFAULT_BOOT_TIMEOUT_S`).

    Example:
        >>> client = Cosmos3ToolUseClient(auto_start=False)
        >>> client.model_id
        'nvidia/Cosmos3-Edge'
        >>> client._base_url
        'http://127.0.0.1:8901/v1'
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_COSMOS3_MODEL,
        api_key: str | None = None,
        base_url: str = COSMOS3_BASE_URL,
        timeout_s: float = 120.0,
        max_tokens: int | None = None,
        auto_start: bool = True,
        boot_timeout_s: float = DEFAULT_BOOT_TIMEOUT_S,
    ) -> None:
        """Stash config; no probe, no spawn, no SDK import until first use."""
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )
        self._auto_start = auto_start and _base_url_is_loopback(base_url)
        self._boot_timeout_s = boot_timeout_s
        self._server_ready = False
        self._child: subprocess.Popen[bytes] | None = None

    # -- managed server lifecycle -------------------------------------------

    def _spawn_argv(self) -> list[str]:
        """Build the sidecar launch argv (split out for unit-testability)."""
        script = find_cosmos3_sidecar_script()
        parsed = urllib.parse.urlparse(self._base_url or COSMOS3_BASE_URL)
        return [
            sys.executable,
            str(script),
            "--model",
            self.model_id,
            "--host",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(parsed.port or 8901),
        ]

    def _ensure_server(self) -> None:
        """Probe the endpoint; auto-start the managed sidecar when appropriate.

        Raises:
            ROSConfigError: When the endpoint is down, autostart is off (or
                the URL is non-loopback), or the spawned server exits early /
                misses ``boot_timeout_s``.
        """
        if self._server_ready:
            return
        base_url = self._base_url or COSMOS3_BASE_URL
        if _endpoint_is_up(base_url):
            self._server_ready = True
            return
        if not self._auto_start:
            raise ROSConfigError(
                f"no Cosmos 3 reasoner endpoint at {base_url} and autostart is "
                f"disabled ({AUTOSTART_ENV}=0 or non-loopback URL); start one with "
                "`python tools/cosmos3_reasoner_sidecar.py` (managed vLLM) or point "
                "OPENRAL_REASONER_LLM_BASE_URL at a running vLLM / Cosmos 3 "
                "Reasoner NIM endpoint."
            )
        argv = self._spawn_argv()
        print(f"[cosmos3] spawning reasoner server: {' '.join(argv)}", flush=True)
        self._child = subprocess.Popen(argv)
        atexit.register(self.close)
        deadline = time.monotonic() + self._boot_timeout_s
        while time.monotonic() < deadline:
            if self._child.poll() is not None:
                raise ROSConfigError(
                    f"Cosmos 3 reasoner sidecar exited early (code {self._child.returncode}); "
                    "run `python tools/cosmos3_reasoner_sidecar.py` in a terminal to see why "
                    "(common causes: no NVIDIA GPU, CUDA driver too old, HF auth missing)."
                )
            if _endpoint_is_up(base_url):
                print("[cosmos3] reasoner server ready", flush=True)
                self._server_ready = True
                return
            time.sleep(2.0)
        raise ROSConfigError(
            f"Cosmos 3 reasoner server not ready within {self._boot_timeout_s:.0f}s "
            f"(first boot provisions a venv and downloads ~8 GB of weights — "
            f"raise {BOOT_TIMEOUT_ENV} if that is still in progress)."
        )

    def close(self) -> None:
        """Terminate the server child if (and only if) we spawned it."""
        child = self._child
        if child is not None and child.poll() is None:
            try:
                child.terminate()
                child.wait(timeout=10)
            except Exception:  # reason: best-effort teardown at exit; escalate to kill
                child.kill()
        self._child = None
        self._server_ready = False

    # -- ToolUseClient ------------------------------------------------------

    def select_tool(
        self,
        *,
        context_text: str,
        palette: ToolPalette,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> ReasonerToolCall:
        """Ensure the managed server is up, then run the OpenAI-compatible path."""
        self._ensure_server()
        return super().select_tool(
            context_text=context_text,
            palette=palette,
            system_prompt=system_prompt,
        )

    def describe_image(self, *, image_jpeg: bytes, question: str) -> str:
        """Ensure the managed server is up, then run the OpenAI-compatible path."""
        self._ensure_server()
        return super().describe_image(image_jpeg=image_jpeg, question=question)
