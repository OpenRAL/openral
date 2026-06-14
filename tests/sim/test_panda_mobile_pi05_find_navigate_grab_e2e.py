"""Live deploy-sim e2e — reasoner dispatches a mobile-manip VLA at the baguette.

The "find → navigate → grab" payoff for the ADR-0044 ladder, on the real
robocasa kitchen digital twin. Brings up the **full** ``openral deploy sim``
graph (panda_mobile HAL + robocasa kitchen with a deterministic baguette, SLAM,
Nav2, the safety kernel, the reasoner, the skill_runner), then dispatches the
``pi05-robocasa365`` mobile-manipulation VLA (whose 12-D action drives the base
*and* the arm — navigate + grab in one policy) at the baguette task and asserts
the **safety-gated actuation loop** runs end to end:

* the VLA loads on the GPU and produces real robocasa actions (``policy_step``),
  driving the base/arm toward the baguette, AND
* the C++ safety kernel gates the chunk stream and E-stops on a geometric
  collision — the goal terminates via ``safety_estop`` rather than silently.

This deliberately asserts the *attempt under safety oversight*, not a completed
grasp: deploy-sim does not publish the object pose, and pi05's trajectory trips
the kernel's self-collision check before the gripper closes (a real check, not a
false positive). What it proves is that every layer — dispatch, VLA inference,
state assembly, the safety kernel, the HAL, the sim — is wired and works
together on the actuation path.

Heavily gated (CLAUDE.md §1.11/§12): needs a GPU + the ``robocasa`` extras + a
sourced ROS 2 workspace with the deploy-sim ROS packages colcon-built, and the
opt-in ``OPENRAL_E2E_ROBOCASA=1`` (it loads a ~5 GB VLA and takes minutes).
Never faked — it skips when any prerequisite is absent.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_MODULES = ("torch", "transformers", "robocasa", "robosuite")
_MISSING = tuple(m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None)


def _gpu_present() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


_SKIP_REASON = ""
if not os.environ.get("OPENRAL_E2E_ROBOCASA"):
    _SKIP_REASON = (
        "opt-in only: set OPENRAL_E2E_ROBOCASA=1 to run the live robocasa "
        "find→navigate→grab e2e (loads a ~5 GB VLA, takes minutes)."
    )
elif not os.environ.get("ROS_DISTRO"):
    _SKIP_REASON = "ROS_DISTRO not set — needs a sourced ROS 2 workspace."
elif _MISSING:
    _SKIP_REASON = f"missing robocasa extras: {', '.join(_MISSING)} (just sync --group robocasa)."
elif not _gpu_present():
    _SKIP_REASON = "no CUDA GPU — pi05 needs one."

pytestmark = pytest.mark.skipif(bool(_SKIP_REASON), reason=_SKIP_REASON)

_DOMAIN = os.environ.get("OPENRAL_E2E_ROS_DOMAIN_ID", "47")
# The dispatch node (this pytest process's rclpy) MUST share the graph's ROS
# domain or it never discovers the action server. Pin it before rclpy.init.
os.environ["ROS_DOMAIN_ID"] = _DOMAIN
_SCENE = _REPO_ROOT / "scenes" / "deploy" / "robocasa_baguette.yaml"
_PI05_ID = "OpenRAL/rskill-pi05-robocasa365-human300-nf4"
_BAGUETTE_PROMPT = "pick the baguette from the counter and place it in the cabinet"


def _graph_env() -> dict[str, str]:
    """The exact env the live graph needs (the hard-won deploy-sim recipe)."""
    env = dict(os.environ)
    # Miniforge's python3.13 contaminates the rosidl C typesupport link; strip it.
    env["PATH"] = ":".join(p for p in env.get("PATH", "").split(":") if "miniforge" not in p)
    venv_site = _REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
    src_dirs = ":".join(str(p) for p in sorted((_REPO_ROOT / "python").glob("*/src")))
    env["PYTHONPATH"] = ":".join(
        x for x in (src_dirs, str(venv_site), env.get("PYTHONPATH", "")) if x
    )
    env["MUJOCO_GL"] = "egl"
    env["ROS_DOMAIN_ID"] = _DOMAIN
    env["OPENRAL_AUTO_INSTALL_DEPS"] = "1"  # HAL's robocasa "install on first use" must not prompt
    # 8 GB-card headroom: pi05 (~4.3 GB) + the kitchen EGL render + the first
    # forward-pass activation spike can tip over without an expandable allocator.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # The reasoner LLM is irrelevant here (we dispatch the skill directly), but a
    # provider must be set or configure warns; ollama is the lab default.
    env.setdefault("OPENRAL_REASONER_LLM_PROVIDER", "ollama")
    env.setdefault("OPENRAL_REASONER_LLM_MODEL", "gemma4:31b-cloud")
    return env


@pytest.fixture(scope="module")
def deploy_sim_graph(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Launch the robocasa-baguette deploy-sim graph; yield its captured log path."""
    log_path = tmp_path_factory.mktemp("e2e") / "graph.log"
    openral = _REPO_ROOT / ".venv" / "bin" / "openral"
    if not openral.exists():
        pytest.skip(f"worktree venv openral entrypoint missing: {openral}")
    cmd = [
        str(openral),
        "deploy",
        "sim",
        "--config",
        str(_SCENE),
        "--no-dashboard",
        # The octomap world-voxel leg false-positives in a dense kitchen (the arm
        # starts ~3 mm inside a counter voxel); envelope + self-collision remain.
        "--no-enable-octomap",
        # Headless: drop the MuJoCo passive-viewer window so its EGL render
        # buffers don't compete with pi05 for the 8 GB card.
        "--hal",
        "viewer_enabled=false",
    ]
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO_ROOT),
            env=_graph_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group so teardown kills the whole tree
        )
    try:
        _wait_for_ready(proc, log_path)
        yield log_path
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _wait_for_ready(proc: subprocess.Popen[bytes], log_path: Path) -> None:
    """Block until the HAL is active, the reasoner is ticking, and pi05 is admitted.

    robocasa builds inside the HAL ``on_configure`` (slow first time), so the
    readiness budget is generous.
    """
    deadline = time.monotonic() + 420.0
    need = (
        re.compile(r"HAL activated"),
        re.compile(r"on_activate: ticking"),
        re.compile(rf"admitting rSkill '{re.escape(_PI05_ID)}'"),
    )
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"deploy-sim graph exited early (rc={proc.returncode}); see {log_path}")
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        if all(p.search(text) for p in need):
            time.sleep(3.0)  # let the action server + skill_runner finish wiring
            return
        time.sleep(2.0)
    pytest.fail(f"graph not ready within budget; see {log_path}")


def test_pi05_drives_toward_baguette_under_safety_oversight(deploy_sim_graph: Path) -> None:
    """Dispatch pi05 at the baguette; assert the VLA drives the robot and the kernel gates it."""
    import rclpy
    from openral_msgs.action import ExecuteRskill
    from rclpy.action import ActionClient

    rclpy.init()
    node = rclpy.create_node("e2e_pi05_dispatch")
    try:
        client: ActionClient = ActionClient(node, ExecuteRskill, "/openral/execute_rskill")
        assert client.wait_for_server(timeout_sec=30.0), "/openral/execute_rskill never advertised"

        goal = ExecuteRskill.Goal()
        goal.rskill_id = _PI05_ID
        goal.prompt = _BAGUETTE_PROMPT
        goal.deadline_s = 180.0

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send_future, timeout_sec=60.0)
        handle = send_future.result()
        assert handle is not None and handle.accepted, "pi05 goal was not accepted"

        result_future = handle.get_result_async()
        # pi05 warms up (~20 s) then runs its horizon until the safety kernel E-stops.
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=240.0)
        wrapper = result_future.result()
        assert wrapper is not None, "no result from the pi05 goal within budget"
        result = wrapper.result
    finally:
        node.destroy_node()
        rclpy.shutdown()

    log = deploy_sim_graph.read_text(errors="replace")

    # 1. The VLA loaded and produced a real robocasa action — a 12-D mobile-manip
    #    vector (arm OSC + gripper + base) flowed onto the actuation path. (How
    #    far it drives before the kernel stops it is non-deterministic: pi05 is a
    #    flow-matching policy, so it samples a different trajectory each run from
    #    the same start state — anywhere from step 1 to a hundred. We assert the
    #    VLA *ran*, not a fixed horizon.)
    assert re.search(r"policy_step step=\d+", log), (
        "pi05 produced no policy_step — the VLA never produced an action"
    )

    # 2. The C++ safety kernel gated that actuation and E-stopped on a geometric
    #    collision (here self-collision) — i.e. the full safety-gated loop ran:
    #    dispatch → VLA → safety kernel → HAL. The goal terminated under safety
    #    oversight, not via a silent success or a crash.
    assert re.search(r"safety\.collision kind=(self|world)", log), (
        "expected the safety kernel to flag a geometric collision on pi05's actions"
    )
    assert "estop_received" in log, "expected the E-stop to propagate to the skill_runner / HAL"
    assert not result.success, f"expected a safety-gated abort, got success={result.success}"
    assert "safety_estop" in (result.failure_reason or ""), (
        f"expected a safety_estop failure_reason, got {result.failure_reason!r}"
    )
