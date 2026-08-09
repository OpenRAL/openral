"""Boot the DA3 metric-depth inference server in an isolated sidecar venv.

`depth-anything/DA3-SMALL` ships as the `depth-anything-3` package
(not transformers-native), so we run it out-of-process in its own Python 3.12
venv and talk to it over ZMQ REQ/REP + msgpack from the
`openral_perception_ros` depth-provider node — the same pattern as
:mod:`tools.locateanything_sidecar`. The provider republishes the result as a
`32FC1` depth Image + CameraInfo for nvblox, giving lidar-less robots a Nav2
cost map.

Measured on an 8 GB Ada (RTX 4070 Laptop): DA3-SMALL loads in ~5 s, ~0.27 GB
peak, ~27 Hz — comfortably real-time for nvblox's depth integration.

Usage::

    python tools/da3_depth_sidecar.py --port 5771

CLAUDE.md compliance:
* Real subprocess running real upstream model code — no mocks (§1.11).
* Version isolation is the bridge between the `depth-anything-3` package and the
  transformers-5.x runtime venv (§3).
* DA3 weights keep their upstream license — verify per checkpoint (§9); no
  license guard is enforced here.
"""

from __future__ import annotations

import argparse

# Load the (pure-stdlib) sidecar helpers by file path instead of
# `from openral_sim._sidecar_common import …`: the package import executes
# `openral_sim.__init__` → full backend registration → the whole sim stack,
# which the deploy launch's autostart interpreter (`slam_depth_sidecar_autostart`
# spawns this under plain python3) does not have. This launcher stays runnable
# by ANY python3 — its only non-stdlib need is `uv` on PATH for provisioning.
import importlib.util as _importlib_util
import os
import sys
from collections.abc import Callable
from pathlib import Path

_common_path = Path(__file__).parent.parent / "python/sim/src/openral_sim/_sidecar_common.py"
_spec = _importlib_util.spec_from_file_location("_sidecar_common", _common_path)
assert _spec is not None and _spec.loader is not None  # stdlib file ships with the repo
_common = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
ensure_pip_venv: Callable[..., Path] = _common.ensure_pip_venv
run_cmd: Callable[..., None] = _common.run_cmd

_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "da3-depth-sidecar"
_VENV_ENV = "OPENRAL_DA3_DEPTH_SIDECAR_VENV"
_HOME_ENV = "OPENRAL_DA3_DEPTH_SIDECAR_HOME"

# Pinned dependency source. A fully hash-locked `.lock` (uv pip compile
# --generate-hashes) is the reproducibility target (CLAUDE.md §1.8); until it is
# generated, install the package set verified working on the 8 GB Ada host.
_REQUIREMENTS = ("depth-anything-3", "pyzmq", "msgpack")


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it if needed.

    ``override`` (or ``$OPENRAL_DA3_DEPTH_SIDECAR_VENV``) reuses an existing venv
    (e.g. a dev `depth-anything-3` env) instead of provisioning one under
    ``home``.
    """

    def _install(uv: str, py: Path) -> None:
        run_cmd(
            "da3-sidecar",
            [uv, "pip", "install", "--python", str(py), "--torch-backend=cu128", *_REQUIREMENTS],
        )

    return ensure_pip_venv(
        label="da3-sidecar",
        home=home,
        python="3.12",
        install=_install,
        override=override,
        override_env=_VENV_ENV,
        # Keyed on the pins so editing them repairs an existing venv.
        spec=_REQUIREMENTS,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="depth-anything/DA3-SMALL")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5771)
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument(
        "--home",
        type=Path,
        default=Path(os.environ.get(_HOME_ENV, _DEFAULT_HOME)),
        help=f"Sidecar work directory (default {_DEFAULT_HOME}).",
    )
    p.add_argument("--venv", default=None, help=f"Reuse this venv (or set {_VENV_ENV}).")
    args = p.parse_args()

    py = ensure_venv(args.home, override=args.venv)
    server = Path(__file__).resolve().parent / "_da3_depth_server.py"

    env = os.environ.copy()
    # Drop PYTHONPATH/PYTHONHOME so the sidecar boots from its own site-packages
    # (ROS 2 / colcon populate PYTHONPATH with the workspace wheels, which would
    # shadow the sidecar's deps) — same as tools/locateanything_sidecar.py.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)

    cmd = [
        str(py),
        str(server),
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--process-res",
        str(args.process_res),
    ]
    print(f"[da3-sidecar] launching server: model={args.model} port={args.port}", flush=True)
    os.execvpe(str(py), cmd, env)


if __name__ == "__main__":
    sys.exit(main() or 0)
