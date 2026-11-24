"""Boot the Depth Pro 2 metric-depth server in an isolated sidecar venv.

Depth Pro 2 ships as the ``depth_pro_2`` package from Apple's GitHub repository (not on
PyPI, not transformers-native), so it runs out-of-process in its own Python 3.12 venv and
answers the ``openral_perception_ros`` depth provider over ZMQ with the DA3 sidecar's
wire protocol (``tools/_depth_pro2_server.py``).

Licensing: the code is under Apple's sample-code license and is installed, never
vendored; the weights (``apple/DepthPro2-*``) are under the Apple Machine Learning
Research Model license, research use only. Start this through
``tools/depth_estimator_sidecar.py --manifest <rskill.yaml>``, which applies the
manifest's license guard (``OPENRAL_ALLOW_NONCOMMERCIAL=1``) before anything loads.

Usage::

    python tools/depth_pro2_sidecar.py --model apple/DepthPro2-S --revision <sha> --port 5772
"""

from __future__ import annotations

import argparse
import importlib.util as _importlib_util
import os
import sys
from collections.abc import Callable
from pathlib import Path

# File-path import of the shared helpers, as tools/da3_depth_sidecar.py does (the
# openral_sim package import would pull the whole sim stack).
_common_path = Path(__file__).parent.parent / "python/sim/src/openral_sim/_sidecar_common.py"
_spec = _importlib_util.spec_from_file_location("_sidecar_common", _common_path)
assert _spec is not None and _spec.loader is not None  # helper file ships with the repo
_common = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
ensure_pip_venv: Callable[..., Path] = _common.ensure_pip_venv
run_cmd: Callable[..., None] = _common.run_cmd
torch_backend: Callable[[], str] = _common.torch_backend
scrub_environ: Callable[..., None] = _common.scrub_environ

_DEFAULT_HOME = Path.home() / ".cache" / "openral" / "depth-pro2-sidecar"
_VENV_ENV = "OPENRAL_DEPTH_PRO2_SIDECAR_VENV"
_HOME_ENV = "OPENRAL_DEPTH_PRO2_SIDECAR_HOME"

#: ``ml-depth-pro-2`` commit the package is installed from (pinned, CLAUDE.md §1.8).
DEPTH_PRO2_SHA = "aa389cbb0b74341664ed663a0966c859905cd6c0"
_REQUIREMENTS = (
    f"depth_pro_2 @ git+https://github.com/apple-aiml-research/ml-depth-pro-2@{DEPTH_PRO2_SHA}",
    "pyzmq",
    "msgpack",
)


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it if needed."""

    def _install(uv: str, py: Path) -> None:
        # --no-config: keep the repo root's [tool.uv] torch overrides out of this venv.
        run_cmd(
            "depth-pro2-sidecar",
            [
                uv,
                "pip",
                "install",
                "--no-config",
                "--python",
                str(py),
                f"--torch-backend={torch_backend()}",
                *_REQUIREMENTS,
            ],
        )

    import fcntl

    home.mkdir(parents=True, exist_ok=True)
    with open(home / ".provision.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return ensure_pip_venv(
            label="depth-pro2-sidecar",
            home=home,
            python="3.12",
            install=_install,
            override=override,
            override_env=_VENV_ENV,
            spec=_REQUIREMENTS,
        )


def main() -> int:
    # The weights are public: no operator credential, HF_TOKEN included, reaches the venv.
    scrub_environ(hf_token=False)
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="HF repo, e.g. apple/DepthPro2-S")
    p.add_argument("--revision", default=None, help="Pinned Hub commit of --model.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5772)
    p.add_argument("--home", type=Path, default=Path(os.environ.get(_HOME_ENV, _DEFAULT_HOME)))
    p.add_argument("--venv", default=None, help=f"Reuse this venv (or set {_VENV_ENV}).")
    args = p.parse_args()

    py = ensure_venv(args.home, override=args.venv)
    server = Path(__file__).resolve().parent / "_depth_pro2_server.py"
    env = os.environ.copy()
    # ROS / colcon PYTHONPATH would shadow the venv's own site-packages.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    cmd = [str(py), str(server), "--model", args.model, "--host", args.host]
    cmd += ["--port", str(args.port)]
    if args.revision:
        cmd += ["--revision", args.revision]
    print(f"[depth-pro2-sidecar] launching server: model={args.model} port={args.port}")
    os.execvpe(str(py), cmd, env)


if __name__ == "__main__":
    sys.exit(main() or 0)
