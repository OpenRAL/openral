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

On aarch64 CUDA hosts (GB10 / DGX Spark, Jetson Thor) the install recipe is
different — see :data:`_AARCH64_REQUIREMENTS` and
:func:`_defer_pycolmap_import`, and docs/reference/aarch64-support.md.

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
import platform
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

# aarch64 (GB10 / DGX Spark, Jetson Thor) needs a hand-rolled install instead:
# two of `depth-anything-3`'s declared dependencies publish **no** linux-aarch64
# wheel on any index, so the one-line `uv pip install depth-anything-3` above is
# unsatisfiable there.
#
#   open3d   — 0.19.0 (the only cp312 release) is manylinux_2_31_x86_64 / macOS /
#              win_amd64; the aarch64 builds stop at 0.18.0 / cp311. Imported
#              *only* by `depth_anything_3.bench.*` (the DTU / ETH3D /
#              ScanNet++ / 7-Scenes evaluators), never by the inference path, so
#              it is simply dropped.
#   pycolmap — every release ever published is x86_64-linux / macOS-arm64 /
#              win_amd64, with no sdist at all; conda-forge has a linux-aarch64
#              build but there is no pip route to it. Unlike open3d it *is* on
#              the import path — `api.py` eagerly pulls in
#              `depth_anything_3.utils.export`, whose `__init__` imports the
#              COLMAP writer — so it needs :func:`_defer_pycolmap_import`.
#
# The pin set below is upstream's `pyproject.toml` dependency list minus those
# two, minus `xformers` (also x86_64/win-only, and already optional upstream:
# `dinov2/layers/swiglu_ffn.py` falls back to a pure-torch SwiGLU when it is
# absent) and minus `pre-commit` (an upstream packaging slip — a dev tool), plus
# `addict` (an *undeclared* upstream dependency of
# `depth_anything_3/model/da3.py`, normally pulled in transitively), plus the
# repo's standard aarch64 torch pins and the ZMQ/msgpack wire deps.
#
# torch 2.9.1, not whatever `torch>=2` resolves to: the cu128 index publishes no
# aarch64 wheel for 2.8.0 and its required triton==3.4.0 has none either — the
# same pin the LocateAnything and LingBot sidecars carry. See
# docs/reference/aarch64-support.md.
#
# Raises nvrtc past the sm_121 ceiling; shared with the other sidecars, see that
# file's header. Load-bearing here: DA3's `@torch.jit.script affine_inverse` is
# NVRTC-fused by the TensorExpr fuser once the profiling executor warms up, so
# without this the FIRST depth request succeeds and every later one dies — the
# worst possible failure shape for a provider streaming frames.
_NVRTC_OVERRIDE = (
    Path(__file__).resolve().parent / "sidecar_requirements" / "aarch64-nvrtc-override.txt"
)
_AARCH64_REQUIREMENTS = (
    "torch==2.9.1",
    "torchvision==0.24.1",
    "addict",
    "e3nn",
    "einops",
    "evo",
    "fastapi",
    "huggingface-hub",
    "imageio",
    "moviepy==1.0.3",
    "numpy<2",
    "omegaconf",
    "opencv-python",
    "pillow",
    "pillow-heif",
    "plyfile",
    "requests",
    "safetensors",
    "trimesh",
    "typer>=0.9.0",
    "uvicorn",
    "pyzmq",
    "msgpack",
)
# Pinned exactly because :func:`_defer_pycolmap_import` is anchored to this
# release's source text; a silent upgrade must not silently skip the rewrite.
_AARCH64_DA3 = "depth-anything-3==0.1.1"

# The module-level statement in `depth_anything_3/utils/export/colmap.py` that
# makes `import depth_anything_3.api` fail on aarch64, and the deferred proxy we
# swap it for. This is a *deferred import*, not a stub (CLAUDE.md §1.11): every
# `pycolmap.<name>` in that file is an attribute read inside `export_to_colmap`,
# so exporting to COLMAP still raises the real `ModuleNotFoundError: pycolmap` —
# it just raises it when that export format is requested rather than when the
# depth model is imported.
_PYCOLMAP_EAGER_IMPORT = "\nimport pycolmap\n"
_PYCOLMAP_LAZY_IMPORT = '''
class _LazyPycolmap:
    """Deferred `pycolmap` proxy, patched in by tools/da3_depth_sidecar.py.

    `pycolmap` has no linux-aarch64 wheel, and this module is imported eagerly
    by `depth_anything_3.api`. Attribute access performs the real import, so
    `export_to_colmap` fails exactly as it would without this shim.
    """

    def __getattr__(self, name):
        import pycolmap

        return getattr(pycolmap, name)


pycolmap = _LazyPycolmap()
'''


def _site_packages(py: Path) -> Path:
    """Return the ``site-packages`` directory of the venv owning ``py``."""
    candidates = sorted((py.parent.parent / "lib").glob("python*/site-packages"))
    if not candidates:
        raise SystemExit(f"da3-sidecar: no site-packages under {py.parent.parent}")
    return candidates[0]


def _defer_pycolmap_import(py: Path) -> None:
    """Rewrite DA3's eager ``import pycolmap`` into a deferred proxy (aarch64 only).

    Idempotent: re-running against an already-patched venv is a no-op. Raises
    ``SystemExit`` rather than patching blind if the anchor statement is not
    found exactly once — an upstream refactor must surface, not be swallowed.
    """
    target = _site_packages(py) / "depth_anything_3" / "utils" / "export" / "colmap.py"
    source = target.read_text(encoding="utf-8")
    if _PYCOLMAP_LAZY_IMPORT in source:
        return
    if source.count(_PYCOLMAP_EAGER_IMPORT) != 1:
        raise SystemExit(
            f"da3-sidecar: expected exactly one 'import pycolmap' in {target}; "
            f"found {source.count(_PYCOLMAP_EAGER_IMPORT)}. Upstream "
            f"{_AARCH64_DA3} changed — re-check the aarch64 recipe."
        )
    # Unlink first: `uv pip install` hardlinks wheel contents out of its shared
    # cache, so writing in place would corrupt the cached copy for every other
    # venv on the host.
    target.unlink()
    target.write_text(
        source.replace(_PYCOLMAP_EAGER_IMPORT, _PYCOLMAP_LAZY_IMPORT), encoding="utf-8"
    )
    print(f"[da3-sidecar] deferred the pycolmap import in {target}", flush=True)


def ensure_venv(home: Path, *, override: str | None = None) -> Path:
    """Return the sidecar venv python, creating + populating it if needed.

    ``override`` (or ``$OPENRAL_DA3_DEPTH_SIDECAR_VENV``) reuses an existing venv
    (e.g. a dev `depth-anything-3` env) instead of provisioning one under
    ``home``.
    """
    aarch64 = platform.machine() == "aarch64"
    spec = (
        (*_AARCH64_REQUIREMENTS, _AARCH64_DA3, _NVRTC_OVERRIDE.read_text(encoding="utf-8"))
        if aarch64
        else _REQUIREMENTS
    )

    def _install(uv: str, py: Path) -> None:
        pip = [uv, "pip", "install", "--python", str(py), "--torch-backend=cu128"]
        if not aarch64:
            run_cmd("da3-sidecar", [*pip, *_REQUIREMENTS])
            return
        # The nvrtc override is what lets DA3 keep TorchScript ON here: its
        # `@torch.jit.script affine_inverse` gets NVRTC-fused once the profiling
        # executor warms up, and the stock cu128 nvrtc 12.8 cannot target
        # sm_121 — so request #1 succeeded and every later one died. Raising
        # nvrtc fixes the codegen itself rather than disabling the fuser.
        pip += ["--overrides", str(_NVRTC_OVERRIDE)]
        run_cmd("da3-sidecar", [*pip, *_AARCH64_REQUIREMENTS])
        # --no-deps: the dependency set above is deliberately not upstream's.
        run_cmd("da3-sidecar", [*pip, "--no-deps", _AARCH64_DA3])
        _defer_pycolmap_import(py)

    return ensure_pip_venv(
        label="da3-sidecar",
        home=home,
        python="3.12",
        install=_install,
        override=override,
        override_env=_VENV_ENV,
        # Keyed on the pins so editing them repairs an existing venv.
        spec=spec,
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
