"""Verify selected optional test environments locally.

Runs the strict dependency lanes declared in ``tools/test_selection.toml``:
sync the matching dependency group, run its selected targets, and fail if any
test skips. Hardware and externally provisioned sidecars are opt-in because this
script cannot create physical devices or proprietary simulator installs.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "tools" / "test_selection.toml"

PROVISIONED_GROUPS = {
    "isaacsim": "OPENRAL_ISAAC_SIDECAR_PYTHON",
    "robocasa-gr1": "active robocasa GR1 fork",
    "robotwin": "OPENRAL_ROBOTWIN_SIDECAR_PYTHON",
    "rlbench": "OPENRAL_RLBENCH_SIDECAR_PYTHON and COPPELIASIM_ROOT",
    "locateanything": "OPENRAL_LOCATEANYTHING_SIDECAR_VENV",
    "qwen-vlm": "OPENRAL_QWEN_VLM_SIDECAR_VENV",
    "simpler-env": "manual simpler-env package install",
}

LANE_EXTRA_GROUPS = {
    "sim": ["dataset"],
}

LANE_DEPENDENCY_GROUP = {
    "robocasa-gr1": "robocasa",
}

MANUAL_INSTALL_GROUPS = {"robocasa-gr1", "simpler-env"}
ONNX_EXPORT = REPO_ROOT / "rskills" / "rtdetr-coco-r18" / "model.onnx"
SIMPLER_ENV_SPEC = "simpler-env @ git+https://github.com/simpler-env/SimplerEnv.git@maniskill3"
ROBOCASA_GR1_REPO = "https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git"
ROBOCASA_GR1_DIR = Path("/tmp/rg1")
LOCATEANYTHING_VENV = Path.home() / ".cache/openral/locateanything-sidecar/.venv"
QWEN_VLM_VENV = Path.home() / ".cache/openral/qwen-vlm-sidecar/.venv"
RLBENCH_HOME = Path.home() / ".cache/openral/rlbench-policy"
RLBENCH_VENV = RLBENCH_HOME / ".venv"
RLBENCH_PYTHON = RLBENCH_VENV / "bin" / "python"
COPPELIASIM_ROOT = Path.home() / ".cache/openral/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
COPPELIASIM_ARCHIVE = (
    Path.home() / ".cache/openral/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
)
COPPELIASIM_URL = (
    "https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
)
POLICY_SRC = Path.home() / ".cache/openral/policy-src"
PYREP_DIR = POLICY_SRC / "PyRep"
RLBENCH_DIR = POLICY_SRC / "RLBench"
THREEDDA_DIR = POLICY_SRC / "3d_diffuser_actor"
PYREP_REPO = "https://github.com/stepjam/PyRep.git"
RLBENCH_REPO = "https://github.com/MohitShridhar/RLBench.git"
THREEDDA_REPO = "https://github.com/nickgkan/3d_diffuser_actor.git"
THREEDDA_INSTRUCTIONS = RLBENCH_HOME / "instructions/instructions/peract/instructions.pkl"


def _run(cmd: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> int:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False).returncode


def _rlbench_env() -> dict[str, str]:
    coppelia = os.environ.get("COPPELIASIM_ROOT", str(COPPELIASIM_ROOT))
    return {
        **os.environ,
        "COPPELIASIM_ROOT": coppelia,
        "LD_LIBRARY_PATH": f"{coppelia}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "QT_QPA_PLATFORM_PLUGIN_PATH": coppelia,
    }


def _ros_sourced_env(base: dict[str, str]) -> dict[str, str]:
    setup = REPO_ROOT / "install/setup.bash"
    local_setup = REPO_ROOT / "install/local_setup.bash"
    if not setup.is_file() and not local_setup.is_file():
        return base
    setup_cmd = "source /opt/ros/jazzy/setup.bash"
    setup_cmd += f" && source {setup if setup.is_file() else local_setup}"
    proc = subprocess.run(
        ["bash", "-lc", f"{setup_cmd} && env -0"],
        cwd=REPO_ROOT,
        env=base,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return base
    sourced = base.copy()
    for item in proc.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        sourced[key.decode()] = value.decode()
    return sourced


def _expand_targets(globs: list[str]) -> list[str]:
    out: set[str] = set()
    for pattern in globs:
        matches = sorted(REPO_ROOT.glob(pattern))
        if matches:
            out.update(p.relative_to(REPO_ROOT).as_posix() for p in matches if p.exists())
        elif (REPO_ROOT / pattern).exists():
            out.add(pattern)
    return sorted(out)


def _patch_3dda_cpu_fps() -> int:
    encoder = THREEDDA_DIR / "diffuser_actor/utils/encoder.py"
    if not encoder.is_file():
        print(f"rlbench: missing 3D Diffuser Actor encoder at {encoder}", file=sys.stderr)
        return 1
    text = encoder.read_text()
    old = ").to(torch.float64),"
    new = ').to(device="cpu", dtype=torch.float64),'
    if new in text:
        return 0
    if old not in text:
        print("rlbench: 3D Diffuser Actor FPS patch context not found", file=sys.stderr)
        return 1
    text = text.replace(old, new, 1)
    text = text.replace("        ).long()\n", "        ).long().to(context_features.device)\n", 1)
    encoder.write_text(text)
    return 0


def _provision_rlbench(dry_run: bool = False) -> int:  # noqa: PLR0911
    coppelia_parent = COPPELIASIM_ROOT.parent
    if not COPPELIASIM_ROOT.is_dir():
        coppelia_parent.mkdir(parents=True, exist_ok=True)
        if not COPPELIASIM_ARCHIVE.is_file():
            rc = _run(
                ["curl", "-L", "--fail", COPPELIASIM_URL, "-o", str(COPPELIASIM_ARCHIVE)],
                dry_run=dry_run,
            )
            if rc != 0:
                return rc
        rc = _run(
            ["tar", "-xJf", str(COPPELIASIM_ARCHIVE), "-C", str(coppelia_parent)],
            dry_run=dry_run,
        )
        if rc != 0:
            return rc
    if not RLBENCH_PYTHON.is_file():
        rc = _run(["uv", "venv", "--python", "3.10", str(RLBENCH_VENV)], dry_run=dry_run)
        if rc != 0:
            return rc
    POLICY_SRC.mkdir(parents=True, exist_ok=True)
    if not PYREP_DIR.exists():
        rc = _run(["git", "clone", "--depth", "1", PYREP_REPO, str(PYREP_DIR)], dry_run=dry_run)
        if rc != 0:
            return rc
    if not RLBENCH_DIR.exists():
        rc = _run(
            ["git", "clone", "--branch", "peract", "--depth", "1", RLBENCH_REPO, str(RLBENCH_DIR)],
            dry_run=dry_run,
        )
        if rc != 0:
            return rc
    if not THREEDDA_DIR.exists():
        rc = _run(["git", "clone", THREEDDA_REPO, str(THREEDDA_DIR)], dry_run=dry_run)
        if rc != 0:
            return rc
    env = _rlbench_env()
    base_deps = [
        "pip",
        "setuptools<81",
        "wheel",
        "pillow",
        "scipy",
        "natsort",
        "pyquaternion",
        "pyzmq",
        "msgpack",
    ]
    rc = _run(
        [str(RLBENCH_PYTHON), "-m", "pip", "install", "--upgrade", *base_deps],
        env=env,
        dry_run=dry_run,
    )
    if rc != 0:
        return rc
    rc = _run(
        [str(RLBENCH_PYTHON), "-m", "pip", "install", "-e", str(PYREP_DIR)],
        env=env,
        dry_run=dry_run,
    )
    if rc != 0:
        return rc
    rc = _run(
        [str(RLBENCH_PYTHON), "-m", "pip", "install", "-e", str(RLBENCH_DIR)],
        env=env,
        dry_run=dry_run,
    )
    if rc != 0:
        return rc
    policy_deps = [
        "torch==2.4.1",
        "torchvision==0.19.1",
        "einops",
        "diffusers",
        "dgl==1.1.3",
        "torchdata==0.8.0",
        "git+https://github.com/openai/CLIP.git",
    ]
    rc = _run([str(RLBENCH_PYTHON), "-m", "pip", "install", *policy_deps], env=env, dry_run=dry_run)
    if rc != 0:
        return rc
    rc = _run(
        [str(RLBENCH_PYTHON), "-m", "pip", "install", "--no-deps", "-e", str(THREEDDA_DIR)],
        env=env,
        dry_run=dry_run,
    )
    if rc != 0:
        return rc
    if not dry_run:
        patch_rc = _patch_3dda_cpu_fps()
        if patch_rc != 0:
            return patch_rc
    download = textwrap.dedent(
        f"""
        from pathlib import Path
        from zipfile import ZipFile
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download('katefgroup/3d_diffuser_actor', 'diffuser_actor_peract.pth')
        inst = hf_hub_download('katefgroup/3d_diffuser_actor', 'instructions.zip')
        out = Path({str(RLBENCH_HOME / "instructions")!r})
        out.mkdir(parents=True, exist_ok=True)
        with ZipFile(inst) as zf:
            zf.extractall(out)
        print(ckpt)
        """
    )
    return _run([".venv/bin/python", "-c", download], dry_run=dry_run)


def _load_group_targets() -> dict[str, list[str]]:
    with CONFIG.open("rb") as fh:
        raw = tomllib.load(fh)
    requirement_globs: dict[str, list[str]] = raw["requirement_globs"]
    return {group: _expand_targets(globs) for group, globs in requirement_globs.items()}


def _preflight_group(group: str, *, include_provisioned: bool) -> str | None:
    if group in PROVISIONED_GROUPS and not include_provisioned:
        return f"{group}: sidecar/proprietary asset lane; rerun with --include-provisioned"
    if group == "rlbench" and not include_provisioned:
        if (
            not os.environ.get("OPENRAL_RLBENCH_SIDECAR_PYTHON")
            and not (Path.home() / ".cache/openral/rlbench-policy/.venv/bin/python").is_file()
        ):
            return "rlbench: missing OPENRAL_RLBENCH_SIDECAR_PYTHON or default sidecar python"
        if (
            not os.environ.get("COPPELIASIM_ROOT")
            and not (
                Path.home() / ".cache/openral/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
            ).is_dir()
        ):
            return "rlbench: missing COPPELIASIM_ROOT or default CoppeliaSim install"
    if (
        group in {"qwen-vlm", "locateanything", "gr00t", "omdet"}
        and shutil.which("nvidia-smi") is None
    ):
        return f"{group}: nvidia-smi not found"
    return None


def _provision_after_sync(  # noqa: PLR0911  # reason: one explicit provisioner per lane
    group: str, *, dry_run: bool = False
) -> int:
    if group == "simpler-env":
        return _run(
            ["uv", "pip", "install", "--python", ".venv/bin/python", SIMPLER_ENV_SPEC],
            dry_run=dry_run,
        )
    if group == "robocasa-gr1":
        mujoco_rc = _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                ".venv/bin/python",
                "h5py>=3.10",
                "robosuite-models>=1.0.0",
                "robosuite==1.5.1",
                "mujoco==3.2.6",
                "numpy==1.26.4",
            ],
            dry_run=dry_run,
        )
        if mujoco_rc != 0:
            return mujoco_rc
        if not ROBOCASA_GR1_DIR.exists():
            clone_rc = _run(
                ["git", "clone", ROBOCASA_GR1_REPO, str(ROBOCASA_GR1_DIR)],
                dry_run=dry_run,
            )
            if clone_rc != 0:
                return clone_rc
        return _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                ".venv/bin/python",
                "--no-deps",
                "-e",
                str(ROBOCASA_GR1_DIR),
            ],
            dry_run=dry_run,
        )
    if group == "locateanything":
        return _run(
            [
                ".venv/bin/python",
                "-c",
                (
                    "from pathlib import Path; "
                    "from tools.locateanything_sidecar import ensure_venv; "
                    f"ensure_venv(Path({str(LOCATEANYTHING_VENV.parent)!r}))"
                ),
            ],
            dry_run=dry_run,
        )
    if group == "qwen-vlm":
        return _run(
            [
                ".venv/bin/python",
                "-c",
                (
                    "from pathlib import Path; "
                    "from tools.qwen_vlm_sidecar import ensure_venv; "
                    f"ensure_venv(Path({str(QWEN_VLM_VENV.parent)!r}))"
                ),
            ],
            dry_run=dry_run,
        )
    if group == "isaacsim":
        return _run(
            [
                ".venv/bin/python",
                "-c",
                "from openral_sim.backends.isaac_sim import _provision_isaac_venv; _provision_isaac_venv()",
            ],
            dry_run=dry_run,
        )
    if group == "robotwin":
        return _run(
            [
                ".venv/bin/python",
                "-c",
                (
                    "from openral_sim.backends.robotwin import _provision_robotwin_venv; "
                    "_provision_robotwin_venv()"
                ),
            ],
            dry_run=dry_run,
        )
    if group == "rlbench":
        return _provision_rlbench(dry_run=dry_run)
    return 0


def _post_provision_blocker(group: str) -> str | None:
    if group == "simpler-env":
        proc = subprocess.run(
            [".venv/bin/python", "-c", "import simpler_env"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode != 0:
            return f"simpler-env: simpler_env package is not importable after provisioning: {proc.stdout}"
    if group == "robocasa-gr1":
        proc = subprocess.run(
            [
                ".venv/bin/python",
                "-c",
                (
                    "from openral_sim._deps import _has_robocasa_gr1; "
                    "raise SystemExit(0 if _has_robocasa_gr1() else 1)"
                ),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode != 0:
            return f"robocasa-gr1: active robocasa install is not the GR1 fork: {proc.stdout}"
    return None


def _strict_pytest(group: str, targets: list[str], *, dry_run: bool = False) -> int:
    if not targets:
        print(f"skip {group}: no targets")
        return 0
    env = os.environ.copy()
    if group == "sim":
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
    else:
        env.setdefault("MUJOCO_GL", "osmesa")
        env.setdefault("PYOPENGL_PLATFORM", "osmesa")
    if group == "locateanything":
        env.setdefault("OPENRAL_LOCATEANYTHING_SIDECAR_VENV", str(LOCATEANYTHING_VENV))
    if group == "qwen-vlm":
        env.setdefault("OPENRAL_QWEN_VLM_SIDECAR_VENV", str(QWEN_VLM_VENV))
        env = _ros_sourced_env(env)
    if group == "rlbench":
        env.setdefault("OPENRAL_RLBENCH_SIDECAR_PYTHON", str(RLBENCH_PYTHON))
        env.setdefault("COPPELIASIM_ROOT", str(COPPELIASIM_ROOT))
        env["LD_LIBRARY_PATH"] = f"{env['COPPELIASIM_ROOT']}:{env.get('LD_LIBRARY_PATH', '')}"
        env.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", env["COPPELIASIM_ROOT"])
    if group in MANUAL_INSTALL_GROUPS:
        cmd = [".venv/bin/python", "-m", "pytest", *targets]
    else:
        cmd = ["uv", "run", "--all-packages", "pytest", *targets]
        group_args = ["--group", LANE_DEPENDENCY_GROUP.get(group, group)]
        for extra_group in LANE_EXTRA_GROUPS.get(group, []):
            group_args.extend(["--group", extra_group])
        cmd[3:3] = group_args
    cmd.extend(["-q", "-rs", "-p", "no:launch_testing", "-p", "no:launch_ros"])
    if dry_run:
        return _run(cmd, env=env, dry_run=True)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        return proc.returncode
    if not re.search(r"\d+ passed", proc.stdout):
        print(f"{group}: no passing tests", file=sys.stderr)
        return 1
    if re.search(r"\d+ skipped", proc.stdout):
        print(f"{group}: selected lane still skipped tests", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="*", help="subset of groups to run")
    parser.add_argument("--include-provisioned", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    group_targets = _load_group_targets()
    requested = args.groups or sorted(group_targets)
    unknown = sorted(set(requested) - set(group_targets))
    if unknown:
        print(f"unknown group(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    rc = 0
    for group in requested:
        reason = _preflight_group(
            group,
            include_provisioned=args.include_provisioned,
        )
        if reason:
            print(f"BLOCKED {reason}", file=sys.stderr)
            rc = 1
            continue
        if not args.no_sync:
            sync_group = LANE_DEPENDENCY_GROUP.get(group, group)
            sync_rc = _run(["just", "sync", "--group", sync_group], dry_run=args.dry_run)
            if sync_rc != 0:
                rc = sync_rc
                continue
        provision_rc = _provision_after_sync(group, dry_run=args.dry_run)
        if provision_rc != 0:
            rc = provision_rc
            continue
        post_blocker = None if args.dry_run else _post_provision_blocker(group)
        if post_blocker:
            print(f"BLOCKED {post_blocker}", file=sys.stderr)
            rc = 1
            continue
        cleanup_onnx = group == "onnx-export" and not ONNX_EXPORT.exists()
        if group == "onnx-export":
            export_rc = _run(
                [
                    "uv",
                    "run",
                    "--all-packages",
                    "--group",
                    "onnx-export",
                    "python",
                    "tools/export_rtdetr_onnx.py",
                    "--out",
                    ONNX_EXPORT.relative_to(REPO_ROOT).as_posix(),
                ],
                dry_run=args.dry_run,
            )
            if export_rc != 0:
                rc = export_rc
                continue
        test_rc = _strict_pytest(group, group_targets[group], dry_run=args.dry_run)
        if cleanup_onnx and ONNX_EXPORT.exists():
            ONNX_EXPORT.unlink()
        if test_rc != 0:
            rc = test_rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
