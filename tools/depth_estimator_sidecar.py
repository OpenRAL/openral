"""Start the depth sidecar a ``kind: depth_estimator`` rSkill names.

The deploy launch calls this with the rSkill's manifest; it applies the manifest's
license guard, then hands over to the backend for the manifest's ``runtime``:

* ``pytorch``: the in-tree sidecar for ``depth_estimator.family``
  (``tools/da3_depth_sidecar.py`` or ``tools/depth_pro2_sidecar.py``), with the
  model repo and revision pinned by ``weights_uri``;
* anything else (``tensorrt``): the backend registered for that runtime under the
  ``openral.depth_estimator_backends`` entry-point group (OpenRAL Pro ships
  ``tensorrt``). The entry point is a callable ``(manifest_path, host, port) -> int``
  that serves the same wire protocol and does not return while serving.

Usage::

    python tools/depth_estimator_sidecar.py \\
        --manifest rskills/rskill-da3metric_large-any-metric_depth-bf16/rskill.yaml --port 5772
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import entry_points
from pathlib import Path

#: Entry-point group out-of-tree depth backends register under, keyed by runtime value.
BACKEND_GROUP = "openral.depth_estimator_backends"

_TOOLS = Path(__file__).resolve().parent
_FAMILY_LAUNCHER = {"da3": "da3_depth_sidecar.py", "depth_pro2": "depth_pro2_sidecar.py"}


def family_command(manifest_path: str, host: str, port: int) -> list[str]:
    """The argv that boots the in-tree ``pytorch`` sidecar for this manifest.

    Example:
        >>> cmd = family_command(
        ...     "rskills/rskill-da3metric_large-any-metric_depth-bf16/rskill.yaml",
        ...     "127.0.0.1",
        ...     5772,
        ... )
        >>> [Path(cmd[1]).name, *cmd[2:4], cmd[5], *cmd[6:]]
        ['da3_depth_sidecar.py', '--model', 'depth-anything/DA3METRIC-LARGE', '4010e39f3634a45bc60553321fb49fb760bd594e', '--host', '127.0.0.1', '--port', '5772', '--process-res', '504']
    """
    from openral_core import RSkillManifest

    manifest = RSkillManifest.from_yaml(manifest_path)
    contract = manifest.depth_estimator
    if contract is None or manifest.weights_uri is None:
        raise SystemExit(f"{manifest_path}: not a kind: depth_estimator rSkill")
    if not manifest.weights_uri.startswith("hf://"):
        raise SystemExit(f"{manifest_path}: weights_uri must be hf://<repo>@<revision>")
    repo, _, revision = manifest.weights_uri.removeprefix("hf://").partition("@")
    cmd = [sys.executable, str(_TOOLS / _FAMILY_LAUNCHER[contract.family.value])]
    cmd += ["--model", repo]
    if revision:
        cmd += ["--revision", revision]
    cmd += ["--host", host, "--port", str(port)]
    if contract.process_res is not None and contract.family.value == "da3":
        # Only the DA3 launcher takes it; Depth Pro 2 runs at its own resolution.
        cmd += ["--process-res", str(contract.process_res)]
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", required=True, help="kind: depth_estimator rskill.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    args = p.parse_args()

    from openral_rskill.loader import rSkill

    manifest = rSkill.from_yaml(args.manifest).manifest  # kind + license gate
    if manifest.kind != "depth_estimator":
        raise SystemExit(f"{args.manifest}: kind {manifest.kind!r}, not depth_estimator")
    runtime = str(getattr(manifest.runtime, "value", manifest.runtime))
    print(f"[depth-sidecar] {manifest.name} runtime={runtime} port={args.port}", flush=True)
    if runtime == "pytorch":
        cmd = family_command(args.manifest, args.host, args.port)
        os.execv(cmd[0], cmd)
    found = [ep for ep in entry_points(group=BACKEND_GROUP) if ep.name == runtime]
    if not found:
        raise SystemExit(
            f"{manifest.name}: runtime {runtime!r} has no backend in entry-point group "
            f"{BACKEND_GROUP!r} (a tensorrt depth backend ships with OpenRAL Pro)"
        )
    return int(found[0].load()(args.manifest, args.host, args.port))


if __name__ == "__main__":
    sys.exit(main())
