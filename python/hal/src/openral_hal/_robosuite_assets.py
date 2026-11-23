"""A robosuite robot composed into one MJCF file, as robosuite assembles it at runtime.

``robots/panda_mobile/robot.yaml`` is RoboCasa's ``PandaOmron``: a Franka Panda
on robosuite's Omron holonomic base with the Panda gripper. No file holds that
robot. robosuite (MIT, >= 1.5, the version RoboCasa pins) builds it in memory
from its arm, base and gripper models. ``compose_robot_mjcf`` performs the same
composition as ``robosuite.robots.Robot.load_model``, with the same ``idn``
naming, so the joint names match the manifest's ``sim_joint_name``s
(``mobilebase0_*``, ``robot0_*``, ``gripper0_right_*``), and writes the XML into
``$OPENRAL_CACHE_DIR/robosuite/<version>-<install hash>/<robot>.xml``. Mesh
paths point into the installed robosuite package, so the cache is keyed by where
that package lives as well as its version: RoboCasa's pinned clone and the PyPI
wheel both report 1.5.2. Nothing is vendored.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from openral_core.exceptions import ROSConfigError

__all__ = ["compose_robot_mjcf"]


def compose_robot_mjcf(name: str) -> Path:
    """Compose robosuite robot ``name`` with its default base and grippers; return the MJCF.

    Raises:
        ROSConfigError: robosuite >= 1.5 is not installed, or ``name`` is not a
            registered robosuite robot.
    """
    try:
        import robosuite  # noqa: PLC0415  # reason: optional sim dep
        from robosuite.models.bases import robot_base_factory  # noqa: PLC0415
        from robosuite.models.grippers import gripper_factory  # noqa: PLC0415
        from robosuite.models.robots import create_robot  # noqa: PLC0415
        from robosuite.models.robots.robot_model import REGISTERED_ROBOTS  # noqa: PLC0415
    except ImportError as exc:
        raise ROSConfigError(
            f"robosuite:{name} needs robosuite >= 1.5 (`just sync --group robocasa`): {exc}"
        ) from exc
    if name not in REGISTERED_ROBOTS:
        raise ROSConfigError(f"robosuite:{name}: not a registered robosuite robot")
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    package = Path(str(robosuite.__file__)).parent
    install = hashlib.sha256(str(package).encode()).hexdigest()[:12]
    out: Path = base / "robosuite" / f"{robosuite.__version__}-{install}" / f"{name}.xml"
    if out.is_file():
        return out
    model = create_robot(name, idn=0)
    model.add_base(base=robot_base_factory(model.default_base, idn=0))
    for arm in model.arms:
        model.add_gripper(
            gripper_factory(model.default_gripper[arm], idn=f"0_{arm}"), model.eef_name[arm]
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(model.get_xml(), encoding="utf-8")
    os.replace(tmp, out)
    return out
