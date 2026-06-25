"""Tests for the deploy-config rSkill guards in ``openral_cli.main``.

`_check_vla_against_robot` and `_preflight_deploy_vla` validate that a deploy
config's `vla` rSkill is installed and embodiment-compatible with the robot,
reusing `openral_detect.check_single_rskill`. Exercised against the **real**
in-tree ``robots/so101_follower`` manifest and real rSkill manifests (an SO-101
policy, a wrong-embodiment ALOHA policy, and a missing ref) — no network, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from openral_cli.main import _check_vla_against_robot, _preflight_deploy_vla
from openral_core.schemas import RobotDescription
from openral_detect import ScaffoldOverrides, scaffold_robot_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
SO101_YAML = REPO_ROOT / "robots" / "so101_follower" / "robot.yaml"


def _so101() -> RobotDescription:
    return RobotDescription.from_yaml(str(SO101_YAML))


class TestCheckVlaAgainstRobot:
    def test_matching_rskill_passes(self) -> None:
        assert _check_vla_against_robot("rskills/molmoact2-so101-nf4", _so101()) is None

    def test_wrong_embodiment_rejected(self) -> None:
        msg = _check_vla_against_robot("rskills/act-aloha", _so101())
        assert msg is not None
        assert "not compatible" in msg and "embodiment" in msg

    def test_uninstalled_rskill_rejected(self) -> None:
        msg = _check_vla_against_robot("rskills/does-not-exist", _so101())
        assert msg is not None
        assert "not installed" in msg


class TestPreflightDeployVla:
    def _env(self, weights_uri: str | None):  # type: ignore[no-untyped-def]
        ov = ScaffoldOverrides(vla_id="molmoact2", vla_weights_uri=weights_uri)
        return scaffold_robot_environment(_so101(), None, overrides=ov)

    def test_placeholder_vla_blocks_deploy(self) -> None:
        env = self._env(None)  # leaves the TODO sentinel
        with pytest.raises(typer.Exit) as exc:
            _preflight_deploy_vla(env, SO101_YAML)
        assert exc.value.exit_code == 1

    def test_wrong_embodiment_blocks_deploy(self) -> None:
        env = self._env("rskills/act-aloha")
        with pytest.raises(typer.Exit) as exc:
            _preflight_deploy_vla(env, SO101_YAML)
        assert exc.value.exit_code == 1

    def test_compatible_vla_passes(self) -> None:
        env = self._env("rskills/molmoact2-so101-nf4")
        # Does not raise.
        _preflight_deploy_vla(env, SO101_YAML)
