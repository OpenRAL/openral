"""The ``openral.cli`` entry-point seam: installed packages add commands to the one CLI."""

from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import EntryPoint

import pytest
from openral_cli import main
from openral_sim.cli import sim_app
from typer.testing import CliRunner

_FAKE = "tests.unit.fakes.fake_cli_plugin"


def _ep(attr: str, module: str = _FAKE, name: str = "fake") -> EntryPoint:
    return EntryPoint(name=name, value=f"{module}:{attr}", group="openral.cli")


@pytest.fixture
def clean_deploy() -> Iterator[None]:
    """Restore every app the seam can touch after a test mutates it."""
    apps = [main.app, main.deploy_app, main.robot_app, sim_app]
    saved = [(list(a.registered_commands), list(a.registered_groups)) for a in apps]
    yield
    for a, (cmds, groups) in zip(apps, saved, strict=True):
        a.registered_commands[:] = cmds
        a.registered_groups[:] = groups


def test_plugin_adds_deploy_command(clean_deploy: None) -> None:
    main._register_plugin_commands([_ep("register")])
    result = CliRunner().invoke(main.app, ["deploy", "fakecmd"])
    assert result.exit_code == 0
    assert "fake" in result.output


def test_broken_plugin_is_skipped(clean_deploy: None, capsys: pytest.CaptureFixture[str]) -> None:
    main._register_plugin_commands([_ep("register", module="no_such_module_xyz", name="brokenpkg")])
    assert CliRunner().invoke(main.app, ["deploy", "--help"]).exit_code == 0
    assert "brokenpkg" in capsys.readouterr().err


def test_plugin_cannot_replace_a_command(
    clean_deploy: None, capsys: pytest.CaptureFixture[str]
) -> None:
    before = list(main.deploy_app.registered_commands)
    main._register_plugin_commands([_ep("register_replacing", name="evil")])
    assert main.deploy_app.registered_commands == before
    assert "evil" in capsys.readouterr().err
    assert "hijacked" not in CliRunner().invoke(main.app, ["deploy", "run", "--help"]).output


def test_raising_plugin_is_rolled_back(
    clean_deploy: None, capsys: pytest.CaptureFixture[str]
) -> None:
    before = list(main.deploy_app.registered_commands)
    main._register_plugin_commands([_ep("register_then_raise", name="halfpkg")])
    assert main.deploy_app.registered_commands == before
    assert "halfpkg" in capsys.readouterr().err


def test_plugin_cannot_replace_a_group(clean_deploy: None) -> None:
    before = list(main.app.registered_groups)
    main._register_plugin_commands([_ep("register_duplicate_group")])
    assert main.app.registered_groups == before


def test_plugin_gets_root_robot_and_sim_groups(clean_deploy: None) -> None:
    main._register_plugin_commands([_ep("register_everywhere")])
    runner = CliRunner()
    assert "root" in runner.invoke(main.app, ["rootfake"]).output
    assert runner.invoke(main.app, ["robot", "robotfake"]).exit_code == 0
    assert runner.invoke(main.app, ["sim", "simfake"]).exit_code == 0
