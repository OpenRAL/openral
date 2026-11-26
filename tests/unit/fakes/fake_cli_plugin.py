"""Real importable ``openral.cli`` plugin used by ``test_cli_plugins.py`` (CLAUDE.md §1.11)."""

from __future__ import annotations

from collections.abc import Mapping

import typer


def register(groups: Mapping[str, typer.Typer]) -> None:
    @groups["deploy"].command("fakecmd")
    def fakecmd() -> None:
        typer.echo("fake")


def register_replacing(groups: Mapping[str, typer.Typer]) -> None:
    @groups["deploy"].command("run")
    def run() -> None:
        typer.echo("hijacked")


def register_then_raise(groups: Mapping[str, typer.Typer]) -> None:
    @groups["deploy"].command("half")
    def half() -> None: ...

    @groups["deploy"].command("run")
    def run() -> None: ...

    raise RuntimeError("boom")


def register_duplicate_group(groups: Mapping[str, typer.Typer]) -> None:
    groups[""].add_typer(typer.Typer(), name="robot")


def register_everywhere(groups: Mapping[str, typer.Typer]) -> None:
    @groups[""].command("rootfake")
    def rootfake() -> None:
        typer.echo("root")

    @groups["robot"].command("robotfake")
    def robotfake() -> None: ...

    @groups["sim"].command("simfake")
    def simfake() -> None: ...
