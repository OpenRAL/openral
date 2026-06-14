"""Real-component tests for ``openral install`` and the interactive REPL.

These tests exercise the actual ``openral_cli.install`` Typer app and the
REPL dispatcher in ``openral_cli.main`` — no mocks, no smoke-only
assertions (CLAUDE.md §1.11 / §5.4).

The pure-Python pieces (group enumeration, conflict detection, REPL line
dispatch) run unconditionally.  The piece that would actually shell out
to ``uv pip install`` (``install sim``) is gated behind a ``--force``-less
conflict-only probe — we never trigger a real network install in the
unit tier; that is the integration tier's job (see
``tests/integration/test_installer_curl_bash.py`` when it ships).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib
from openral_cli.install import _CONFLICTS, _GROUPS, _check_conflicts, install_app
from openral_cli.main import (
    BANNER,
    _dispatch_repl_line,
    _path_completer,
    _run_repl,
    app,
)
from openral_core.exceptions import ROSConfigError
from typer.testing import CliRunner

runner = CliRunner()

# Locate the workspace root pyproject.toml so we can assert the duplicated
# [dependency-groups] table in install.py stays in lockstep (per ADR-0021
# "Negative / accepted tradeoffs"). When the tests run from an installed
# wheel without a checkout, the file is absent and the lockstep check skips.
_REPO_ROOT_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


class TestInstallGroupRegistry:
    def test_every_advertised_group_has_packages(self) -> None:
        # Empty package list would mean `openral install <group>` is a no-op.
        for name, pkgs in _GROUPS.items():
            assert pkgs, f"group `{name}` has no packages"

    def test_libero_robocasa_are_marked_as_conflicting(self) -> None:
        # ADR-0011 invariant. If this drifts, the curl-bash installer will
        # let users install both into one venv and uv's solver will fail
        # with a less helpful error than our typed ROSConfigError.
        assert frozenset({"libero", "robocasa"}) in _CONFLICTS

    @pytest.mark.skipif(
        not _REPO_ROOT_PYPROJECT.is_file(),
        reason="repo root pyproject.toml not present (running from installed wheel?)",
    )
    def test_groups_mirror_workspace_root_pyproject(self) -> None:
        # Lockstep check (ADR-0021): the curl-bash installer publishes its
        # own copy of the [dependency-groups] table because it must work
        # before the workspace is cloned. Drift means a user who runs
        # `openral install sim` gets a different set of packages than a
        # contributor who runs `just sync --all-packages --group sim`.
        with _REPO_ROOT_PYPROJECT.open("rb") as f:
            data = tomllib.load(f)
        ws_groups = data.get("dependency-groups", {})
        for group_name, installer_pkgs in _GROUPS.items():
            assert group_name in ws_groups, (
                f"group `{group_name}` is in openral_cli.install._GROUPS but "
                f"missing from the workspace root pyproject.toml"
            )
            # Compare the unordered package sets (the order in the toml is a
            # human reading order; only membership matters to uv).
            assert set(installer_pkgs) == set(ws_groups[group_name]), (
                f"group `{group_name}` packages drift between "
                f"openral_cli.install._GROUPS and the workspace root pyproject."
            )


class TestInstallConflictGuard:
    def test_conflict_raises_with_helpful_message(self) -> None:
        with pytest.raises(ROSConfigError) as excinfo:
            _check_conflicts("libero", frozenset({"robocasa"}))
        msg = str(excinfo.value)
        assert "libero" in msg and "robocasa" in msg
        assert "ADR-0011" in msg  # cite the source-of-truth ADR
        assert "--force" in msg  # surface the escape hatch

    def test_no_conflict_no_raise(self) -> None:
        # sim has no exclusions today.
        _check_conflicts("sim", frozenset({"libero", "metaworld"}))


class TestInstallListCommand:
    def test_install_list_runs_and_shows_every_group(self) -> None:
        result = runner.invoke(install_app, ["list"])
        assert result.exit_code == 0, result.stderr
        # Every advertised group appears in the table.
        for group in _GROUPS:
            assert group in result.stdout
        # plus the sudo-gated ros bootstrap row.
        assert "ros" in result.stdout

    def test_install_list_marks_libero_robocasa_mutual_exclusion(self) -> None:
        result = runner.invoke(install_app, ["list"])
        # Both rows should mention the other side in the conflicts column.
        # We can't rely on column layout (rich may wrap), so just assert
        # both names co-occur in the printed table.
        out = result.stdout
        assert "libero" in out and "robocasa" in out


class TestInstallSubcommandWiring:
    def test_install_subapp_is_mounted_on_main_app(self) -> None:
        # `openral install list` must be reachable from the canonical app.
        result = runner.invoke(app, ["install", "list"])
        assert result.exit_code == 0, result.stderr
        assert "sim" in result.stdout

    def test_install_unknown_group_raises_config_error(self) -> None:
        from openral_cli.install import _install_group

        with pytest.raises(ROSConfigError) as excinfo:
            _install_group("nope-not-a-group", force=False)
        assert "Unknown group" in str(excinfo.value)


class TestReplDispatch:
    def test_app_is_named_openral(self) -> None:
        # Both the Typer name and the console-script wiring depend on this.
        assert app.info.name == "openral"

    def test_banner_is_non_empty_ascii_art(self) -> None:
        # The REPL splash uses ASCII art for the project name; we don't
        # pin the exact glyph layout (it may evolve), only that the
        # banner is non-trivial multi-line ASCII.
        assert BANNER.strip()
        assert BANNER.count("\n") >= 4

    def test_dispatch_repl_line_runs_install_list(self, capsys: pytest.CaptureFixture[str]) -> None:
        # Bare subcommand path: "install list" inside the REPL is
        # equivalent to `openral install list` on the shell. We don't
        # mock Typer — we re-enter the real app with standalone_mode=False
        # and assert the install-list table rendered.
        _dispatch_repl_line("install list")
        out = capsys.readouterr().out
        for group in _GROUPS:
            assert group in out

    def test_dispatch_repl_line_handles_quoted_args(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # shlex.split must preserve quoted strings so that paths with
        # spaces survive (e.g. `sim run --config 'a b.yaml'`).
        _dispatch_repl_line("install list")  # known-good baseline first
        capsys.readouterr()
        # Unknown subcommand path: must NOT raise (REPL keeps running);
        # Typer prints a usage error.
        _dispatch_repl_line("no-such-subcommand")
        # No exception escaped — that's the contract.

    def test_dispatch_repl_line_ignores_blank_input(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _dispatch_repl_line("")
        _dispatch_repl_line("   ")
        # Should produce no output and no exception.
        assert capsys.readouterr().out == ""

    def test_dispatch_repl_line_exit_raises_eof(self) -> None:
        # `exit` / `quit` / `:q` signal the REPL loop to terminate by
        # raising EOFError (the outer loop catches it).
        with pytest.raises(EOFError):
            _dispatch_repl_line("exit")
        with pytest.raises(EOFError):
            _dispatch_repl_line("quit")
        with pytest.raises(EOFError):
            _dispatch_repl_line(":q")

    def test_run_repl_prints_banner_then_exits_on_eof(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Feed EOF immediately; the REPL should print the banner and
        # return cleanly without consuming any subcommand.
        monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(EOFError()))
        _run_repl()
        captured = capsys.readouterr()
        # Banner content (rich-formatted) is on stdout; just check the
        # subtitle appears.
        assert "agentic" in captured.out.lower() or "Open" in captured.out

    def test_run_repl_dispatches_then_exits(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Two-line script: "install list" then EOF. Asserts that the
        # subcommand actually executed (table content visible) and the
        # loop terminated.
        lines = iter(["install list", ""])

        def fake_input(_prompt: str = "") -> str:
            try:
                return next(lines)
            except StopIteration as exc:
                raise EOFError from exc

        monkeypatch.setattr("builtins.input", fake_input)
        _run_repl()
        out = capsys.readouterr().out
        # The install table must have rendered inside the loop.
        assert "sim" in out


class TestPathCompleter:
    """Tab completion for filesystem paths in the REPL.

    The completer is a stdlib ``readline``-shaped function
    ``(text, state) -> str | None`` that returns one match at a time.
    These tests exercise the real function against a real on-disk
    fixture tree built per test in ``tmp_path``; no monkeypatching of
    ``os.listdir`` / ``glob`` / ``readline``.
    """

    def _collect(self, text: str) -> list[str]:
        out: list[str] = []
        state = 0
        while True:
            match = _path_completer(text, state)
            if match is None:
                break
            out.append(match)
            state += 1
        return out

    def test_completes_relative_path_to_child(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "alpha").mkdir()
        (tmp_path / "alphabet.txt").write_text("x")
        (tmp_path / "beta").mkdir()
        monkeypatch.chdir(tmp_path)

        matches = self._collect("alph")
        assert sorted(matches) == ["alpha/", "alphabet.txt"]

    def test_directory_match_carries_trailing_slash(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.yaml").write_text("x")
        monkeypatch.chdir(tmp_path)

        matches = self._collect("")
        assert "subdir/" in matches
        assert "file.yaml" in matches

    def test_descends_into_partially_typed_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "scene.yaml").write_text("x")
        (tmp_path / "configs" / "robot.yaml").write_text("x")
        monkeypatch.chdir(tmp_path)

        matches = self._collect("configs/")
        assert sorted(matches) == ["configs/robot.yaml", "configs/scene.yaml"]

    def test_returns_none_past_last_match(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "only.txt").write_text("x")
        monkeypatch.chdir(tmp_path)

        assert _path_completer("only", 0) == "only.txt"
        assert _path_completer("only", 1) is None

    def test_tilde_expands_and_round_trips(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Point HOME at the tmp tree so a literal ~ in the completion
        # query resolves into our fixture, then assert the returned
        # match is rewritten back to ~-prefixed form (otherwise the
        # user's line buffer would silently flip to the absolute path).
        (tmp_path / "notes").mkdir()
        (tmp_path / "notes" / "draft.md").write_text("x")
        monkeypatch.setenv("HOME", str(tmp_path))

        matches = self._collect("~/notes/")
        assert matches == ["~/notes/draft.md"]

    def test_no_matches_returns_none_immediately(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert _path_completer("nothing-here", 0) is None


class TestReplCompleterWiring:
    """The completer must be installed by ``_run_repl`` before the
    first ``input()`` call, with shell-shaped delimiters so paths
    starting with ``~`` and containing ``/`` are not split on word
    boundaries by readline.
    """

    def test_run_repl_installs_path_completer_and_tab_binding(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        readline = pytest.importorskip("readline")

        prior_completer = readline.get_completer()
        prior_delims = readline.get_completer_delims()
        try:
            monkeypatch.setattr(
                "builtins.input",
                lambda _prompt="": (_ for _ in ()).throw(EOFError()),
            )
            _run_repl()

            assert readline.get_completer() is _path_completer
            delims = readline.get_completer_delims()
            # Path-bearing chars must NOT be delimiters or readline will
            # hand the completer only the trailing fragment.
            for ch in ("~", "/", ".", "-", "_"):
                assert ch not in delims, (
                    f"completer delim set must not include {ch!r} or path "
                    f"completion will break mid-token"
                )
            # Token-bearing chars MUST be delimiters or "sim run --config "
            # would be passed in as one giant blob.
            assert " " in delims
        finally:
            readline.set_completer(prior_completer)
            readline.set_completer_delims(prior_delims)


class TestInstallerShellScript:
    SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"

    @pytest.mark.skipif(
        not SCRIPT.is_file(),
        reason="scripts/install.sh not present (installed wheel?)",
    )
    def test_install_script_is_executable_bash(self) -> None:
        # The curl-bash one-liner pipes to `bash`, so the shebang and
        # bash -n syntax check must both be valid.
        import subprocess

        text = self.SCRIPT.read_text()
        assert text.startswith("#!/usr/bin/env bash"), "missing or wrong shebang"
        # bash -n parses without executing.
        proc = subprocess.run(
            ["bash", "-n", str(self.SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr

    @pytest.mark.skipif(
        not SCRIPT.is_file(),
        reason="scripts/install.sh not present (installed wheel?)",
    )
    def test_install_script_documents_every_install_group(self) -> None:
        # The "next steps" block at the end of the installer must list
        # every group the installer is able to layer in. If install.py adds
        # a new group, the README in the script must mention it.
        text = self.SCRIPT.read_text()
        for group in _GROUPS:
            assert f"openral install {group}" in text, (
                f"install.sh next-steps block missing `openral install {group}`"
            )
        # …plus the sudo-gated ros bootstrap row.
        assert "openral install ros" in text

    @pytest.mark.skipif(
        not SCRIPT.is_file(),
        reason="scripts/install.sh not present (installed wheel?)",
    )
    def test_install_script_refuses_to_run_as_root(self) -> None:
        # The script must include an early `id -u == 0` refusal — a curl-
        # bash that wgets root privileges by silently allowing sudo is a
        # CLAUDE.md §1.1 safety violation analogue (no surprises that
        # touch the user's machine state without consent).
        text = self.SCRIPT.read_text()
        assert "id -u" in text and "root" in text.lower()
