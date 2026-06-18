"""Tests for the install-plan fixes surfaced by the rSkill audit GPU smoke tests.

Covers the four bootstrap rough edges that fired during real-component
end-to-end runs of pi05 / rldx × LIBERO / GR1 / RC365 on an RTX 4070:

1+2. Editable-install shadow directory cleanup
     (``_remove_editable_shadow_step`` wired into the robocasa kitchen
     + GR1 plans).
3.   RLDX client deps gated by their own backend plan so a sibling
     ``uv sync --group robocasa`` does not silently strip them.
4.   ``hf-libero==0.1.3`` distutils-installed-metadata uninstall barrier
     bypassed via ``--reinstall-package hf-libero`` on the libero plan,
     plus ``--inexact`` on the robocasa plans so they preserve
     cross-backend deps instead of trying to uninstall them.

These are install-plan tests — they assert the plan SHAPE (argv flags,
step ordering), not that the subprocess actually runs. Real-component
verification happened on the GPU host.
"""

from __future__ import annotations

import importlib.metadata
import sysconfig
from pathlib import Path

import pytest
from openral_sim import _deps
from openral_sim._deps import (
    _ROBOSUITE_PIN,
    _libero_plan,
    _openarm_robosuite_plan,
    _refresh_editable_finders,
    _remove_editable_shadow_step,
    _rldx_client_plan,
    _robocasa_gr1_plan,
    _robocasa_kitchen_plan,
    get_plan,
)

# ─── Fix #1+2: shadow dir cleanup ─────────────────────────────────────────


class TestRemoveEditableShadow:
    def test_step_is_idempotent_when_no_shadow(self, tmp_path: Path) -> None:
        """The cleanup step is safe to run unconditionally — no shadow → no-op.

        We run the step's argv as a real subprocess to confirm the
        embedded Python snippet exits 0 and prints the no-shadow
        message, even when the named package does not exist.
        """
        import subprocess

        step = _remove_editable_shadow_step("definitely_not_a_package_xyz_42")
        result = subprocess.run(step.argv, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "no shadow to clean" in result.stdout

    def test_step_removes_namespace_shadow(self, tmp_path: Path) -> None:
        """When a shadow dir without __init__.py exists, the step removes it."""
        import subprocess

        # Build a fake shadow directly in the real site-packages, then
        # run the step and confirm it's gone.
        sp = Path(sysconfig.get_paths()["purelib"])
        shadow_name = "shadow_test_pkg_xyz_42"
        shadow = sp / shadow_name
        shadow.mkdir()
        (shadow / "macros_private.py").write_text("# stub\n")
        try:
            # Sanity: no __init__.py → this is a namespace stub.
            assert not (shadow / "__init__.py").is_file()
            assert shadow.is_dir()

            step = _remove_editable_shadow_step(shadow_name)
            result = subprocess.run(step.argv, capture_output=True, text=True, check=False)
            assert result.returncode == 0, result.stderr
            assert "removed editable-install shadow dir" in result.stdout
            assert not shadow.exists(), "shadow dir should be gone"
        finally:
            # Belt-and-suspenders cleanup in case the step itself failed.
            if shadow.exists():
                import shutil

                shutil.rmtree(shadow)

    def test_step_preserves_real_package_dir(self, tmp_path: Path) -> None:
        """A site-packages dir WITH __init__.py is a real install — do not touch."""
        import subprocess

        sp = Path(sysconfig.get_paths()["purelib"])
        pkg_name = "real_pkg_test_xyz_42"
        pkg_dir = sp / pkg_name
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("# real pkg\n")
        try:
            step = _remove_editable_shadow_step(pkg_name)
            result = subprocess.run(step.argv, capture_output=True, text=True, check=False)
            assert result.returncode == 0, result.stderr
            assert "no shadow to clean" in result.stdout
            assert pkg_dir.is_dir(), "real package dir must survive"
            assert (pkg_dir / "__init__.py").is_file()
        finally:
            import shutil

            if pkg_dir.exists():
                shutil.rmtree(pkg_dir)

    def test_robocasa_kitchen_plan_includes_shadow_cleanup_after_editable(self) -> None:
        """Kitchen plan adds shadow cleanup after the robosuite editable install."""
        plan = _robocasa_kitchen_plan()
        # Walk steps; find the editable install of robosuite and confirm
        # the very next step is the shadow cleanup for robosuite.
        descriptions = [s.description for s in plan.steps]
        rs_editable_idx = next(
            i for i, d in enumerate(descriptions) if "uv pip install -e" in d and "/robosuite" in d
        )
        assert "namespace shadow for 'robosuite'" in descriptions[rs_editable_idx + 1], descriptions
        # And the last step's a shadow cleanup for robocasa (after the
        # non-editable robocasa install — kitchen variant).
        assert "namespace shadow for 'robocasa'" in descriptions[-1]

    def test_robocasa_gr1_plan_includes_shadow_cleanup_after_each_editable(self) -> None:
        """GR1 plan adds shadow cleanup after BOTH editable installs (robosuite + robocasa)."""
        plan = _robocasa_gr1_plan()
        descriptions = [s.description for s in plan.steps]
        cleanups = [d for d in descriptions if "namespace shadow" in d]
        # One cleanup per editable install: robosuite + robocasa-gr1 fork.
        assert len(cleanups) == 2, descriptions
        assert any("'robosuite'" in d for d in cleanups)
        assert any("'robocasa'" in d for d in cleanups)


# ─── Fix #3: rldx_client plan ─────────────────────────────────────────────


class TestRldxClientPlan:
    def test_plan_is_registered(self) -> None:
        """`rldx_client` is a first-class backend plan, addressable via get_plan."""
        plan = get_plan("rldx_client")
        assert plan.backend_id == "rldx_client"

    def test_plan_uses_inexact_so_cross_backend_deps_survive(self) -> None:
        """`--inexact` keeps pyzmq/msgpack across sibling --group syncs.

        This is the wedge that fired three times during the audit GPU
        smoke pass: every `uv sync --group robocasa` stripped
        pyzmq/msgpack, leaving the rldx adapter unable to import zmq.
        The fix lives on the robocasa plans (their first sync now uses
        --inexact) AND on this client plan so a re-install also preserves
        siblings.
        """
        plan = _rldx_client_plan()
        sync_step = plan.steps[0]
        assert "--inexact" in sync_step.argv

    def test_rldx_adapter_calls_ensure_backend_deps_before_zmq_import(self) -> None:
        """The rldx adapter's __post_init__ calls ensure_backend_deps('rldx_client').

        Cheaper than spinning up a sidecar: read the source and confirm
        the call exists in the right order (before the ``import zmq``).
        """
        import inspect

        from openral_sim.policies import rldx as rldx_mod

        src = inspect.getsource(rldx_mod._RLDXSidecarAdapter.__post_init__)
        ensure_idx = src.find('ensure_backend_deps("rldx_client")')
        import_idx = src.find("import zmq")
        assert ensure_idx != -1, "rldx adapter must call ensure_backend_deps('rldx_client')"
        assert import_idx != -1, "rldx adapter must import zmq"
        assert ensure_idx < import_idx, (
            "ensure_backend_deps must run BEFORE the zmq import so a stripped "
            "venv is auto-rehydrated rather than raising ROSConfigError"
        )


# ─── Fix #4: hf-libero distutils workaround ───────────────────────────────


class TestHfLiberoWorkaround:
    def test_libero_plan_reinstalls_hf_libero(self) -> None:
        """`--reinstall-package hf-libero` clears the distutils-installed-metadata wedge.

        Reason: ``hf-libero==0.1.3`` ships with distutils-installed
        metadata. uv refuses to uninstall it during ANY subsequent
        `uv sync` of a different group, breaking every cross-backend
        install. Forcing a reinstall on this single package replaces
        the distutils stub with proper modern metadata.
        """
        plan = _libero_plan()
        sync_step = plan.steps[0]
        # Both flags present, in any order.
        assert "--reinstall-package" in sync_step.argv
        assert "hf-libero" in sync_step.argv
        # And we use --inexact so installing libero on top of a venv
        # that already has robocasa/metaworld does not strip them.
        assert "--inexact" in sync_step.argv

    def test_libero_manual_hint_carries_the_same_flags(self) -> None:
        """The manual_hint surfaced to the user mirrors the auto-installer's flags."""
        plan = _libero_plan()
        assert "--inexact" in plan.manual_hint
        assert "--reinstall-package hf-libero" in plan.manual_hint


class TestLiberoReadinessProbe:
    """`_has_libero` must report readiness via metadata, never crash on it.

    Regression: the probe read ``robosuite.__version__``, which the
    openral-vendored robosuite 1.5.x builds do not expose. On a
    robocasa-provisioned venv (robosuite 1.5.1) the bare attribute access
    raised ``AttributeError: module 'robosuite' has no attribute
    '__version__'``, crashing ``ensure_backend_deps('libero')`` instead of
    falling through to the install plan.
    """

    @pytest.fixture(autouse=True)
    def _modules_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Both prerequisite modules resolve; isolate the version branch.
        monkeypatch.setattr(_deps, "_has_module", lambda _module: True)

    def test_non_1_4_robosuite_is_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A >=1.5 robosuite (robocasa-provisioned) → LIBERO not ready, no raise."""
        monkeypatch.setattr(_deps.importlib.metadata, "version", lambda _pkg: "1.5.1")
        assert _deps._has_libero() is False

    def test_missing_robosuite_metadata_is_not_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Absent distribution metadata → not ready (falls through to plan)."""

        def _raise(_pkg: str) -> str:
            raise importlib.metadata.PackageNotFoundError(_pkg)

        monkeypatch.setattr(_deps.importlib.metadata, "version", _raise)
        assert _deps._has_libero() is False

    def test_1_4_robosuite_is_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The LIBERO-pinned robosuite 1.4 series → ready."""
        monkeypatch.setattr(_deps.importlib.metadata, "version", lambda _pkg: "1.4.0")
        assert _deps._has_libero() is True


# ─── Fix #5: openarm_robosuite plan + preflight call ──────────────────────


class TestOpenarmRobosuitePlan:
    """Regression for the openarm_tabletop_pnp install-prompt gap.

    Before this plan landed, every other sim backend
    (libero / robocasa_{kitchen,gr1} / metaworld / aloha / maniskill3 /
    simpler_env) called ``ensure_backend_deps(...)`` at the top of its
    SCENES factory and got the Rich banner + typer.confirm flow on
    first use; openarm_robosuite alone skipped straight to ``from
    robosuite ... import MjSim`` and crashed with a bare
    ``ModuleNotFoundError`` when the user had only run ``uv sync
    --all-packages``.
    """

    def test_plan_is_registered(self) -> None:
        """`openarm_robosuite` is addressable via :func:`get_plan`."""
        plan = get_plan("openarm_robosuite")
        assert plan.backend_id == "openarm_robosuite"

    def test_plan_pulls_robocasa_group(self) -> None:
        """The single install step is `uv sync --group robocasa --inexact`.

        That's where the workspace declares `robosuite>=1.5`; openarm
        does not need the robocasa kitchen/GR1 repo clones, so the plan
        is intentionally one step (no editable installs, no shadow
        cleanup, no mink / qpsolvers).
        """
        plan = _openarm_robosuite_plan()
        assert len(plan.steps) == 1, [s.description for s in plan.steps]
        argv = plan.steps[0].argv
        assert argv[1:6] == ["sync", "--all-packages", "--group", "robocasa", "--inexact"]

    def test_manual_hint_matches_yaml_setup_block(self) -> None:
        """`manual_hint` must mirror the install command documented in the example YAML.

        ``scenes/sim/openarm_tabletop.yaml`` says
        ``just sync --all-packages --group robocasa     # pulls
        robosuite>=1.5``; the plan's hint is what the user sees in the
        ROSConfigError when they decline the prompt, so the two must
        agree. ``just sync`` (not bare ``uv sync``) is required because
        ``uv sync`` without ``--all-packages`` would uninstall every
        workspace member; ``just sync`` also repairs the hf-libero
        distutils-uninstall trap before+after.
        """
        plan = _openarm_robosuite_plan()
        assert "just sync --all-packages" in plan.manual_hint
        assert "--group robocasa" in plan.manual_hint
        assert "--inexact" in plan.manual_hint

    def test_scene_factory_calls_ensure_backend_deps_before_robosuite_import(self) -> None:
        """The SCENES factory must call ensure_backend_deps('openarm_robosuite') first.

        Cheaper than spinning up MuJoCo: read the source of
        ``_build_openarm_tabletop_scene`` and confirm the call exists
        BEFORE ``from robosuite...`` so a fresh venv gets the prompt
        instead of ``ModuleNotFoundError: robosuite``.
        """
        import inspect

        from openral_sim.backends.openarm_robosuite import env as openarm_env

        src = inspect.getsource(openarm_env._build_openarm_tabletop_scene)
        ensure_idx = src.find('ensure_backend_deps("openarm_robosuite")')
        robosuite_import_idx = src.find("from robosuite")
        assert ensure_idx != -1, (
            "openarm scene factory must call ensure_backend_deps('openarm_robosuite')"
        )
        assert robosuite_import_idx != -1, "openarm scene factory must import from robosuite"
        assert ensure_idx < robosuite_import_idx, (
            "ensure_backend_deps must run BEFORE the robosuite import so a venv "
            "without --group robocasa gets the install prompt rather than a bare "
            "ModuleNotFoundError"
        )


# ─── Robocasa plans use --inexact (transitively closes Fix #3) ────────────


class TestRobocasaPlansUseInexact:
    @pytest.mark.parametrize(
        "plan_fn", [_robocasa_kitchen_plan, _robocasa_gr1_plan], ids=["kitchen", "gr1"]
    )
    def test_first_step_uses_inexact(self, plan_fn) -> None:  # type: ignore[no-untyped-def]
        """Both robocasa plans must use --inexact on their leading uv sync.

        Without it, `uv sync --group robocasa` uninstalls every package
        not in the robocasa group's resolved set — including pyzmq +
        msgpack from --group rldx (Fix #3), bringing the rldx adapter
        down. With --inexact, sibling-group deps survive.
        """
        plan = plan_fn()
        sync_step = plan.steps[0]
        assert sync_step.argv[0:3] == [sync_step.argv[0], "sync", "--all-packages"]
        assert "--inexact" in sync_step.argv


# ─── Issue #44: both robocasa plans pin robosuite to one commit ───────────


class TestRobocasaPlansPinRobosuite:
    """Both robocasa forks must install robosuite at a *pinned* commit.

    Regression for issue #44. The kitchen + GR1 forks share the editable
    robosuite-master clone. The GR1 fork (robocasa-gr1-tabletop-tasks
    0.2.0) only supports robosuite 1.5.0/1.5.1; riding floating master
    means a future master commit that refactors the robot base-class API
    breaks the GR1 env build with ``Invalid base type to add to robot!``
    while the kitchen fork keeps working — and master always reports
    ``"1.5.2"`` so the two are indistinguishable by version string.
    Pinning both plans to one verified commit makes the build
    deterministic and lets the forks share a single robosuite install.

    These assert the plan SHAPE (the clone step pins a 40-char SHA, the
    manual hint mirrors it). The commit itself is validated end-to-end on
    the GPU host by ``tests/sim/test_gr1_rldx_robocasa.py``.
    """

    def test_pin_is_an_immutable_full_sha(self) -> None:
        """The pin must be a 40-char hex commit, never a tag or branch.

        A tag/branch would re-introduce the drift the pin exists to kill.
        """
        assert len(_ROBOSUITE_PIN) == 40, _ROBOSUITE_PIN
        assert all(c in "0123456789abcdef" for c in _ROBOSUITE_PIN), _ROBOSUITE_PIN

    @pytest.mark.parametrize(
        "plan_fn", [_robocasa_kitchen_plan, _robocasa_gr1_plan], ids=["kitchen", "gr1"]
    )
    def test_robosuite_clone_step_pins_the_commit(self, plan_fn) -> None:  # type: ignore[no-untyped-def]
        """Exactly one step clones robosuite, and it checks out the pinned SHA.

        Guards against a regression to the old floating-master clone
        (``[ -d … ] || git clone --depth=1 …robosuite.git`` with no
        subsequent checkout): the clone step must both clone *and* pin.
        """
        plan = plan_fn()
        clone_cmds = [
            s.argv[-1]
            for s in plan.steps
            if s.argv and "robosuite.git" in s.argv[-1] and "robosuite_models" not in s.argv[-1]
        ]
        assert len(clone_cmds) == 1, [s.description for s in plan.steps]
        cmd = clone_cmds[0]
        assert _ROBOSUITE_PIN in cmd, cmd
        assert f"checkout --quiet --detach {_ROBOSUITE_PIN}" in cmd, cmd

    @pytest.mark.parametrize(
        "plan_fn", [_robocasa_kitchen_plan, _robocasa_gr1_plan], ids=["kitchen", "gr1"]
    )
    def test_manual_hint_mirrors_the_pin(self, plan_fn) -> None:  # type: ignore[no-untyped-def]
        """The user-facing manual hint must check out the same pinned commit."""
        plan = plan_fn()
        assert _ROBOSUITE_PIN in plan.manual_hint
        assert f"checkout --detach {_ROBOSUITE_PIN}" in plan.manual_hint


# ─── Fix #5: refresh sys.meta_path after an in-process editable swap ──────


class TestRefreshEditableFinders:
    """Regression for the robocasa_gr1 install probe failure.

    setuptools-editable installs an ``__editable___<pkg>_<ver>_finder.py``
    + ``__editable__.<pkg>-<ver>.pth`` shim that registers an
    ``_EditableFinder`` on ``sys.meta_path`` with a ``MAPPING`` dict
    baked in at import time. When ``uv pip install -e`` swaps an
    editable install mid-process (e.g. the GR1 plan steps from the
    kitchen fork ``robocasa==1.0.1`` to the GR1 fork
    ``robocasa==0.2.0``), the old finder stays on ``sys.meta_path``
    with its stale MAPPING. ``importlib.invalidate_caches()`` does NOT
    refresh ``sys.meta_path``, so the post-install probe falsely
    reports the install never landed.

    Real-component: we build two tiny editable packages in a tmp dir
    and exercise ``uv pip install -e`` against the live venv to
    reproduce the swap. No mocks; the helper runs against the real
    ``sys.meta_path`` and the real .pth shims uv writes.
    """

    PKG_NAME = "openral_test_refresh_finders_xyz_42"

    @staticmethod
    def _build_pkg(root: Path, marker: str, version: str, name: str) -> Path:
        pkg_dir = root / "src" / name
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text(f"MARKER = {marker!r}\n")
        (root / "pyproject.toml").write_text(
            "[build-system]\n"
            'requires = ["setuptools>=61"]\n'
            'build-backend = "setuptools.build_meta"\n'
            "\n"
            "[project]\n"
            f'name = "{name}"\n'
            f'version = "{version}"\n'
            "\n"
            "[tool.setuptools.packages.find]\n"
            'where = ["src"]\n'
        )
        return root

    @staticmethod
    def _flush(pkg: str) -> None:
        import sys

        for mod_name in list(sys.modules):
            if mod_name == pkg or mod_name.startswith(pkg + "."):
                sys.modules.pop(mod_name, None)

    @pytest.mark.slow
    def test_refresh_picks_up_new_editable_and_resolves_swap(self, tmp_path: Path) -> None:
        """A runtime ``uv pip install -e`` only resolves via find_spec after refresh."""
        import importlib
        import importlib.util
        import subprocess
        import sys

        pkg = self.PKG_NAME
        a = self._build_pkg(tmp_path / "a", "marker-A", "0.0.1", pkg)
        b = self._build_pkg(tmp_path / "b", "marker-B", "0.0.2", pkg)

        try:
            # First install: find_spec returns None until we refresh.
            subprocess.run(
                ["uv", "pip", "install", "--no-deps", "-e", str(a)],
                check=True,
                capture_output=True,
            )
            importlib.invalidate_caches()
            self._flush(pkg)
            assert importlib.util.find_spec(pkg) is None, (
                "Runtime-installed editable .pth is invisible without site.addsitedir / refresh; "
                "if this assertion ever flips, uv changed its install semantics and the helper "
                "may be redundant for this case."
            )
            _refresh_editable_finders()
            spec_a = importlib.util.find_spec(pkg)
            assert spec_a is not None and spec_a.origin is not None
            assert str(tmp_path / "a") in spec_a.origin, spec_a.origin

            # Swap to B: without refresh, find_spec sticks to A; with refresh, follows B.
            subprocess.run(
                ["uv", "pip", "install", "--force-reinstall", "--no-deps", "-e", str(b)],
                check=True,
                capture_output=True,
            )
            importlib.invalidate_caches()
            self._flush(pkg)
            spec_stale = importlib.util.find_spec(pkg)
            assert spec_stale is not None and spec_stale.origin is not None
            assert str(tmp_path / "a") in spec_stale.origin, (
                "expected stale A path before refresh (this is the bug the helper fixes); "
                f"got {spec_stale.origin}"
            )

            _refresh_editable_finders()
            spec_fresh = importlib.util.find_spec(pkg)
            assert spec_fresh is not None and spec_fresh.origin is not None
            assert str(tmp_path / "b") in spec_fresh.origin, spec_fresh.origin
        finally:
            subprocess.run(["uv", "pip", "uninstall", pkg], capture_output=True, check=False)
            # Drop any meta_path finder we might have registered for this pkg.
            for finder in list(sys.meta_path):
                mod_name = getattr(finder, "__module__", "")
                if isinstance(mod_name, str) and pkg in mod_name:
                    sys.meta_path.remove(finder)
                    sys.modules.pop(mod_name, None)
            for path_entry in list(sys.path):
                if str(tmp_path) in path_entry:
                    sys.path.remove(path_entry)
            self._flush(pkg)


# ─── Concurrent-prompt serialisation ──────────────────────────────────────


class TestEnsureBackendDepsLock:
    """``ensure_backend_deps`` must serialise prompts across threads.

    ``SimRunner._build_env_and_policy`` spins up env + policy on a
    2-worker ``ThreadPoolExecutor`` (see ``python/sim/src/openral_sim/sim_runner.py``).
    When both sides need a first-install (the user-reported case was
    ``sim run --config simpler_env_widowx_carrot.yaml --rskill
    rldx1-ft-simpler-widowx-nf4``), both threads raced into
    ``ensure_backend_deps`` and interleaved their Rich banners +
    ``typer.confirm`` reads — only one of the two prompts actually
    consumed the user's ``y``. The module-level ``_INSTALL_LOCK`` fixes
    this; this test exercises the contract.
    """

    def test_two_threads_do_not_interleave_typer_confirm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two concurrent ensure_backend_deps calls must not overlap inside typer.confirm."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        import openral_sim._deps as deps_mod
        from openral_sim._deps import BackendInstallPlan, ensure_backend_deps

        # Hermetic: this test verifies the typer.confirm PROMPT path. The
        # default is now auto-install (OPENRAL_AUTO_INSTALL_DEPS defaults to
        # "1"), so explicitly set "0" to force the confirm branch under test.
        monkeypatch.setenv("OPENRAL_AUTO_INSTALL_DEPS", "0")

        # Two independent plans (mirroring simpler_env + rldx_client),
        # each with a probe that flips True after the prompt is acknowledged
        # so the install-step loop is a no-op on this host.
        installed: dict[str, bool] = {"alpha": False, "beta": False}

        def make_plan(bid: str) -> BackendInstallPlan:
            return BackendInstallPlan(
                backend_id=bid,
                display_name=f"{bid} (test)",
                license_note="test-only",
                probe=lambda: installed[bid],
                steps=(),
                manual_hint=f"manual hint for {bid}",
            )

        monkeypatch.setitem(deps_mod._PLANS, "alpha", lambda: make_plan("alpha"))
        monkeypatch.setitem(deps_mod._PLANS, "beta", lambda: make_plan("beta"))

        # The interleaving is driven by explicit synchronization primitives
        # (Barrier / Event / an instrumented Lock) instead of a wall-clock
        # ``time.sleep`` race window. The old version slept 0.05 s inside the
        # prompt and *hoped* the scheduler overlapped the two threads within
        # that budget — under full-suite load thread B was often not even
        # scheduled before the sleep elapsed, so the race was never exercised
        # and the assertion passed for the wrong reason (or, with timing skew,
        # could spuriously interleave). Here the window is opened and closed
        # deterministically:
        #
        #   1. ``both_started`` (Barrier) rendezvous: both worker threads are
        #      live before either calls ``ensure_backend_deps``.
        #   2. ``_INSTALL_LOCK`` is replaced with ``InstrumentedLock``, which
        #      fires ``second_acquire_blocking`` the instant a SECOND thread
        #      blocks on ``acquire()`` — i.e. the holder is in the prompt and
        #      thread B is provably parked at the lock. No polling, no sleep.
        #   3. The lock holder parks inside ``fake_confirm`` on
        #      ``release_holder`` until the main thread has observed (2), so
        #      the prompt is held open exactly across B's contention attempt.
        #
        # If ``_INSTALL_LOCK`` failed to serialise, B would fall through into
        # ``fake_confirm`` and ``max_active`` would reach 2 — the assertion
        # that still encodes the contract, unchanged.
        active = 0
        max_active = 0
        active_lock = threading.Lock()
        confirm_entered = threading.Event()
        release_holder = threading.Event()
        second_acquire_blocking = threading.Event()

        class InstrumentedLock:
            """``_INSTALL_LOCK`` stand-in that flags a contended acquire.

            Wraps a real :class:`threading.Lock` so production semantics are
            unchanged. When ``acquire`` is called while the lock is already
            held by another thread, the caller is about to block; we set
            ``second_acquire_blocking`` first so the test knows thread B has
            reached the lock and is parked there — the deterministic signal
            that replaces the old timing window.
            """

            def __init__(self) -> None:
                self._lock = threading.Lock()

            def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
                if self._lock.locked():
                    # Lock is already held; this acquire will block. Signal
                    # before blocking so the waiter is observable. ``locked()``
                    # reflects the real lock state with no separate flag to go
                    # stale, so there is no window between "held" and "signal".
                    second_acquire_blocking.set()
                return self._lock.acquire(blocking, timeout)

            def release(self) -> None:
                self._lock.release()

            def __enter__(self) -> bool:
                return self.acquire()

            def __exit__(self, *exc: object) -> None:
                self.release()

        monkeypatch.setattr(deps_mod, "_INSTALL_LOCK", InstrumentedLock())

        def fake_confirm(prompt: str, default: bool = False) -> bool:
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                # Mark the corresponding plan as installed so the post-
                # install probe inside ensure_backend_deps succeeds.
                for bid in installed:
                    if bid in prompt:
                        installed[bid] = True
                        break
                # The first thread to reach the prompt is the lock holder. It
                # parks here (on an Event, not a sleep) until the main thread
                # has confirmed the second thread is blocked on _INSTALL_LOCK,
                # holding the prompt open across B's contention attempt.
                if not confirm_entered.is_set():
                    confirm_entered.set()
                    assert release_holder.wait(timeout=5.0), (
                        "main thread never released the prompt holder"
                    )
                return True
            finally:
                with active_lock:
                    active -= 1

        monkeypatch.setattr(deps_mod.typer, "confirm", fake_confirm)
        # Suppress the Rich banner in test output.
        monkeypatch.setattr(deps_mod, "_display_install_banner", lambda plan: None)

        errors: list[BaseException] = []
        both_started = threading.Barrier(2, timeout=5.0)

        def call(bid: str) -> None:
            try:
                # Rendezvous so both threads are live before either races into
                # ensure_backend_deps — removes the scheduler-timing dependency
                # that made the old sleep-based version flaky.
                both_started.wait()
                ensure_backend_deps(bid)
            except BaseException as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(call, "alpha")
            f2 = pool.submit(call, "beta")

            # One thread is now parked inside the prompt holding _INSTALL_LOCK.
            assert confirm_entered.wait(timeout=5.0), (
                "no thread entered typer.confirm — ensure_backend_deps never reached the prompt"
            )
            # The second thread has reached _INSTALL_LOCK and is blocked on it
            # (it cannot enter fake_confirm while the holder owns the lock).
            assert second_acquire_blocking.wait(timeout=5.0), (
                "second thread never contended for _INSTALL_LOCK — the race "
                "window was not exercised"
            )
            # Release the holder; the second thread then acquires the lock,
            # re-probes, runs its own (serialised) prompt, and finishes.
            release_holder.set()

            f1.result(timeout=5.0)
            f2.result(timeout=5.0)

        assert not errors, f"ensure_backend_deps raised: {errors!r}"
        assert max_active == 1, (
            f"typer.confirm was entered by {max_active} threads concurrently; "
            "ensure_backend_deps must serialise prompts via _INSTALL_LOCK"
        )
        assert installed == {"alpha": True, "beta": True}
