"""_probe_nvmm_available — L4T libnvbufsurface.so detection.

ADR-0013 PR 2/3. The probe surfaces whether the NVMM zero-copy
sensor-ingest path is available on the host so
``rSkill.check_capabilities`` can refuse skills that require it on a
host that cannot provide it.
"""

from __future__ import annotations

from pathlib import Path

from openral_detect.probes.gpu import _probe_nvmm_available


def test_returns_false_when_lib_missing(tmp_path: Path) -> None:
    # tmp_path has no libnvbufsurface.so.
    assert _probe_nvmm_available(search_paths=[tmp_path]) is False


def test_returns_true_when_lib_present(tmp_path: Path) -> None:
    (tmp_path / "libnvbufsurface.so").write_bytes(b"")
    assert _probe_nvmm_available(search_paths=[tmp_path]) is True


def test_searches_multiple_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / "libnvbufsurface.so").write_bytes(b"")
    assert _probe_nvmm_available(search_paths=[first, second]) is True


def test_default_paths_safe_on_non_tegra() -> None:
    # On any non-Tegra dev host the canonical Tegra path will be absent;
    # the helper must NOT raise.
    result = _probe_nvmm_available()
    assert isinstance(result, bool)


def test_returns_false_when_search_paths_empty() -> None:
    assert _probe_nvmm_available(search_paths=[]) is False
