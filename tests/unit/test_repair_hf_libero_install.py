"""Unit tests for ``scripts/repair_hf_libero_install.py``.

The script removes a spurious ``hf_libero-<ver>.egg-info`` FILE that the
PyPI sdist drops alongside the proper ``.dist-info/`` directory and
which makes ``uv sync`` fall back to a failing distutils uninstall
path. These tests build a synthetic venv-shaped tree under ``tmp_path``
(no mocks, no monkey-patching — CLAUDE.md §1.11 / §5.4) and assert the
script's pre / post invariants directly on the filesystem.

Pins:

* Broken install (``.egg-info`` regular file + RECORD line) is repaired
  — the file is gone, the RECORD line is gone, every other RECORD line
  survives, every other file in the dist-info is untouched.
* No-op when ``hf-libero`` isn't installed.
* No-op when ``.egg-info`` is already absent (so a second invocation is
  cheap and side-effect-free).
* Real setuptools ``.egg-info`` *directory* (different shape) is left
  alone — the repair only targets the FILE shape produced by the sdist.
* Multiple ``hf_libero-*.dist-info`` versions side-by-side (unusual but
  possible during a botched reinstall) are all repaired in one pass.
* Trailing newline in RECORD is preserved when present and not
  fabricated when absent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Resolve the script path once. The script lives at ``scripts/`` next to
# the repo root; tests/unit runs with cwd=repo-root.
_REPAIR_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "repair_hf_libero_install.py"


def _make_venv(tmp_path: Path) -> Path:
    """Return a synthetic ``<venv>/lib/python3.12/site-packages`` path."""
    sp = tmp_path / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    return sp


def _make_broken_install(site_packages: Path, version: str = "0.1.3") -> tuple[Path, Path]:
    """Create the malformed install shape: ``.dist-info/`` + ``.egg-info`` FILE.

    Returns ``(dist_info_dir, egg_info_file)``.
    """
    dist_info = site_packages / f"hf_libero-{version}.dist-info"
    dist_info.mkdir()
    metadata = dist_info / "METADATA"
    metadata.write_text(
        "Metadata-Version: 2.1\nName: hf-libero\nVersion: 0.1.3\n",
    )
    # The bogus egg-info file is a byte-for-byte duplicate of METADATA in
    # the real-world bug, but the repair only cares about the file's
    # existence + the RECORD line — content is incidental.
    egg_info = site_packages / f"hf_libero-{version}.egg-info"
    egg_info.write_text(metadata.read_text())

    record = dist_info / "RECORD"
    record.write_text(
        "hf_libero-0.1.3.dist-info/INSTALLER,sha256=AAA,1\n"
        "hf_libero-0.1.3.dist-info/METADATA,sha256=BBB,200\n"
        "hf_libero-0.1.3.dist-info/RECORD,,\n"
        f"hf_libero-{version}.egg-info,sha256=CCC,200\n"
        "libero/__init__.py,sha256=DDD,1\n"
    )
    return dist_info, egg_info


def _run_repair(venv_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run the repair script against ``venv_dir``."""
    return subprocess.run(  # reason: invoking our own script with a fixed path
        [sys.executable, str(_REPAIR_SCRIPT), "--venv", str(venv_dir)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_repair_fixes_broken_install(tmp_path: Path) -> None:
    """The egg-info file is removed AND the matching RECORD line is stripped."""
    sp = _make_venv(tmp_path)
    dist_info, egg_info = _make_broken_install(sp)

    proc = _run_repair(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "removed_egg_info_file=True" in proc.stdout
    assert "stripped_record_line=True" in proc.stdout
    assert not egg_info.exists()
    record_lines = (dist_info / "RECORD").read_text().splitlines()
    assert not any(ln.startswith("hf_libero-0.1.3.egg-info,") for ln in record_lines)
    # Every other RECORD line survives.
    assert "hf_libero-0.1.3.dist-info/INSTALLER,sha256=AAA,1" in record_lines
    assert "hf_libero-0.1.3.dist-info/METADATA,sha256=BBB,200" in record_lines
    assert "libero/__init__.py,sha256=DDD,1" in record_lines


def test_repair_no_op_when_hf_libero_absent(tmp_path: Path) -> None:
    """No hf-libero install → exit 0, no output, no side effects."""
    sp = _make_venv(tmp_path)
    sp.joinpath("some_other_pkg-1.0.dist-info").mkdir()

    proc = _run_repair(tmp_path)

    assert proc.returncode == 0
    assert proc.stdout == ""
    assert proc.stderr == ""
    # Sibling package's dist-info is untouched.
    assert (sp / "some_other_pkg-1.0.dist-info").is_dir()


def test_repair_no_op_when_egg_info_already_absent(tmp_path: Path) -> None:
    """Idempotent — repairing twice in a row produces no second-pass changes."""
    sp = _make_venv(tmp_path)
    _make_broken_install(sp)

    first = _run_repair(tmp_path)
    assert first.returncode == 0
    assert "removed_egg_info_file=True" in first.stdout

    second = _run_repair(tmp_path)
    assert second.returncode == 0
    assert second.stdout == ""


def test_repair_leaves_real_setuptools_egg_info_directory_alone(tmp_path: Path) -> None:
    """A real ``*.egg-info`` *directory* (setuptools-built) is not touched."""
    sp = _make_venv(tmp_path)
    dist_info = sp / "hf_libero-0.1.3.dist-info"
    dist_info.mkdir()
    (dist_info / "RECORD").write_text(
        "hf_libero-0.1.3.dist-info/RECORD,,\nlibero/__init__.py,sha256=DDD,1\n",
    )
    real_egg = sp / "hf_libero-0.1.3.egg-info"
    real_egg.mkdir()
    (real_egg / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: hf-libero\n")

    proc = _run_repair(tmp_path)

    assert proc.returncode == 0
    assert real_egg.is_dir(), "real setuptools .egg-info directory must survive"
    assert (real_egg / "PKG-INFO").is_file()
    assert "directory" in proc.stderr  # the explanatory message fires


def test_repair_handles_multiple_versions_in_one_pass(tmp_path: Path) -> None:
    """Two side-by-side broken installs (different versions) both get repaired."""
    sp = _make_venv(tmp_path)
    _, egg_v1 = _make_broken_install(sp, version="0.1.3")
    _, egg_v2 = _make_broken_install(sp, version="0.2.0")

    proc = _run_repair(tmp_path)

    assert proc.returncode == 0
    assert not egg_v1.exists()
    assert not egg_v2.exists()
    # Both repairs were logged.
    assert proc.stdout.count("removed_egg_info_file=True") == 2


def test_repair_preserves_record_trailing_newline(tmp_path: Path) -> None:
    """RECORD ending with ``\\n`` keeps its trailing newline post-repair."""
    sp = _make_venv(tmp_path)
    dist_info, _ = _make_broken_install(sp)
    assert (dist_info / "RECORD").read_text().endswith("\n")

    _run_repair(tmp_path)

    assert (dist_info / "RECORD").read_text().endswith("\n")


def test_repair_returns_zero_when_venv_missing(tmp_path: Path) -> None:
    """Pointing at a path without ``lib/`` is a silent no-op (exit 0)."""
    empty = tmp_path / "no-venv-here"
    empty.mkdir()

    proc = _run_repair(empty)

    assert proc.returncode == 0
    assert proc.stdout == ""
