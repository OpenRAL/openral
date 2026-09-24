"""The Fast-DDS shm clean only unlinks files no live process uses.

``openral deploy sim`` used to unlink every ``/dev/shm/fastrtps_*`` the user
owned, silencing a ZED driver started before it on Thor (2026-09-24). These
tests drive ``_clean_stale_fastrtps_shm`` against a temp dir and the real
``/proc``, with real child processes holding the files open, mapped or locked.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from openral_cli.deploy_sim import (
    DDS_TRANSPORT_READY_MARKER,
    FASTDDS_SHM_CLEAN_ENV,
    _apply_rmw_default,
    _clean_stale_fastrtps_shm,
    _fastrtps_group,
    _in_initial_pid_namespace,
)

# Liveness is only provable from the initial PID namespace; inside a container the clean
# (correctly) keeps everything, so the purge assertions below cannot hold there.
_needs_init_pidns = pytest.mark.skipif(
    not _in_initial_pid_namespace(),
    reason="not in the initial PID namespace (container): the clean keeps every file here",
)

# Holds <path> the way argv[1] says, prints "ready", and waits for stdin to close.
_HOLDER = r"""
import ctypes, fcntl, mmap, os, sys
mode, path = sys.argv[1], sys.argv[2]
if mode == "nodump-flock":
    ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 4
fd = os.open(path, os.O_RDWR)
if mode == "mmap":
    m = mmap.mmap(fd, 4096)
    os.close(fd)
elif mode == "nodump-flock":
    fcntl.flock(fd, fcntl.LOCK_EX)
print("ready", flush=True)
sys.stdin.read()
"""


def _hold(mode: str, path: Path) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, mode, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    return proc


def _release(proc: subprocess.Popen[str]) -> None:
    assert proc.stdin is not None
    proc.stdin.close()
    proc.wait(timeout=10)


@pytest.fixture
def shm(tmp_path: Path) -> Iterator[Path]:
    """A participant segment + its lock, and a stale port + its lock."""
    for name in ("fastrtps_ab12cd", "fastrtps_port7400"):
        (tmp_path / name).write_bytes(b"\0" * 4096)
        (tmp_path / f"{name}_el").touch()
    (tmp_path / "unrelated").touch()
    yield tmp_path


def _fake_proc(factory: pytest.TempPathFactory, pidns: str) -> Path:
    """A procfs stand-in: ``self/ns/pid`` naming ``pidns`` and an empty ``locks``."""
    root = factory.mktemp("proc")
    (root / "self" / "ns").mkdir(parents=True)
    (root / "self" / "ns" / "pid").symlink_to(pidns)
    (root / "locks").touch()
    return root


def _left(shm_dir: Path) -> set[str]:
    return {p.name for p in shm_dir.iterdir()}


def test_group_strips_lock_suffixes() -> None:
    assert _fastrtps_group("fastrtps_port7000_el") == "fastrtps_port7000"
    assert _fastrtps_group("fastrtps_ab12_sl") == "fastrtps_ab12"
    assert _fastrtps_group("fastrtps_port7000") == "fastrtps_port7000"


@_needs_init_pidns
@pytest.mark.parametrize("mode", ["open", "mmap"])
def test_open_or_mapped_group_is_kept_then_purged_once_released(shm: Path, mode: str) -> None:
    holder = _hold(mode, shm / "fastrtps_ab12cd")
    try:
        assert _clean_stale_fastrtps_shm(shm) == (2, 2)
        assert _left(shm) == {"fastrtps_ab12cd", "fastrtps_ab12cd_el", "unrelated"}
    finally:
        _release(holder)
    assert _clean_stale_fastrtps_shm(shm) == (2, 0)
    assert _left(shm) == {"unrelated"}


def test_locked_file_of_unreadable_process_is_kept(shm: Path) -> None:
    """A non-dumpable process hides its fd/maps; its flock still proves it live."""
    holder = _hold("nodump-flock", shm / "fastrtps_ab12cd_el")
    try:
        try:
            os.listdir(f"/proc/{holder.pid}/fd")
        except PermissionError:
            pass
        else:
            pytest.skip(reason="running with CAP_SYS_PTRACE; cannot make /proc/<pid>/fd unreadable")
        assert _clean_stale_fastrtps_shm(shm) == (2, 2)
        assert _left(shm) == {"fastrtps_ab12cd", "fastrtps_ab12cd_el", "unrelated"}
    finally:
        _release(holder)


def test_unreadable_proc_locks_keeps_everything(
    shm: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    proc_root = _fake_proc(tmp_path_factory, "pid:[4026531836]")
    locks = proc_root / "locks"
    locks.chmod(0)
    if os.access(locks, os.R_OK):
        pytest.skip(reason="running as root; chmod 0 does not make the file unreadable")
    assert _clean_stale_fastrtps_shm(shm, proc_root) == (0, 4)
    assert len(_left(shm)) == 5


@_needs_init_pidns
def test_fastdds_3_names_are_cleaned_too(tmp_path: Path) -> None:
    """Fast-DDS 3.x renamed the shm domain ``fastrtps`` -> ``fastdds``."""
    for name in ("fastdds_ab12cd", "fastdds_port7400"):
        (tmp_path / name).write_bytes(b"\0" * 4096)
        (tmp_path / f"{name}_el").touch()
    holder = _hold("open", tmp_path / "fastdds_ab12cd")
    try:
        assert _clean_stale_fastrtps_shm(tmp_path) == (2, 2)
        assert _left(tmp_path) == {"fastdds_ab12cd", "fastdds_ab12cd_el"}
    finally:
        _release(holder)
    assert _fastrtps_group("fastdds_port7400_sl") == "fastdds_port7400"


def test_outside_the_initial_pid_namespace_nothing_is_unlinked(
    shm: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """In a container, a driver sharing /dev/shm from another PID namespace is invisible to
    /proc/locks and /proc/<pid>: its files would look stale. Keep everything."""
    proc_root = _fake_proc(tmp_path_factory, "pid:[4026532999]")
    assert not _in_initial_pid_namespace(proc_root)
    assert _clean_stale_fastrtps_shm(shm, proc_root) == (0, 4)
    assert len(_left(shm)) == 5


def test_initial_pid_namespace_with_no_holders_purges(
    shm: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Same fake procfs, initial namespace: nothing holds the files, so they go."""
    proc_root = _fake_proc(tmp_path_factory, "pid:[4026531836]")
    assert _clean_stale_fastrtps_shm(shm, proc_root) == (4, 0)
    assert _left(shm) == {"unrelated"}


def test_env_opt_out_skips_the_clean(capsys: pytest.CaptureFixture[str]) -> None:
    _apply_rmw_default({FASTDDS_SHM_CLEAN_ENV: "0"})
    out = capsys.readouterr().out
    assert DDS_TRANSPORT_READY_MARKER in out
    assert "shm_purged=n/a" in out
