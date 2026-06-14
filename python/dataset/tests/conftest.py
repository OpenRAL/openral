"""Pytest fixtures for openral_dataset tests.

Per CLAUDE.md §1.11 — fixtures load real RobotDescription manifests
from ``robots/`` and never substitute a fake. Tests that need lerobot
``pytest.skip`` with a typed reason when it isn't installed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from openral_core import RobotDescription


def _find_repo_root(start: Path) -> Path:
    """Walk upward from ``start`` until a sibling ``robots/`` directory exists."""
    for candidate in (start, *start.parents):
        if (candidate / "robots").is_dir() and (candidate / "python").is_dir():
            return candidate
    raise RuntimeError(
        f"could not locate repo root walking up from {start!r}; "
        f"expected to find both 'robots/' and 'python/' siblings"
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the OpenRAL repo root (parent of ``robots/``)."""
    return _find_repo_root(Path(__file__).resolve())


@pytest.fixture(scope="session")
def so100_robot(repo_root: Path) -> RobotDescription:
    """Real SO-100 follower :class:`RobotDescription` from ``robots/so100_follower/``.

    6 joints, 6-D action, 2 RGB cameras. Used by every recorder/sink test.
    """
    return RobotDescription.from_yaml(str(repo_root / "robots" / "so100_follower" / "robot.yaml"))


@pytest.fixture(scope="session")
def aloha_robot(repo_root: Path) -> RobotDescription:
    """Real Aloha bimanual :class:`RobotDescription` from ``robots/aloha_bimanual/``.

    14 joints (7+7 bimanual), 14-D action, 1 RGB camera (`top`), 50 Hz.
    Used by multi-robot smoke tests to prove the bridge isn't
    SO-100-specific.
    """
    return RobotDescription.from_yaml(str(repo_root / "robots" / "aloha_bimanual" / "robot.yaml"))


@pytest.fixture
def has_lerobot() -> bool:
    """True iff ``lerobot>=0.5.1`` is importable in this venv."""
    try:
        import lerobot  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def require_video_decode() -> Callable[[], None]:
    """Return a probe that skips the test unless lerobot's video backend can decode.

    Reading a frame back from a written ``LeRobotDataset`` (``ds[i]``) makes
    lerobot decode the recorded MP4 via ``torchcodec.decoders.VideoDecoder``,
    which loads ``libtorchcodec`` against a system FFmpeg on first use. That
    load fails when torchcodec is absent (``ImportError``) or when its native
    libs don't match the installed FFmpeg / PyTorch — a common CI-runner
    condition (e.g. PyTorch ``2.9.1+cu128`` vs the shipped TorchCodec). Per
    CLAUDE.md §1.11 an unavailable dependency is a typed ``pytest.skip``, not
    a failure: callers invoke the returned probe immediately before the first
    decode so every no-decode assertion (episode/frame counts, success-rate
    aggregates, parquet round-trip) stays covered on backend-less hosts.
    """

    def _check() -> None:
        try:
            from torchcodec.decoders import VideoDecoder  # noqa: F401
        except (ImportError, RuntimeError, OSError) as exc:
            head = str(exc).splitlines()[0] if str(exc) else ""
            pytest.skip(
                "lerobot video backend cannot decode on this host "
                f"({type(exc).__name__}: {head}); torchcodec/libtorchcodec "
                "unavailable or incompatible with the installed FFmpeg/PyTorch"
            )

    return _check
