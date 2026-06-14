"""Tests for the shared URDF-path resolver (lifted from sim_e2e.launch.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
from openral_core.urdf_resolve import resolve_urdf_path


def test_resolves_python_module_attribute_reference() -> None:
    pytest.importorskip("robot_descriptions")
    resolved = resolve_urdf_path("python:robot_descriptions.panda_description:URDF_PATH")
    assert resolved is not None
    assert Path(resolved).is_file()


def test_absolute_path_passthrough(tmp_path: Path) -> None:
    f = tmp_path / "robot.urdf"
    f.write_text("<robot name='x'/>", encoding="utf-8")
    assert resolve_urdf_path(str(f)) == str(f)


def test_missing_file_returns_none() -> None:
    assert resolve_urdf_path("/nonexistent/abs/path.urdf") is None


def test_malformed_python_reference_returns_none() -> None:
    assert resolve_urdf_path("python:onlyonecolon") is None
