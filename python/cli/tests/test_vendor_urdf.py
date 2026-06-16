"""Unit tests for ``openral robot vendor-urdf`` (ADR-0057 description vendoring).

The command expands an upstream xacro to a flat, committed URDF so end users
need no xacro tooling at runtime. These tests exercise the real expander
(``robot_descriptions`` + ``xacrodoc`` + ``yourdfpy``); they skip cleanly when
that toolchain is unavailable (e.g. CI without the ``lowering`` group).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("xacrodoc")
pytest.importorskip("yourdfpy")

from openral_cli.robot import vendor_urdf


def test_vendor_ur5e_writes_flat_urdf(tmp_path: Path) -> None:
    out = vendor_urdf("ur5e", upstream="rd:ur5e_description", out_dir=tmp_path)
    text = out.read_text()
    assert "${" not in text  # xacro fully expanded
    assert '<joint name="' in text
    assert out.name == "ur5e.urdf"


def test_vendor_ur5e_has_provenance_header(tmp_path: Path) -> None:
    out = vendor_urdf("ur5e", upstream="rd:ur5e_description", out_dir=tmp_path)
    text = out.read_text()
    assert "Vendored by" in text
    assert "rd:ur5e_description" in text
    assert "0057" in text


def test_vendor_ur5e_joint_names_match_manifest(tmp_path: Path) -> None:
    out = vendor_urdf("ur5e", upstream="rd:ur5e_description", out_dir=tmp_path)
    text = out.read_text()
    for joint in ("shoulder_pan_joint", "elbow_joint", "wrist_3_joint"):
        assert f'name="{joint}"' in text


def test_vendor_ur5e_output_is_well_formed_xml(tmp_path: Path) -> None:
    """The provenance comment must not precede the XML declaration (else the
    document is not well-formed and every URDF parser rejects it)."""
    import xml.dom.minidom

    out = vendor_urdf("ur5e", upstream="rd:ur5e_description", out_dir=tmp_path)
    dom = xml.dom.minidom.parse(str(out))  # raises ExpatError if malformed
    assert dom.getElementsByTagName("robot")[0].getAttribute("name") == "ur5e"


def test_vendor_openarm_strips_prefix(tmp_path: Path) -> None:
    """File-path upstream + rename: ``openarm_`` prefix stripped from references."""
    src = Path("/tmp/openarm_description/output.urdf")
    if not src.exists():
        pytest.skip("openarm upstream not cloned to /tmp/openarm_description")
    out = vendor_urdf(
        "openarm",
        upstream=f"file:{src}",
        out_dir=tmp_path,
        rename=(r'"openarm_', '"'),
    )
    text = out.read_text()
    assert "${" not in text
    assert 'name="left_joint1"' in text
    assert 'name="right_joint1"' in text
    assert '"openarm_left_joint1"' not in text
