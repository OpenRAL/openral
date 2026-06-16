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


# ── Raw-text mode (ADR-0057): joint-name-patched URDFs for so100/so101/gr1/h1 ──
#
# These exercise the real ``robot_descriptions`` cache. They skip cleanly when a
# given upstream module is not installed/cached.

pytest.importorskip("robot_descriptions")


def _read_upstream(rd_module: str) -> str:
    """Read an ``rd:`` module's cached URDF text verbatim (preserving CRLF)."""
    import importlib

    mod = importlib.import_module(f"robot_descriptions.{rd_module}")
    with open(mod.URDF_PATH, encoding="utf-8", newline="") as fh:
        return fh.read()


def test_vendor_so100_renames_numeric_joints(tmp_path: Path) -> None:
    """so100 numeric joints '1'..'6' → SO-ARM semantic HAL names; links untouched."""
    out = vendor_urdf(
        "so100_follower", upstream="rd:so_arm100_description", out_dir=tmp_path, raw_text=True
    )
    text = out.read_text()
    for sem in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"):
        assert f'<joint name="{sem}"' in text
    # No numeric joint name survives, and the semantic links are byte-preserved.
    import re

    assert not re.search(r'name="[1-6]"', text)
    assert '<link name="base"' in text and '<link name="jaw"' in text


def test_vendor_h1_strips_joint_suffix_preserves_package_meshes(tmp_path: Path) -> None:
    """h1 ``_joint`` suffix stripped; ``package://`` mesh paths preserved verbatim."""
    out = vendor_urdf("h1", upstream="rd:h1_description", out_dir=tmp_path, raw_text=True)
    text = out.read_text()
    assert '<joint name="torso"' in text and '<joint name="left_knee"' in text
    import re

    assert not re.search(r'name="[^"]*_joint"', text)
    # Mesh paths are preserved byte-for-byte (the round-trip would absolutize them).
    upstream = _read_upstream("h1_description")
    up_meshes = re.findall(r'filename="(package://[^"]*)"', upstream)
    assert up_meshes  # upstream really uses package:// meshes
    for m in up_meshes:
        assert f'filename="{m}"' in text


def test_vendor_gr1_collapses_elbow_to_manifest_name(tmp_path: Path) -> None:
    """gr1 ``*_elbow_pitch_joint`` → ``*_elbow`` (manifest HAL name); no link hit."""
    out = vendor_urdf("gr1", upstream="rd:gr1_description", out_dir=tmp_path, raw_text=True)
    text = out.read_text()
    assert '<joint\n    name="left_elbow"' in text.replace("\r\n", "\n")
    assert '<joint\n    name="right_elbow"' in text.replace("\r\n", "\n")
    assert "_elbow_pitch_joint" not in text


def test_vendor_raw_text_diff_is_joint_names_only(tmp_path: Path) -> None:
    """Raw-text vendoring changes ONLY joint ``name=`` attrs (+ provenance line).

    This is the safety-critical invariant: link names, geometry, inertials and
    mesh paths must be byte-identical to upstream so the committed
    ``collision_geometry`` (lowered from the URDF) cannot drift.
    """
    out = vendor_urdf("h1", upstream="rd:h1_description", out_dir=tmp_path, raw_text=True)
    vendored = out.read_text().replace("\r\n", "\n").splitlines()
    upstream = _read_upstream("h1_description").replace("\r\n", "\n").splitlines()

    def _strip(lines: list[str]) -> list[str]:
        # Drop the provenance comment and normalize the XML-declaration quoting;
        # everything else must match line-for-line apart from joint-name attrs.
        return [
            ln
            for ln in lines
            if "Vendored by" not in ln and not ln.lstrip().startswith("<?xml")
        ]

    v, u = _strip(vendored), _strip(upstream)
    assert len(v) == len(u)
    import re

    for vl, ul in zip(v, u, strict=True):
        if vl != ul:
            # The only permitted divergence is a joint-name attribute.
            assert re.search(r'name="[^"]*"', vl) and re.search(r'name="[^"]*"', ul), (vl, ul)


def test_vendor_raw_text_rejects_xacro(tmp_path: Path) -> None:
    """Raw-text mode refuses an unexpanded xacro (``${…}`` survivor) — no silent pass."""
    bad = tmp_path / "bad.urdf"
    bad.write_text('<?xml version="1.0"?>\n<robot name="x"><joint name="${j}"/></robot>')
    with pytest.raises(ValueError, match="flat URDF"):
        vendor_urdf("x", upstream=f"file:{bad}", out_dir=tmp_path, raw_text=True)
