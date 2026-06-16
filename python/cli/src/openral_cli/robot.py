"""``openral robot vendor-urdf <id>`` — expand an upstream xacro to a flat URDF.

The flat, committed URDF means end users need no xacro tooling at runtime
(ADR-0058). The xacro-only arms (ur5e/ur10e/rizon4) ship a ``XACRO_PATH`` in
``robot_descriptions``; we let ``robot_descriptions``' ``yourdfpy`` loader run
``xacrodoc`` to expand every ``${…}`` substitution, then serialize the resulting
flat URDF with a provenance header. ``openarm`` ships only MJCF upstream, so its
flattened URDF is cloned separately and passed as a ``file:`` upstream; the
``--rename`` hook strips the ``openarm_`` joint/link prefix to the OpenRAL HAL
convention (``left_joint1..7`` / ``right_joint1..7``).

**Raw-text mode** (``raw_text=True``). The yourdfpy round-trip absolutizes /
mangles ``package://`` mesh paths, which is fatal for already-flat upstream
URDFs that ship relative or ``package://`` meshes (so100/so101/gr1/h1). For
those we copy the upstream text verbatim and apply joint-name renames with
``re.sub`` directly on the raw XML — preserving every mesh path byte-for-byte.
The renames target **joint names only** (``<joint name="X"`` and any
``joint="X"`` mimic/transmission references); link names are never touched. A
:class:`list` of ``(pattern, repl)`` pairs is applied in order, so so100/so101
take six numeric renames and gr1/h1 take one ``_joint``-suffix strip.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

_PROVENANCE = (
    "<!-- Vendored by `openral robot vendor-urdf {id}` from {src}. "
    "Upstream license applies — see docs/adr/0058. -->\n"
)

# SO-ARM numeric-joint → semantic HAL-name map, applied by joint NUMBER (the
# upstream so100/so101 URDFs name joints "1".."6"; the SO-ARM motor convention
# and the manifest use these semantic names). Both follower arms share the map.
_SO_ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",  # 1
    "shoulder_lift",  # 2
    "elbow_flex",  # 3
    "wrist_flex",  # 4
    "wrist_roll",  # 5
    "gripper",  # 6
)

# Per-robot raw-text joint renames (regex pattern, replacement), applied in
# order. Patterns are scoped to the joint-name context so no <link …> is hit:
#  * so100/so101: rewrite ``name="N"`` (only ever a <joint> in these URDFs;
#    links are semantic, transmissions are ``N_trans`` / ``motorN``).
#  * gr1/h1: strip the ``_joint`` suffix from every ``name="…_joint"`` (no link
#    ends in ``_joint`` in either URDF — verified).
#  * gr1 also collapses ``*_elbow_pitch`` → ``*_elbow`` to match the manifest's
#    HAL joint name (the upstream URDF spells the single-DoF elbow joint
#    ``*_elbow_pitch_joint``; the manifest/control contract calls it ``*_elbow``).
#    Applied AFTER the ``_joint`` strip; link-safe (no ``*_elbow_pitch`` link
#    exists — only the two ``*_elbow_pitch_joint`` joints — verified).
_RAW_RENAMES: dict[str, list[tuple[str, str]]] = {
    "so100_follower": [
        (rf'name="{n}"', f'name="{sem}"') for n, sem in enumerate(_SO_ARM_JOINT_NAMES, start=1)
    ],
    "so101_follower": [
        (rf'name="{n}"', f'name="{sem}"') for n, sem in enumerate(_SO_ARM_JOINT_NAMES, start=1)
    ],
    "gr1": [
        (r'name="([^"]*)_joint"', r'name="\1"'),
        (r'name="(left|right)_elbow_pitch"', r'name="\1_elbow"'),
    ],
    "h1": [(r'name="([^"]*)_joint"', r'name="\1"')],
}

# Joint-name normalization to the OpenRAL HAL convention. openarm: strip the
# "openarm_" prefix so joints become left_joint1..7 / right_joint1..7.
_RENAME: dict[str, tuple[str, str]] = {"openarm": (r'"openarm_', '"')}

# A ``(pattern, repl)`` rename is a 2-tuple; named to satisfy the magic-value lint.
_PAIR_LEN = 2


def _model_to_xml(model: object) -> str:
    """Serialize a loaded ``yourdfpy.URDF`` to a pretty-printed flat URDF string.

    yourdfpy ≥0.0.56 exposes ``write_xml_string()`` returning the serialized URDF
    as ASCII ``bytes`` on a single line (verified against the installed package —
    it wraps ``lxml.etree.tostring``; its ``**kwargs`` forwarding is broken so we
    cannot pass ``pretty_print`` through it). We re-parse and pretty-print via lxml
    so the committed file is human-reviewable + git-diffable.
    """
    from lxml import etree

    # ``write_xml_string`` is the public serializer in yourdfpy ≥0.0.56; one line.
    raw = model.write_xml_string()  # type: ignore[attr-defined] # reason: yourdfpy.URDF has no stubs
    root = etree.fromstring(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
    pretty: bytes = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="utf-8")
    return pretty.decode("utf-8")


def _load_model(upstream: str) -> object:
    """Load an upstream description into a ``yourdfpy.URDF``.

    ``rd:<module>`` resolves through ``robot_descriptions`` (xacro expanded via
    ``xacrodoc``); ``file:<path>`` loads an already-flat URDF directly.
    """
    if upstream.startswith("file:"):
        import yourdfpy

        path = Path(upstream[len("file:") :])
        # build_scene_graph/build_collision_scene_graph are off so an
        # upstream that references missing meshes still parses for re-export.
        return yourdfpy.URDF.load(
            str(path),
            build_scene_graph=False,
            build_collision_scene_graph=False,
            load_meshes=False,
            load_collision_meshes=False,
        )
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    module = upstream[len("rd:") :] if upstream.startswith("rd:") else upstream
    return load_robot_description(module)  # expands xacro via xacrodoc


def vendor_urdf(
    robot_id: str,
    *,
    upstream: str,
    out_dir: Path,
    rename: tuple[str, str] | Sequence[tuple[str, str]] | None = None,
    raw_text: bool = False,
) -> Path:
    """Expand ``upstream`` to a flat URDF at ``out_dir/<robot_id>.urdf``.

    Args:
        robot_id: OpenRAL robot id; names the output file and selects the
            default rename rule.
        upstream: ``rd:<robot_descriptions module>`` for xacro arms, or
            ``file:<path>`` for an already-flat upstream URDF (openarm).
        out_dir: Directory to write ``<robot_id>.urdf`` into (created if absent).
        rename: A ``(pattern, repl)`` pair, or a sequence of them, applied with
            ``re.sub`` in order. ``None`` selects the per-robot default
            (``_RAW_RENAMES[robot_id]`` in raw-text mode, else ``_RENAME``).
        raw_text: When ``True``, copy the already-flat upstream URDF text
            verbatim and apply ``rename`` directly to the raw XML — no yourdfpy
            round-trip — so ``package://`` / relative mesh paths are preserved
            byte-for-byte. Only valid for ``file:`` / ``rd:`` URDFs that are
            already flat (so100/so101/gr1/h1). Mutually exclusive with xacro
            expansion: the upstream must contain no ``${…}``.

    Returns:
        Path to the written URDF.

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> d = Path(tempfile.mkdtemp())
        >>> out = vendor_urdf("ur5e", upstream="rd:ur5e_description", out_dir=d)
        >>> out.name
        'ur5e.urdf'
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{robot_id}.urdf"
    if raw_text:
        xml = _read_raw_text(upstream)
        renames = _resolve_renames(robot_id, rename, default=_RAW_RENAMES.get(robot_id))
        for pat, repl in renames:
            xml = re.sub(pat, repl, xml)
        # Preserve the upstream's exact bytes apart from the renamed joint names
        # and the inserted provenance comment — including its newline style — so
        # the only diff vs upstream is the joint renames (gate: lowering must
        # stay byte-identical, mesh paths verbatim). newline="" disables write-side
        # newline translation (Path.write_text gained the kwarg only in 3.13).
        with out.open("w", encoding="utf-8", newline="") as fh:
            fh.write(_with_provenance_raw(xml, id=robot_id, src=upstream))
        return out
    model = _load_model(upstream)
    xml = _model_to_xml(model)
    default = [_RENAME[robot_id]] if robot_id in _RENAME else None
    renames = _resolve_renames(robot_id, rename, default=default)
    for pat, repl in renames:
        xml = re.sub(pat, repl, xml)
    out.write_text(_with_provenance(xml, id=robot_id, src=upstream))
    return out


def _resolve_renames(
    robot_id: str,
    rename: tuple[str, str] | Sequence[tuple[str, str]] | None,
    *,
    default: list[tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """Normalize the ``rename`` argument to a flat list of ``(pattern, repl)``.

    A bare ``(pattern, repl)`` pair (two strings) is wrapped into a one-element
    list; a sequence of pairs is passed through; ``None`` falls back to the
    per-robot ``default``.
    """
    if rename is None:
        return list(default) if default else []
    # A bare ``(pattern, repl)`` is a 2-tuple of strings; a sequence of such pairs
    # is anything else iterable. ``_PAIR_LEN`` names the 2 to satisfy PLR2004.
    if (
        isinstance(rename, tuple)
        and len(rename) == _PAIR_LEN
        and all(isinstance(x, str) for x in rename)
    ):
        return [cast("tuple[str, str]", rename)]
    return list(cast("Sequence[tuple[str, str]]", rename))


def _read_raw_text(upstream: str) -> str:
    """Read an already-flat upstream URDF's text verbatim (no XML round-trip).

    ``file:<path>`` reads that path; ``rd:<module>`` resolves the cached
    upstream ``.urdf`` via ``robot_descriptions`` and reads it directly. The
    upstream must already be flat — raising if any ``${…}`` xacro substitution
    survives, since raw-text mode does no expansion.
    """
    if upstream.startswith("file:"):
        path = Path(upstream[len("file:") :])
    elif upstream.startswith("rd:"):
        import importlib

        module = importlib.import_module(f"robot_descriptions.{upstream[len('rd:') :]}")
        path = Path(module.URDF_PATH)
    else:
        raise ValueError(f"raw_text mode needs a 'file:' or 'rd:' upstream, got {upstream!r}")
    # newline="" disables universal-newline translation so an upstream that uses
    # CRLF (e.g. gr1) is preserved verbatim rather than silently rewritten to LF.
    # (Path.read_text gained a newline kwarg only in 3.13; open() is the 3.12 path.)
    with path.open("r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    if "${" in text:
        raise ValueError(f"raw_text mode requires a flat URDF; {path} still has '${{…}}' xacro")
    return text


def _with_provenance_raw(xml: str, *, id: str, src: str) -> str:
    """Insert the provenance comment into a raw-text URDF, preserving its bytes.

    Unlike :func:`_with_provenance` (which rewrites the XML declaration emitted
    by the yourdfpy round-trip), this keeps the upstream document verbatim and
    only inserts the provenance comment on the line *after* the declaration (or
    at the top if there is none), matching the document's own newline style so
    the diff vs upstream is exactly the renamed joint names plus this comment.
    """
    nl = "\r\n" if "\r\n" in xml else "\n"
    header = _PROVENANCE.format(id=id, src=src).rstrip("\n")
    decl_match = re.match(r"^\s*<\?xml[^>]*\?>[ \t]*\r?\n?", xml)
    if decl_match:
        return xml[: decl_match.end()] + header + nl + xml[decl_match.end() :]
    return header + nl + xml


def _with_provenance(xml: str, *, id: str, src: str) -> str:
    """Prepend the provenance comment without breaking XML well-formedness.

    An XML declaration (``<?xml …?>``) must be byte 0 of the document — a
    comment may not precede it. When the serialized URDF leads with one, the
    provenance comment is inserted on the line *after* the declaration; otherwise
    it goes at the top.
    """
    header = _PROVENANCE.format(id=id, src=src)
    decl_match = re.match(r"^\s*<\?xml[^>]*\?>\s*\n?", xml)
    if decl_match:
        # The provenance comment carries a non-ASCII em-dash, so normalize the
        # declared encoding to UTF-8 (yourdfpy emits ``encoding='ASCII'``) to
        # keep the document well-formed.
        body = xml[decl_match.end() :]
        return '<?xml version="1.0" encoding="utf-8"?>\n' + header + body
    return header + xml
