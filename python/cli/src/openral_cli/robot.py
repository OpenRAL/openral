"""``openral robot vendor-urdf <id>`` — expand an upstream xacro to a flat URDF.

The flat, committed URDF means end users need no xacro tooling at runtime
(ADR-0057). The xacro-only arms (ur5e/ur10e/rizon4) ship a ``XACRO_PATH`` in
``robot_descriptions``; we let ``robot_descriptions``' ``yourdfpy`` loader run
``xacrodoc`` to expand every ``${…}`` substitution, then serialize the resulting
flat URDF with a provenance header. ``openarm`` ships only MJCF upstream, so its
flattened URDF is cloned separately and passed as a ``file:`` upstream; the
``--rename`` hook strips the ``openarm_`` joint/link prefix to the OpenRAL HAL
convention (``left_joint1..7`` / ``right_joint1..7``).
"""

from __future__ import annotations

import re
from pathlib import Path

_PROVENANCE = (
    "<!-- Vendored by `openral robot vendor-urdf {id}` from {src}. "
    "Upstream license applies — see docs/adr/0057. -->\n"
)

# Joint-name normalization to the OpenRAL HAL convention. openarm: strip the
# "openarm_" prefix so joints become left_joint1..7 / right_joint1..7.
_RENAME: dict[str, tuple[str, str]] = {"openarm": (r'"openarm_', '"')}


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
    rename: tuple[str, str] | None = None,
) -> Path:
    """Expand ``upstream`` to a flat URDF at ``out_dir/<robot_id>.urdf``.

    Args:
        robot_id: OpenRAL robot id; names the output file and selects the
            default rename rule.
        upstream: ``rd:<robot_descriptions module>`` for xacro arms, or
            ``file:<path>`` for an already-flat upstream URDF (openarm).
        out_dir: Directory to write ``<robot_id>.urdf`` into (created if absent).
        rename: Optional ``(pattern, repl)`` applied with ``re.sub`` to the
            serialized URDF. Defaults to the per-robot rule in ``_RENAME``.

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
    model = _load_model(upstream)
    xml = _model_to_xml(model)
    rename = rename if rename is not None else _RENAME.get(robot_id)
    if rename:
        xml = re.sub(rename[0], rename[1], xml)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{robot_id}.urdf"
    out.write_text(_with_provenance(xml, id=robot_id, src=upstream))
    return out


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
