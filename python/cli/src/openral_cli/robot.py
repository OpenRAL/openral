"""``openral robot vendor-urdf <id>`` — expand an upstream xacro to a flat URDF.

Xacro-only arms (ur5e/ur10e/rizon4) ship a ``XACRO_PATH`` in
``robot_descriptions``; its ``yourdfpy`` loader runs ``xacrodoc`` to expand
every ``${…}``, then the result is serialized with a provenance header.
``openarm`` ships only MJCF upstream, so its flattened URDF is cloned
separately and passed as a ``file:`` upstream. Its names are vendored **as
upstream spells them** (``openarm_left_joint1``): the vendored URDF has to agree
with the arm's own ros2_control graph and ``/joint_states``, which no rename
here can change. See ``_RENAME``.

**Portable mesh refs.** The yourdfpy round-trip absolutizes every mesh
``filename`` into the vendoring machine's ``robot_descriptions`` cache (once
broke collision lowering fleet-wide in CI — empty ACMs). For ``rd:``
upstreams, ``_portable_mesh_refs`` rewrites cache-absolute paths to
``rd:<module>:<path-relative-to-repository>``, expanded at load time by
``openral_safety.urdf_lowering`` through the same pinned clone.

**Raw-text mode** (``raw_text=True``). The yourdfpy round-trip also mangles
``package://`` mesh paths, fatal for already-flat upstream URDFs
(so100/so101/gr1/h1). Those are copied verbatim with joint-name-only renames
(``<joint name="X"`` / mimic/transmission ``joint="X"`` refs; links untouched)
applied via ``re.sub`` on the raw XML. A ``list`` of ``(pattern, repl)``
pairs is applied in order: so100/so101 take six numeric renames, gr1/h1 take
a ``_joint``-suffix strip.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

_PROVENANCE = (
    "<!-- Vendored by `openral robot vendor-urdf {id}` from {src}. Upstream license applies. -->\n"
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
# order, scoped to the joint-name context so no <link …> is hit:
#  * so100/so101: rewrite ``name="N"`` (only <joint> in these URDFs; links
#    are semantic, transmissions are ``N_trans``/``motorN``).
#  * gr1/h1: strip ``_joint`` suffix from ``name="…_joint"`` (no link ends
#    in ``_joint`` in either URDF — verified).
#  * gr1 also collapses ``*_elbow_pitch`` → ``*_elbow`` (manifest/control
#    contract name; upstream spells it ``*_elbow_pitch_joint``), applied
#    AFTER the ``_joint`` strip — link-safe, verified.
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

# Per-robot default rename applied to a `rd:`/`file:` upstream. Empty today.
#
# `openarm` used to strip the `openarm_` prefix here so joints read
# `left_joint1..7`, matching the manifest's logical joint names. That made the
# vendored URDF the only artefact in the repo using that spelling: the arm's
# ros2_control graph, its `/joint_states`, its MJCF and `OpenArmRealHAL.
# ros2_control_joint_names()` all say `openarm_left_joint1`. Two conventions on
# one `/robot_description` topic empty out `joint_state_broadcaster`, whose
# `use_urdf_to_filter` publishes only joints the URDF also names — and a
# `robot_state_publisher` fed the stripped URDF never matches a name on the real
# arm's `/joint_states`, so its whole tree stays at the rest pose in `/tf`.
# The manifest's logical names stay as they are: they are the action-vector
# contract, shared with other robots and pinned by published rSkill manifests,
# and `sim_joint_name` / `ros2_control_joint_names()` are the declared bridges.
_RENAME: dict[str, tuple[str, str]] = {}

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
        >>> # vendor-urdf needs the optional `lowering` group (yourdfpy/xacrodoc):
        >>> out = vendor_urdf(  # doctest: +SKIP
        ...     "ur5e", upstream="rd:ur5e_description", out_dir=d
        ... )
        >>> out.name  # doctest: +SKIP
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
    if upstream.startswith("rd:"):
        xml = _portable_mesh_refs(xml, module=upstream[len("rd:") :])
    default = [_RENAME[robot_id]] if robot_id in _RENAME else None
    renames = _resolve_renames(robot_id, rename, default=default)
    for pat, repl in renames:
        xml = re.sub(pat, repl, xml)
    out.write_text(_with_provenance(xml, id=robot_id, src=upstream))
    return out


def _portable_mesh_refs(xml: str, *, module: str) -> str:
    """Rewrite cache-absolutized mesh paths to portable ``rd:<module>:<relpath>`` refs.

    The yourdfpy round-trip resolves every mesh ``filename`` to an absolute path
    inside the vendoring machine's ``robot_descriptions`` clone
    (``~/.cache/robot_descriptions/<repo>/…``) — paths that exist nowhere else, so
    committing them breaks collision lowering on every other host (CI's re-lower
    then finds no meshes and emits an empty ACM). Rewrite them to
    ``rd:<module>:<path-relative-to-repository>``, which
    ``openral_safety.urdf_lowering`` expands at load time through the same pinned
    ``robot_descriptions`` clone on any machine.
    """
    import importlib

    mod = importlib.import_module(f"robot_descriptions.{module}")
    repo = str(Path(mod.REPOSITORY_PATH))
    return xml.replace(f'filename="{repo}/', f'filename="rd:{module}:')


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

    Unlike ``_with_provenance`` (which rewrites the XML declaration emitted
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


def _write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as mode 0600 (parent 0700), never through a symlink."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)  # an existing file keeps its old mode through O_CREAT
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def _secret_path() -> Path:
    from openral_core import enrolment_path

    return enrolment_path().with_name("robot.secret")


def _settings_cache() -> Path:
    from openral_core import enrolment_path

    return enrolment_path().with_name("settings.json")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: the request carries the robot's credential."""

    def redirect_request(self, *_: object, **__: object) -> urllib.request.Request | None:
        return None


_open = urllib.request.build_opener(_NoRedirect).open


ACCOUNT_SETTINGS = frozenset(
    {
        "OPENRAL_REASONER_API_KEY",
        "OPENRAL_REASONER_MODEL",
        "OPENRAL_REASONER_ENDPOINT",
        "OPENRAL_REASONER_DIALECT",
    }
)
"""What openral.com may set on a robot. Never the dashboard's write- or remote-control
switches: those stay on the robot (ADR-0129, Safety-WG), whatever the site sends."""


def apply_console_settings(*, echo: Callable[[str], None]) -> list[str]:
    """Put the reasoner settings set in this robot's console into ``os.environ``.

    The console's settings pane saves the provider keys and the model on the robot
    (``openral_observability.dashboard.robot_settings``); a deploy maps them to the
    reasoner's variables (``reasoner_env``). A variable already in the environment wins;
    :func:`apply_reasoner_settings` skips the account's settings altogether when this
    sets anything, so the robot's own choice wins over its openral.com account's and the
    robot's key is never paired with an account's endpoint. Returns the names it set;
    never raises.
    """
    from openral_observability.dashboard.robot_settings import load_settings, reasoner_env

    applied = {k: v for k, v in reasoner_env(load_settings()).items() if k not in os.environ}
    os.environ.update(applied)
    if applied:
        echo(f"console settings: {', '.join(sorted(applied))}")
    return sorted(applied)


def apply_reasoner_settings(*, echo: Callable[[str], None]) -> list[str]:
    """The reasoner settings a deploy starts with: the robot console's, else the account's.

    When the console's settings choose the reasoner (a model, and its key) the account's
    reasoner settings are not applied at all: an account's endpoint or dialect would
    otherwise carry the robot's own key to a URL the site picked. A variable already in
    the environment wins over both. Returns the names it set; never raises.
    """
    return apply_console_settings(echo=echo) or apply_account_settings(echo=echo)


def apply_account_settings(*, echo: Callable[[str], None]) -> list[str]:
    """Put this robot's openral.com account settings into ``os.environ`` (ADR-0129).

    A linked robot (``openral robot link``) fetches its org's (or owner's) settings,
    e.g. ``OPENRAL_REASONER_API_KEY``, with its own credential from
    ``<site>/api/robot/settings``, and caches them (0600) for offline starts. A variable
    already in the environment wins; only ``ACCOUNT_SETTINGS`` are applied. A refused
    credential (the robot was removed, or given a new link code) deletes the cache.
    Returns the names it set; never raises.
    """
    from urllib.parse import urlsplit

    from openral_core import enrolment_path
    from openral_observability.dashboard.remote_auth import load_enrolment

    secret_file = _secret_path()
    try:
        enrolment = load_enrolment(enrolment_path())
    except ValueError as exc:
        echo(f"account settings: {exc}")
        return []
    if enrolment is None or not secret_file.is_file():
        return []
    site = "{0.scheme}://{0.netloc}".format(urlsplit(enrolment.iss))
    request = urllib.request.Request(
        f"{site}/api/robot/settings",
        headers={
            "Authorization": f"Bearer {secret_file.read_text(encoding='utf-8').strip()}",
            # Cloudflare refuses urllib's default "Python-urllib/x.y" (error 1010).
            "User-Agent": "openral-robot",
        },
    )
    try:
        # The enrolled site's fixed https URL, not user input; never a redirect, which
        # would carry the robot's credential to another host.
        with _open(request, timeout=10) as resp:
            settings = json.loads(resp.read())["settings"]
        _write_private(_settings_cache(), json.dumps(settings) + "\n")
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 403):
            return _cached_settings(site, exc, echo)
        # Revoked: no cached copy outlives the credential.
        _settings_cache().unlink(missing_ok=True)
        echo(f"account settings: {site} refused this robot's credential; run a new link code")
        return []
    except (OSError, ValueError, KeyError) as exc:
        return _cached_settings(site, exc, echo)
    return _apply(settings, site, echo)


def _cached_settings(site: str, exc: Exception, echo: Callable[[str], None]) -> list[str]:
    """The cached settings while ``site`` is unreachable (``exc``), else none."""
    try:
        cached = json.loads(_settings_cache().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        echo(f"account settings: {site} unreachable ({exc}); none cached")
        return []
    echo(f"account settings: {site} unreachable ({exc}); using the cached ones")
    return _apply(cached if isinstance(cached, dict) else {}, site, echo)


def _apply(settings: dict[str, object], site: str, echo: Callable[[str], None]) -> list[str]:
    applied = [
        k
        for k, v in settings.items()
        if k in ACCOUNT_SETTINGS and isinstance(v, str) and k not in os.environ
    ]
    for name in applied:
        os.environ[name] = str(settings[name])
    if applied:
        echo(f"account settings from {site}: {', '.join(sorted(applied))}")
    return applied


def link_robot(code: str, *, echo: Callable[[str], None], service: bool = True) -> Path:
    """``openral robot link <code>``: enrol this robot with openral.com (ADR-0129 Interface B).

    Writes everything in the link code except the tunnel token to the enrolment file
    (``openral_core.enrolment_path()``, mode 0600), which ``openral dashboard`` reads
    to open its remote listener; stores the tunnel token beside it (``tunnel.env``,
    mode 0600); fetches the pinned ``cloudflared``; and, with ``service``, installs it as
    the user service ``openral-tunnel.service`` (enabled at boot, no root) so the
    tunnel survives reboots. Without the service ``openral dashboard`` runs the tunnel
    itself while it is up.

    Returns:
        The enrolment file written.

    Raises:
        ROSConfigError: ``code`` is not a valid link code.
    """
    from openral_core import enrolment_path, parse_robot_link_code
    from openral_observability.dashboard.tunnel import (
        SERVICE,
        ensure_cloudflared,
        install_service,
        write_tunnel_token,
    )

    enrolment, tunnel, secret = parse_robot_link_code(code)
    path = enrolment_path()
    _write_private(path, enrolment.model_dump_json(indent=2) + "\n")
    write_tunnel_token(tunnel)
    # Nothing of a previously linked robot survives: its credential would fetch its
    # org's settings, and its cache would serve them.
    _settings_cache().unlink(missing_ok=True)
    _secret_path().unlink(missing_ok=True)
    if secret is not None:
        _write_private(_secret_path(), secret + "\n")
    echo(f"Linked robot {enrolment.robot} ({enrolment.aud}); enrolment in {path}")
    exe = ensure_cloudflared()
    if exe is None:
        echo("Couldn't fetch cloudflared for this machine; the dashboard retries when it starts.")
    elif service:
        problem = install_service(exe)
        if problem is None:
            echo(f"Tunnel installed as the user service {SERVICE}: up now and after every reboot.")
        else:
            echo(f"Tunnel service not installed ({problem}); `openral dashboard` runs it instead.")
    echo(
        "Start `openral dashboard` (or your deploy): the robot then shows on openral.com/dashboard."
    )
    return path
