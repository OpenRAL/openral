"""The one gate for ``"module.path:attribute"`` code references named in config.

A robot manifest's ``hal.sim`` / ``hal.real`` and a scene's ``composition.composer``
name a Python object that OpenRAL imports and calls with config-supplied keyword
arguments. Unchecked, a crafted manifest or scene (or a DDS peer that could rewrite the
HAL's ``scene_composition_json``) runs any importable callable — ``os:system`` included.

``check_code_ref`` admits only objects in OpenRAL's own packages: a top-level module
named ``openral_*`` (the open ``openral_hal`` / ``openral_sim`` and OpenRAL Pro's
``openral_pro_*``). ``OPENRAL_ALLOW_EXTERNAL_PLUGINS=1`` admits any other module, with
a warning on every use. The check runs both when the schema loads and right before the
import (``import_code_ref``), so neither path can skip it.
"""

from __future__ import annotations

import importlib
import os
import re
from typing import Final

import structlog

from openral_core.exceptions import ROSConfigError

__all__ = [
    "ALLOW_EXTERNAL_PLUGINS_ENV",
    "check_code_ref",
    "import_code_ref",
]

ALLOW_EXTERNAL_PLUGINS_ENV: Final[str] = "OPENRAL_ALLOW_EXTERNAL_PLUGINS"

_REF = re.compile(r"^(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):(?P<attr>[A-Za-z_]\w*)$")
_TRUSTED_TOP_LEVEL = re.compile(r"^openral_\w+$")

_log = structlog.get_logger(__name__)


def check_code_ref(ref: str, *, setting: str) -> str:
    """Refuse a code reference outside OpenRAL's own packages.

    Args:
        ref: ``"module.path:attribute"``.
        setting: Where the reference came from (``"hal.real"``,
            ``"composition.composer"``), named in the error.

    Returns:
        ``ref`` unchanged, so a pydantic validator can return it.

    Raises:
        ROSConfigError: ``ref`` is malformed, or its top-level module is not
            ``openral_*`` and ``OPENRAL_ALLOW_EXTERNAL_PLUGINS`` is not ``1``.

    Example:
        >>> check_code_ref("openral_hal.ur_real:UR5eRealHAL", setting="hal.real")
        'openral_hal.ur_real:UR5eRealHAL'
        >>> try:
        ...     check_code_ref("os:system", setting="hal.real")
        ... except ROSConfigError as exc:
        ...     print(str(exc).split(";")[0])
        hal.real = 'os:system' names code outside OpenRAL's packages (openral_*)
    """
    match = _REF.match(ref)
    if match is None:
        raise ROSConfigError(f"{setting} = {ref!r} is malformed; expected 'module.path:attribute'.")
    top_level = match["module"].split(".", 1)[0]
    if _TRUSTED_TOP_LEVEL.match(top_level):
        return ref
    if os.environ.get(ALLOW_EXTERNAL_PLUGINS_ENV) == "1":
        _log.warning(
            "code_ref.external_plugin_allowed",
            setting=setting,
            ref=ref,
            env=ALLOW_EXTERNAL_PLUGINS_ENV,
        )
        return ref
    raise ROSConfigError(
        f"{setting} = {ref!r} names code outside OpenRAL's packages (openral_*); "
        f"set {ALLOW_EXTERNAL_PLUGINS_ENV}=1 to load it from a trusted source."
    )


def import_code_ref(ref: str, *, setting: str) -> object:
    """``check_code_ref``, then import the module and return the attribute.

    Raises:
        ROSConfigError: The reference is refused, its module is not importable,
            or the attribute is absent.

    Example:
        >>> import_code_ref("openral_core.code_refs:check_code_ref", setting="x").__name__
        'check_code_ref'
    """
    check_code_ref(ref, setting=setting)
    module_path, _, attr = ref.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ROSConfigError(
            f"{setting} = {ref!r}: module {module_path!r} is not importable ({exc})."
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ROSConfigError(
            f"{setting} = {ref!r}: module {module_path!r} has no attribute {attr!r}."
        ) from exc
