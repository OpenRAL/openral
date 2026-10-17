"""Compatibility alias: the object-primitive helpers now live in ``openral_core``.

Moved to ``openral_core.object_primitives`` (issue #418) so Layer 2 can import them;
import from there. This module only re-exports their public names.
"""

from openral_core.object_primitives import *  # noqa: F403
