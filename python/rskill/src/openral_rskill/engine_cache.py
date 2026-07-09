"""Filesystem-based per-host engine cache for compiled skill runtimes.

Compiled engine files (TensorRT ``.plan``, ONNX ``.onnx``, torchscript ``.pt``)
are keyed by a hash of ``(rskill_id, backend, QuantizationConfig)`` and stored
under ``~/.cache/openral/engines/``.  This avoids re-compiling on each
launch when the skill manifest and hardware have not changed.

Public surface
--------------
- ``EngineCache``: Cache get/put/invalidate/clear operations.
- ``DEFAULT_CACHE_DIR``: Platform default cache directory.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from openral_core.schemas import QuantizationConfig

DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "openral" / "engines"
"""Default filesystem path for compiled engine files.

Override by passing a different *cache_dir* to :class:`EngineCache`.
"""


class EngineCache:
    """Filesystem-backed cache for compiled skill engine files.

    Each entry is a plain file stored as ``<cache_dir>/<key>.engine``.
    Cache keys are 16-character SHA-256 truncations derived from the skill
    identifier, backend name, and serialised ``QuantizationConfig``.

    Args:
        cache_dir: Root directory for cache files.  Created on first use.

    Example:
        >>> import tempfile, pathlib
        >>> tmp = pathlib.Path(tempfile.mkdtemp())
        >>> cache = EngineCache(cache_dir=tmp)
        >>> key = cache.cache_key("openral/rskill-pick-cube", "pytorch", QuantizationConfig())
        >>> cache.get(key) is None
        True
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
        """Initialize the cache, creating *cache_dir* if it does not exist."""
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Key derivation ────────────────────────────────────────────────────────

    def cache_key(
        self,
        rskill_id: str,
        backend: str,
        config: QuantizationConfig,
    ) -> str:
        """Derive a stable cache key for the given skill + runtime + quant tuple.

        The key is a 16-character hex prefix of the SHA-256 hash of the JSON-
        serialised payload, sorted by key for determinism.

        Args:
            rskill_id: HuggingFace Hub repo ID or local path
                (e.g. ``"openral/rskill-pick-cube-so100"``).
            backend: Runtime backend name (e.g. ``"pytorch"``, ``"tensorrt"``).
            config: Quantization configuration.

        Returns:
            16-character lowercase hex string.

        Example:
            >>> import tempfile, pathlib
            >>> cache = EngineCache(pathlib.Path(tempfile.mkdtemp()))
            >>> k1 = cache.cache_key("my/skill", "onnx", QuantizationConfig())
            >>> k2 = cache.cache_key("my/skill", "onnx", QuantizationConfig())
            >>> k1 == k2
            True
        """
        payload = json.dumps(
            {
                "rskill_id": rskill_id,
                "backend": backend,
                "config": config.model_dump(mode="json"),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ── Cache operations ──────────────────────────────────────────────────────

    def get(self, key: str) -> Path | None:
        """Return the cached engine path for *key*, or ``None`` on a miss.

        Args:
            key: 16-character cache key from :meth:`cache_key`.

        Returns:
            Path to the cached file if it exists, else ``None``.

        Example:
            >>> import tempfile, pathlib
            >>> cache = EngineCache(pathlib.Path(tempfile.mkdtemp()))
            >>> cache.get("nonexistent") is None
            True
        """
        p = self._key_path(key)
        return p if p.exists() else None

    def put(self, key: str, engine_path: Path) -> Path:
        """Copy *engine_path* into the cache under *key*.

        Args:
            key: 16-character cache key from :meth:`cache_key`.
            engine_path: Source file to cache.  Must exist.

        Returns:
            Path of the newly cached file.

        Raises:
            FileNotFoundError: If *engine_path* does not exist.
        """
        if not engine_path.exists():
            raise FileNotFoundError(f"EngineCache.put: source file not found: '{engine_path}'")
        dest = self._key_path(key)
        shutil.copy2(engine_path, dest)
        return dest

    def invalidate(self, key: str) -> None:
        """Remove the cached entry for *key* (no-op on a miss).

        Args:
            key: 16-character cache key from :meth:`cache_key`.
        """
        p = self._key_path(key)
        if p.exists():
            p.unlink()

    def clear(self) -> None:
        """Remove all engine files from the cache directory."""
        for p in self._dir.glob("*.engine"):
            p.unlink()

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def size_bytes(self) -> int:
        """Total size in bytes of all cached engine files.

        Example:
            >>> import tempfile, pathlib
            >>> cache = EngineCache(pathlib.Path(tempfile.mkdtemp()))
            >>> cache.size_bytes
            0
        """
        return sum(p.stat().st_size for p in self._dir.glob("*.engine"))

    @property
    def entry_count(self) -> int:
        """Number of engine files currently in the cache.

        Example:
            >>> import tempfile, pathlib
            >>> cache = EngineCache(pathlib.Path(tempfile.mkdtemp()))
            >>> cache.entry_count
            0
        """
        return sum(1 for _ in self._dir.glob("*.engine"))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _key_path(self, key: str) -> Path:
        return self._dir / f"{key}.engine"
