"""Open-vocabulary text embedder for spatial memory (ADR-0038 §5, Phase 4).

Provides a small :class:`TextEmbedder` Protocol and an :class:`OpenClipEmbedder`
backed by **OpenCLIP ViT-B/32** (MIT code + weights). :class:`SpatialMemory`
takes an optional embedder so a free-text query ("the red wine") matches object
nodes by CLIP cosine similarity, not just exact/substring labels — handling
synonyms and paraphrases that label matching misses.

This is **optional and compute-gated** (ADR-0038 §5): with no embedder the memory
works on label + pose + recency. The embedder embeds *per object label* (and the
query) — single-digit-ms per text on a robot GPU, negligible vs the dense
feature fields the ADR deferred. Install with ``uv sync --group clip``.

Embeddings are L2-normalized, so cosine similarity is a plain dot product.

Example:
    >>> import numpy as np
    >>> emb = OpenClipEmbedder()  # doctest: +SKIP
    >>> v = emb.embed_text(["bottle of wine"])  # doctest: +SKIP
    >>> float(np.linalg.norm(v[0]))  # doctest: +SKIP
    1.0
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError

DEFAULT_CLIP_MODEL = "ViT-B-32-quickgelu"
"""OpenCLIP model name (ADR-0038 §5 default) — the quickgelu variant matches the
``openai`` pretrained weights' activation (avoids the QuickGELU mismatch warning)."""

DEFAULT_CLIP_PRETRAINED = "openai"
"""OpenCLIP pretrained tag — the original CLIP weights (MIT)."""


@runtime_checkable
class TextEmbedder(Protocol):
    """A text → unit-vector embedder for open-vocabulary matching."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    def embed_text(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed ``texts`` into an ``(N, dim)`` L2-normalized float32 array."""
        ...


class OpenClipEmbedder:
    """OpenCLIP text embedder (ViT-B/32, MIT) for open-vocab matching (ADR-0038 §5).

    Args:
        model_name: OpenCLIP architecture (default ``"ViT-B-32"``).
        pretrained: OpenCLIP pretrained tag (default ``"openai"`` — MIT weights).
        device: Torch device; defaults to ``"cuda"`` when available else ``"cpu"``.

    Raises:
        ROSConfigError: When ``open_clip`` / ``torch`` are not installed
            (``uv sync --group clip``) or the model cannot be created.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_CLIP_MODEL,
        pretrained: str = DEFAULT_CLIP_PRETRAINED,
        device: str | None = None,
    ) -> None:
        """Create the embedder, loading the OpenCLIP model onto ``device``."""
        try:
            import open_clip  # noqa: PLC0415  # reason: optional dep; mypy via mypy.ini ignore_missing_imports
            import torch  # noqa: PLC0415  # reason: optional/compute-gated import (clip group)
        except ImportError as exc:  # pragma: no cover - exercised via the clip group
            raise ROSConfigError(
                "OpenClipEmbedder requires open-clip-torch + torch (`uv sync --group clip`)."
            ) from exc

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self._device
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:  # reason: open_clip raises bare exceptions on bad model/download
            raise ROSConfigError(
                f"OpenClipEmbedder failed to load {model_name!r}/{pretrained!r}: {exc}"
            ) from exc
        model.eval()
        self._model = model
        self._dim = int(model.text_projection.shape[-1])
        self.model_name = model_name
        self.pretrained = pretrained

    @property
    def dim(self) -> int:
        """Embedding dimensionality (512 for ViT-B/32)."""
        return self._dim

    def embed_text(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed ``texts`` into an ``(N, dim)`` L2-normalized float32 array."""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        torch = self._torch
        tokens = self._tokenizer(list(texts)).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)  # type: ignore[no-any-return]  # reason: torch→numpy is untyped
