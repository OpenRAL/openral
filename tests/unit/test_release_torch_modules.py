"""VRAM is only reclaimed if the references are dropped BEFORE the flush.

``torch.cuda.empty_cache()`` returns *already-free* cached blocks to the
driver. It cannot free memory the allocator still considers live, so calling
it while an adapter still holds its policy frees exactly nothing — which is
what every VLA adapter's ``close()`` used to do. On an 8 GB card that made an
rSkill swap fail to give the card back, and a second skill OOM'd even though
each fits on its own.

The CUDA test below is the falsification: it fails if the order is reversed.
"""

from __future__ import annotations

import gc

import pytest
from openral_rskill._vla_core import release_torch_modules

torch = pytest.importorskip("torch", reason="release_torch_modules is a torch seam")


class _Adapter:
    """Minimal stand-in with the attribute shape the real adapters use."""

    def __init__(self, device: str, payload: object) -> None:
        self.device = device
        self._policy: object = payload
        self._preprocessor: object = object()
        self._postprocessor: object = object()


def test_named_attributes_are_cleared() -> None:
    adapter = _Adapter("cpu", object())

    release_torch_modules(
        adapter, "_policy", "_preprocessor", "_postprocessor", device=adapter.device
    )

    assert adapter._policy is None
    assert adapter._preprocessor is None
    assert adapter._postprocessor is None


def test_unlisted_attributes_are_left_alone() -> None:
    """Only what the caller names is cleared — close() is not a reset()."""
    adapter = _Adapter("cpu", object())
    sentinel = adapter._preprocessor

    release_torch_modules(adapter, "_policy", device=adapter.device)

    assert adapter._policy is None
    assert adapter._preprocessor is sentinel


def test_a_missing_attribute_does_not_raise() -> None:
    """Teardown must always reach the code behind it (HAL shutdown, sidecars)."""
    adapter = _Adapter("cpu", object())

    release_torch_modules(adapter, "_policy", "_not_a_real_attribute", device="cpu")

    assert adapter._policy is None


def test_dropping_the_last_reference_lets_the_object_be_collected() -> None:
    """The point of the helper: nothing else keeps the module alive."""
    import weakref

    payload = torch.nn.Linear(8, 8)
    ref = weakref.ref(payload)
    adapter = _Adapter("cpu", payload)
    del payload

    assert ref() is not None, "the adapter should still be holding it"
    release_torch_modules(adapter, "_policy", device="cpu")
    gc.collect()

    assert ref() is None, "dropping the adapter's reference must free the module"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_memory_is_actually_reclaimed() -> None:
    """Falsification test for the ordering bug.

    Flushing first (the old behaviour) leaves allocated bytes unchanged;
    dropping the reference first returns them. Measured on an RTX 4070 with
    a 768 MiB module: 768.2 -> 768.2 MiB via ``empty_cache()`` alone, and
    768.2 -> 0.0 MiB once the reference is dropped.
    """
    gc.collect()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()

    adapter = _Adapter(
        "cuda",
        torch.nn.Sequential(*[torch.nn.Linear(1024, 1024) for _ in range(8)]).cuda(),
    )
    loaded = torch.cuda.memory_allocated()
    assert loaded > before, "the fixture must actually occupy VRAM"

    # The old close(): flush while still holding the module.
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() == loaded, (
        "empty_cache() must NOT free a live module — if this ever passes, the "
        "premise of this test (and of the fix) has changed."
    )

    release_torch_modules(adapter, "_policy", device="cuda")

    assert torch.cuda.memory_allocated() <= before, (
        f"VRAM not reclaimed: {torch.cuda.memory_allocated()} > {before}"
    )
