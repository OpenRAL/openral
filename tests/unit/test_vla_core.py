"""Unit tests for ``openral_rskill._vla_core``.

The shared core owns the three seams every VLA adapter goes through:
``resolve_device``, ``resolve_rskill_repo_id``, and ``run_inference``
(plus ``to_numpy_action``). These tests exercise them with real
``VLASpec`` instances, real torch tensors, and a real OTel
``InMemorySpanExporter`` — no MagicMocks for OpenRAL types.

The lerobot policy itself is not an OpenRAL type; like
``tests/unit/test_smolvla_adapter.py`` we stand in a tiny class that
exposes ``select_action(batch) -> Tensor``. That's fixture, not mock.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from openral_core.exceptions import ROSConfigError
from openral_core.schemas import VLASpec
from openral_observability import semconv
from openral_rskill._vla_core import (
    resolve_device,
    resolve_inference_engine,
    resolve_rskill_repo_id,
    resolve_rskill_repo_revision,
    run_inference,
    to_numpy_action,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class _FakePolicy:
    """Fixed-output policy stub — same shape lerobot policies expose."""

    def __init__(self, action_dim: int = 7) -> None:
        self._action_dim = action_dim
        self.calls = 0

    def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
        self.calls += 1
        return torch.arange(self._action_dim, dtype=torch.float32).unsqueeze(0)


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    """Install an in-memory exporter on a fresh TracerProvider for one test.

    The observability tracing helpers go through ``trace.get_tracer`` so
    overriding the global TracerProvider is enough to capture spans.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    yield exporter
    provider.shutdown()


# ── resolve_device ────────────────────────────────────────────────────────────


class TestResolveDevice:
    def test_explicit_cpu(self) -> None:
        spec = VLASpec(id="smolvla", weights_uri="x", device="cpu")
        assert resolve_device(spec) == "cpu"

    def test_explicit_cuda_passes_through(self) -> None:
        spec = VLASpec(id="smolvla", weights_uri="x", device="cuda:1")
        assert resolve_device(spec) == "cuda:1"

    def test_auto_resolves_against_real_torch(self) -> None:
        spec = VLASpec(id="smolvla", weights_uri="x", device="auto")
        resolved = resolve_device(spec)
        if torch.cuda.is_available():
            assert resolved == "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            assert resolved == "mps"
        else:
            assert resolved == "cpu"


# ── resolve_rskill_repo_id ────────────────────────────────────────────────────


class TestResolveRskillRepoId:
    def test_rejects_hf_uri(self) -> None:
        with pytest.raises(ROSConfigError, match="hf://"):
            resolve_rskill_repo_id("hf://lerobot/smolvla_base", adapter_name="SmolVLA")

    def test_rejects_local_scheme(self) -> None:
        with pytest.raises(ROSConfigError, match="local://"):
            resolve_rskill_repo_id("local://rskills/smolvla-libero", adapter_name="xVLA")

    def test_resolves_real_rskill_manifest(self) -> None:
        """Resolves the SmolVLA-LIBERO rSkill manifest committed under skills/."""
        repo_id = resolve_rskill_repo_id("rskills/smolvla-libero", adapter_name="SmolVLA")
        assert "/" in repo_id  # bare HF Hub repo id like "owner/name"
        assert not repo_id.startswith("hf://")
        assert not repo_id.startswith("rskill://")


# ── resolve_rskill_repo_revision (H4) ─────────────────────────────────────────


class TestResolveRskillRepoRevision:
    def test_rejects_hf_scheme(self) -> None:
        with pytest.raises(ROSConfigError, match="hf://"):
            resolve_rskill_repo_revision("hf://lerobot/smolvla_base", adapter_name="SmolVLA")

    def test_real_unpinned_fixture_returns_none_revision(self) -> None:
        """The committed SmolVLA-LIBERO rSkill is unpinned → revision is None."""
        repo_id, revision = resolve_rskill_repo_revision(
            "rskills/smolvla-libero", adapter_name="SmolVLA"
        )
        assert "/" in repo_id and not repo_id.startswith(("hf://", "rskill://"))
        assert revision is None  # fixture's weights_uri carries no @<sha>


# ── run_inference ─────────────────────────────────────────────────────────────


class TestRunInference:
    def test_returns_policy_tensor(self) -> None:
        policy = _FakePolicy(action_dim=7)
        out = run_inference(policy, batch={})
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 7)
        assert policy.calls == 1

    def test_emits_inference_span_with_attributes(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        policy = _FakePolicy()
        run_inference(policy, batch={}, chunk_index=3, kind="prefetch", chunk_size=10)
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        attrs = dict(span.attributes or {})
        assert attrs.get("inference.kind") == "prefetch"
        assert attrs.get("inference.chunk_index") == 3
        assert attrs.get("inference.chunk_size") == 10
        assert attrs.get(semconv.INFERENCE_DURATION_MS, -1.0) >= 0.0

    def test_default_kind_is_single(self, span_exporter: InMemorySpanExporter) -> None:
        run_inference(_FakePolicy(), batch={})
        (span,) = span_exporter.get_finished_spans()
        assert dict(span.attributes or {}).get("inference.kind") == "single"

    def test_runs_with_no_grad(self) -> None:
        """``run_inference`` must disable autograd so policies can't leak grads."""

        class _GradAssertingPolicy:
            def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
                assert not torch.is_grad_enabled()
                return torch.zeros(1, 7)

        run_inference(_GradAssertingPolicy(), batch={})

    def test_named_chunk_method_synchronizes_before_return(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A prefetch-ready result means CUDA work completed, not merely launched."""
        synchronized: list[object] = []

        class _CudaLikeResult:
            is_cuda = True
            device = "cuda:0"

        class _ChunkPolicy:
            def predict_action_chunk(self, batch: dict[str, Any]) -> _CudaLikeResult:
                return _CudaLikeResult()

        monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
        policy = _ChunkPolicy()
        out = run_inference(
            policy,
            batch={},
            call=policy.predict_action_chunk,
            synchronize=True,
        )

        assert isinstance(out, _CudaLikeResult)
        assert synchronized == ["cuda:0"]

    def test_engine_and_device_attrs_emitted(self, span_exporter: InMemorySpanExporter) -> None:
        """``inference.engine`` defaults to ``"torch"``; ``device`` is lifted from policy."""

        class _PolicyWithDevice:
            device = torch.device("cpu")

            def select_action(self, batch: dict[str, Any]) -> torch.Tensor:
                return torch.zeros(1, 4)

        run_inference(_PolicyWithDevice(), batch={})
        (span,) = span_exporter.get_finished_spans()
        attrs = dict(span.attributes or {})
        assert attrs.get("inference.engine") == "torch"
        assert attrs.get("inference.device") == "cpu"

    def test_engine_override_passes_through(self, span_exporter: InMemorySpanExporter) -> None:
        """TRT / ONNX adapters can pass their own engine label."""
        run_inference(_FakePolicy(), batch={}, engine="trt")
        (span,) = span_exporter.get_finished_spans()
        assert dict(span.attributes or {}).get("inference.engine") == "trt"

    def test_released_pro_trt_attachment_overrides_manifest_runtime(
        self, span_exporter: InMemorySpanExporter
    ) -> None:
        """A runtime-attached TRT sampler wins over declared PyTorch."""

        class _TrtSampler:
            __module__ = "openral_pro_trt.smolvla_trt"

        class _Model:
            sample_actions = _TrtSampler()

        class _Policy(_FakePolicy):
            model = _Model()

        policy = _Policy()
        assert resolve_inference_engine(policy, "pytorch") == "trt"
        run_inference(policy, batch={})
        (span,) = span_exporter.get_finished_spans()
        assert dict(span.attributes or {}).get("inference.engine") == "trt"

    def test_explicit_plugin_marker_is_generic(self) -> None:
        class _Policy:
            _openral_inference_engine = "tensorrt"

        assert resolve_inference_engine(_Policy(), "pytorch") == "trt"


# ── to_numpy_action ───────────────────────────────────────────────────────────


def test_to_numpy_action_shape_and_dtype() -> None:
    t = torch.arange(6, dtype=torch.float64).unsqueeze(0)  # (1, 6) float64
    out = to_numpy_action(t)
    assert isinstance(out, np.ndarray)
    assert out.shape == (6,)
    assert out.dtype == np.float32
    assert out.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_to_numpy_action_detaches_grad() -> None:
    t = torch.zeros(1, 4, requires_grad=True)
    out = to_numpy_action(t)
    assert out.shape == (4,)


# ---------------------------------------------------------------------------
# call_make_processors_cached_first — HF revalidation suppression
# ---------------------------------------------------------------------------
#
# Reproduces the multi-HEAD HF burst the user reported when reloading a pi05
# rSkill against a warm paligemma tokenizer cache. The wrapper reads the
# preprocessor JSON, probes the local HF cache for the tokenizer it needs,
# and flips ``huggingface_hub.constants.HF_HUB_OFFLINE`` for the duration of
# the inner ``make_pre_post_processors`` call so transformers' AutoTokenizer
# fast-paths to cache-only reads (``is_offline_mode()`` returns ``True``).


def _write_preprocessor_json(directory: Any, tokenizer_name: str | None) -> None:
    """Write a minimal lerobot-shaped ``policy_preprocessor.json``."""
    import json
    from pathlib import Path

    payload: dict[str, Any] = {"name": "test_preprocessor", "steps": []}
    if tokenizer_name is not None:
        payload["steps"].append(
            {
                "registry_name": "tokenizer_processor",
                "config": {"tokenizer_name": tokenizer_name},
            }
        )
    (Path(directory) / "policy_preprocessor.json").write_text(json.dumps(payload))


def test_call_make_processors_cached_first_offline_when_tokenizer_cached(
    tmp_path: Any,
) -> None:
    import huggingface_hub.constants as hc
    from openral_rskill import _vla_core

    _write_preprocessor_json(tmp_path, tokenizer_name="google/paligemma-3b-pt-224")

    observed_offline: list[bool] = []

    def fake_make(policy_config: Any, *, pretrained_path: str, **_kwargs: Any) -> Any:
        observed_offline.append(hc.HF_HUB_OFFLINE)
        return ("PRE", "POST")

    # Pretend the tokenizer is cached.
    monkey_target = _vla_core._hf_tokenizer_is_cached
    _vla_core._hf_tokenizer_is_cached = lambda repo: True  # type: ignore[assignment]
    saved = hc.HF_HUB_OFFLINE
    try:
        out = _vla_core.call_make_processors_cached_first(
            fake_make, "CFG", pretrained_path=str(tmp_path)
        )
    finally:
        _vla_core._hf_tokenizer_is_cached = monkey_target  # type: ignore[assignment]
        hc.HF_HUB_OFFLINE = saved

    assert out == ("PRE", "POST")
    assert observed_offline == [True], (
        "wrapper must set HF_HUB_OFFLINE=True for the duration of the inner call"
    )
    # Restored after exit.
    assert hc.HF_HUB_OFFLINE is saved


def test_call_make_processors_cached_first_passthrough_when_not_cached(
    tmp_path: Any,
) -> None:
    import huggingface_hub.constants as hc
    from openral_rskill import _vla_core

    _write_preprocessor_json(tmp_path, tokenizer_name="google/paligemma-3b-pt-224")

    observed_offline: list[bool] = []

    def fake_make(policy_config: Any, *, pretrained_path: str, **_kwargs: Any) -> Any:
        observed_offline.append(hc.HF_HUB_OFFLINE)
        return ("PRE", "POST")

    monkey_target = _vla_core._hf_tokenizer_is_cached
    _vla_core._hf_tokenizer_is_cached = lambda repo: False  # type: ignore[assignment]
    saved = hc.HF_HUB_OFFLINE
    try:
        _vla_core.call_make_processors_cached_first(fake_make, "CFG", pretrained_path=str(tmp_path))
    finally:
        _vla_core._hf_tokenizer_is_cached = monkey_target  # type: ignore[assignment]
        hc.HF_HUB_OFFLINE = saved

    # Cold cache → don't force offline; let lerobot's first download happen.
    assert observed_offline == [saved]


def test_call_make_processors_cached_first_passthrough_when_no_tokenizer_step(
    tmp_path: Any,
) -> None:
    """ACT / Diffusion Policy preprocessors carry no tokenizer step."""
    import huggingface_hub.constants as hc
    from openral_rskill import _vla_core

    _write_preprocessor_json(tmp_path, tokenizer_name=None)

    observed_offline: list[bool] = []

    def fake_make(policy_config: Any, *, pretrained_path: str, **_kwargs: Any) -> Any:
        observed_offline.append(hc.HF_HUB_OFFLINE)
        return ("PRE", "POST")

    saved = hc.HF_HUB_OFFLINE
    try:
        _vla_core.call_make_processors_cached_first(fake_make, "CFG", pretrained_path=str(tmp_path))
    finally:
        hc.HF_HUB_OFFLINE = saved

    assert observed_offline == [saved]


def test_call_make_processors_cached_first_forwards_kwargs(tmp_path: Any) -> None:
    """Verify ``preprocessor_overrides`` / ``dataset_stats`` reach the inner call."""
    from openral_rskill import _vla_core

    _write_preprocessor_json(tmp_path, tokenizer_name=None)

    captured: dict[str, Any] = {}

    def fake_make(policy_config: Any, *, pretrained_path: str, **kwargs: Any) -> Any:
        captured["pretrained_path"] = pretrained_path
        captured["kwargs"] = kwargs
        return (None, None)

    _vla_core.call_make_processors_cached_first(
        fake_make,
        "CFG",
        pretrained_path=str(tmp_path),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    assert captured["pretrained_path"] == str(tmp_path)
    assert captured["kwargs"] == {"preprocessor_overrides": {"device_processor": {"device": "cpu"}}}


def test_read_tokenizer_repo_handles_missing_and_malformed(tmp_path: Any) -> None:
    from openral_rskill._vla_core import _read_tokenizer_repo_from_preprocessor

    # None path
    assert _read_tokenizer_repo_from_preprocessor(None) is None
    # Missing file
    assert _read_tokenizer_repo_from_preprocessor(str(tmp_path)) is None
    # Malformed JSON
    (tmp_path / "policy_preprocessor.json").write_text("{not json")
    assert _read_tokenizer_repo_from_preprocessor(str(tmp_path)) is None
    # Valid JSON, valid step
    _write_preprocessor_json(tmp_path, tokenizer_name="org/tok-x")
    assert _read_tokenizer_repo_from_preprocessor(str(tmp_path)) == "org/tok-x"
    # Restores after exit ensured by ``_write_preprocessor_json`` overwrite


# ── maybe_compile_chunk_forward safety gates ──────────────────────────────────


class _ChunkPolicy(torch.nn.Module):
    """Minimal ``nn.Module`` policy exposing a chunk forward over a persistent buffer.

    Stand-in for a lerobot policy (not an OpenRAL type — fixture, not mock).
    The persistent ``_buf`` mimics a CUDA-graph static output buffer: the
    eager forward returns the same storage on every call, which is exactly
    the aliasing pattern the cudagraph-mode clone gate must neutralise.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self._buf = torch.zeros(1, 4)

    def _get_action_chunk(self, batch: dict[str, Any]) -> torch.Tensor:
        self._buf.copy_(batch["observation.state"])
        return self._buf


class TestMaybeCompileChunkForward:
    def test_opt_in_required(self) -> None:
        from openral_rskill._vla_core import maybe_compile_chunk_forward

        policy = _ChunkPolicy()
        assert maybe_compile_chunk_forward(policy, {}, "cuda:0", torch) is False
        assert "_get_action_chunk" not in policy.__dict__  # method not wrapped

    def test_skipped_on_cpu_device(self) -> None:
        from openral_rskill._vla_core import maybe_compile_chunk_forward

        policy = _ChunkPolicy()
        assert maybe_compile_chunk_forward(policy, {"compile": True}, "cpu", torch) is False
        assert "_get_action_chunk" not in policy.__dict__

    def test_skips_bnb_quantized_policy(self) -> None:
        bnb = pytest.importorskip("bitsandbytes", reason="bitsandbytes not installed")
        from openral_rskill._vla_core import (
            _has_bnb_quantized_modules,
            maybe_compile_chunk_forward,
        )

        policy = _ChunkPolicy()
        assert _has_bnb_quantized_modules(policy) is False
        # Real bnb module, same rewrite quantize_nf4_in_place performs.
        policy.quantized = bnb.nn.Linear4bit(8, 8, bias=False, quant_type="nf4")
        assert _has_bnb_quantized_modules(policy) is True
        assert maybe_compile_chunk_forward(policy, {"compile": True}, "cuda:0", torch) is False
        assert "_get_action_chunk" not in policy.__dict__

    def test_has_bnb_handles_non_module_policy(self) -> None:
        from openral_rskill._vla_core import _has_bnb_quantized_modules

        assert _has_bnb_quantized_modules(_FakePolicy()) is False

    def test_cudagraph_mode_output_never_aliases_policy_buffer(self) -> None:
        """reduce-overhead outputs must own their storage (no static-buffer views).

        Holds on both branches of the wrapper: if Inductor compiles, the
        compiled output is cloned; if compilation fails at first call (no
        host compiler), the eager fallback output is cloned too.
        """
        from openral_rskill._vla_core import maybe_compile_chunk_forward

        policy = _ChunkPolicy()
        installed = maybe_compile_chunk_forward(
            policy,
            {"compile": True, "compile_mode": "reduce-overhead"},
            "cuda:0",
            torch,
        )
        assert installed is True
        batch = {"observation.state": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
        out = policy._get_action_chunk(batch)
        assert torch.equal(out, batch["observation.state"])
        assert out.data_ptr() != policy._buf.data_ptr()

        # Second chunk overwrites the policy's persistent buffer; the
        # first chunk's returned actions must be unaffected.
        batch2 = {"observation.state": torch.tensor([[9.0, 9.0, 9.0, 9.0]])}
        policy._get_action_chunk(batch2)
        assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


class TestCloneChunkOutput:
    def test_tensor_gets_new_storage(self) -> None:
        from openral_rskill._vla_core import _clone_chunk_output

        src = torch.arange(6, dtype=torch.float32)
        out = _clone_chunk_output(src, torch)
        assert torch.equal(out, src)
        assert out.data_ptr() != src.data_ptr()

    def test_nested_containers(self) -> None:
        from openral_rskill._vla_core import _clone_chunk_output

        a = torch.ones(2)
        b = torch.zeros(3)
        out = _clone_chunk_output({"chunk": (a, [b])}, torch)
        assert torch.equal(out["chunk"][0], a)
        assert out["chunk"][0].data_ptr() != a.data_ptr()
        assert out["chunk"][1][0].data_ptr() != b.data_ptr()

    def test_non_tensor_leaves_pass_through(self) -> None:
        from openral_rskill._vla_core import _clone_chunk_output

        assert _clone_chunk_output(None, torch) is None
        assert _clone_chunk_output(3.5, torch) == 3.5
        assert _clone_chunk_output({"n": 7}, torch) == {"n": 7}


# ── suppress_hf_weight_init / assert_all_parameters_finite ────────────────────


class TestSuppressHfWeightInit:
    """The load-time optimisation that skips HF's throwaway random init.

    Every case patches ``transformers.modeling_utils.PreTrainedModel``
    directly, so the whole class needs the real library — there is nothing
    meaningful left to assert without it (CLAUDE.md §1.11: skip, never fake).
    The base CI job installs the workspace without the VLA extras.
    """

    @pytest.fixture(autouse=True)
    def _require_transformers(self) -> None:
        pytest.importorskip(
            "transformers",
            reason="install the VLA stack to exercise the HF init suppression",
        )

    def test_init_is_suppressed_inside_and_restored_after(self) -> None:
        from openral_rskill._vla_core import suppress_hf_weight_init
        from transformers.modeling_utils import PreTrainedModel

        before = PreTrainedModel._init_weights
        with suppress_hf_weight_init():
            assert PreTrainedModel._init_weights is not before
        assert PreTrainedModel._init_weights is before

    def test_restored_on_exception(self) -> None:
        from openral_rskill._vla_core import suppress_hf_weight_init
        from transformers.modeling_utils import PreTrainedModel

        before = PreTrainedModel._init_weights
        with pytest.raises(RuntimeError, match="synthetic"), suppress_hf_weight_init():
            raise RuntimeError("synthetic")
        assert PreTrainedModel._init_weights is before

    def test_suppressed_init_is_a_noop(self) -> None:
        """The patched hook must accept HF's (self, module) signature and do nothing."""
        from openral_rskill._vla_core import suppress_hf_weight_init
        from transformers.modeling_utils import PreTrainedModel

        layer = torch.nn.Linear(4, 4)
        with torch.no_grad():
            layer.weight.fill_(1.5)
        with suppress_hf_weight_init():
            PreTrainedModel._init_weights(None, layer)  # type: ignore[arg-type]
        assert torch.allclose(layer.weight, torch.full_like(layer.weight, 1.5))


class TestAssertAllParametersFinite:
    """The guard that makes suppressed init safe."""

    def test_passes_on_finite_module(self) -> None:
        from openral_rskill._vla_core import assert_all_parameters_finite

        assert_all_parameters_finite(torch.nn.Linear(3, 3), repo_id="unit/finite")

    def test_raises_on_nan_parameter(self) -> None:
        from openral_rskill._vla_core import assert_all_parameters_finite

        module = torch.nn.Linear(3, 3)
        with torch.no_grad():
            module.weight[0, 0] = float("nan")
        with pytest.raises(ROSConfigError, match="NaN/Inf"):
            assert_all_parameters_finite(module, repo_id="unit/broken")

    def test_raises_on_inf_parameter(self) -> None:
        from openral_rskill._vla_core import assert_all_parameters_finite

        module = torch.nn.Linear(3, 3)
        with torch.no_grad():
            module.bias[1] = float("inf")
        with pytest.raises(ROSConfigError, match="unit/broken"):
            assert_all_parameters_finite(module, repo_id="unit/broken")

    def test_raises_on_all_zero_weight_matrix(self) -> None:
        """Fresh zero pages are finite — the other uninitialised state.

        A parameter a non-strict load skipped (key renamed after a
        dependency bump) lands on freshly-mapped all-zero memory under
        suppressed init; isfinite passes and the policy would silently emit
        wrong actions. A trained weight MATRIX is never exactly all-zero.
        """
        from openral_rskill._vla_core import assert_all_parameters_finite

        module = torch.nn.Linear(3, 3)
        with torch.no_grad():
            module.weight.zero_()
        with pytest.raises(ROSConfigError, match="entirely zero"):
            assert_all_parameters_finite(module, repo_id="unit/zeroed")

    def test_zero_bias_is_legitimate(self) -> None:
        """1-D parameters (biases, norms) are commonly zero-init — never flagged."""
        from openral_rskill._vla_core import assert_all_parameters_finite

        module = torch.nn.Linear(3, 3)
        with torch.no_grad():
            module.bias.zero_()
        assert_all_parameters_finite(module, repo_id="unit/zero-bias")
