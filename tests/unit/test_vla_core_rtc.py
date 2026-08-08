"""policy_extras.rtc parsing → lerobot RTCConfig, and its wiring into the executor."""

from typing import Any

import pytest
from openral_core.exceptions import ROSConfigError
from openral_rskill._vla_core import (
    _parse_rtc_config,
    build_chunk_executor,
    rtc_enabled_in_extra,
)


def test_no_rtc_block_returns_none() -> None:
    assert _parse_rtc_config({}, adapter_name="smolvla") is None


def test_full_block_parses() -> None:
    from lerobot.configs import RTCAttentionSchedule

    cfg = _parse_rtc_config(
        {
            "rtc": {
                "enabled": True,
                "execution_horizon": 8,
                "max_guidance_weight": 5.0,
                "prefix_attention_schedule": "exp",
            }
        },
        adapter_name="smolvla",
    )
    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.execution_horizon == 8
    assert cfg.max_guidance_weight == 5.0
    assert cfg.prefix_attention_schedule == RTCAttentionSchedule.EXP


def test_defaults_exp_schedule_and_enabled() -> None:
    from lerobot.configs import RTCAttentionSchedule

    cfg = _parse_rtc_config({"rtc": {}}, adapter_name="pi05")
    assert cfg.enabled is True
    assert cfg.execution_horizon == 10
    assert cfg.max_guidance_weight == 10.0
    assert cfg.prefix_attention_schedule == RTCAttentionSchedule.EXP


@pytest.mark.parametrize(
    "block",
    [
        "yes",  # not a mapping
        {"execution_horizon": 0},  # not positive
        {"execution_horizon": True},  # bool is not an int here
        {"max_guidance_weight": 0.0},  # RTCConfig rejects <= 0
        {"max_guidance_weight": None},  # a hole, not a number
        {"max_guidance_weight": "10.0"},  # quoted YAML number
        {"prefix_attention_schedule": "cubic"},  # unknown schedule
        {"enabled": "false"},  # quoted YAML bool: bool("false") is True
        {"debug": 1},  # ints are not booleans here
        {"bogus_knob": 1},  # unknown key
    ],
)
def test_invalid_blocks_raise(block: object) -> None:
    with pytest.raises(ROSConfigError):
        _parse_rtc_config({"rtc": block}, adapter_name="smolvla")


def test_real_booleans_are_accepted() -> None:
    """The flag guard rejects look-alikes, not the honest YAML booleans."""
    cfg = _parse_rtc_config({"rtc": {"enabled": False, "debug": True}}, adapter_name="smolvla")
    assert cfg is not None
    assert cfg.enabled is False
    assert cfg.debug is True


def test_non_flow_matching_adapter_rejected() -> None:
    with pytest.raises(ROSConfigError, match="flow-matching"):
        _parse_rtc_config({"rtc": {}}, adapter_name="act")


# ── rtc_enabled_in_extra: the smolvla factory's torch.compile gate ─────────
#
# The smolvla factory skips maybe_compile_chunk_forward on this predicate (RTC and
# torch.compile rewrite the same flow-matching forward). Testing the predicate rather
# than the factory: _build_smolvla loads a real checkpoint from the Hub, which is not
# a unit test. The factory's own branch is covered by the sim tier.


def test_rtc_enabled_in_extra_true_for_enabled_block() -> None:
    assert rtc_enabled_in_extra({"rtc": {}}, adapter_name="smolvla") is True
    assert rtc_enabled_in_extra({"rtc": {"enabled": True}}, adapter_name="smolvla") is True


def test_rtc_enabled_in_extra_false_for_disabled_block_keeps_compile() -> None:
    """A disabled rtc block must NOT cost the user torch.compile."""
    extra = {"rtc": {"enabled": False}, "compile": True, "compile_mode": "reduce-overhead"}
    assert rtc_enabled_in_extra(extra, adapter_name="smolvla") is False


def test_rtc_enabled_in_extra_false_without_block() -> None:
    assert rtc_enabled_in_extra({"compile": True}, adapter_name="smolvla") is False


def test_rtc_enabled_in_extra_propagates_malformed_block() -> None:
    """A bad manifest still fails loudly at this earlier call site."""
    with pytest.raises(ROSConfigError):
        rtc_enabled_in_extra({"rtc": {"bogus_knob": 1}}, adapter_name="smolvla")


# ── build_chunk_executor RTC wiring ─────────────────────────────────────────


class _FlowPolicyStub:  # NOT a mock of a boundary: exercises the documented lerobot surface
    """Minimal object satisfying the lerobot RTC-activation surface.

    Uses the real lerobot RTCProcessor via the same init_rtc_processor contract
    (mirrors SmolVLAPolicy.init_rtc_processor's documented behavior).
    """

    def __init__(self, chunk_size: int = 10) -> None:
        import types

        self.config = types.SimpleNamespace(
            n_action_steps=chunk_size, chunk_size=chunk_size, rtc_config=None
        )
        self.rtc_processor = None

    def init_rtc_processor(self) -> None:
        from lerobot.policies.rtc import RTCProcessor

        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

    def predict_action_chunk(self, batch: Any, **kwargs: Any) -> Any:
        import torch

        return torch.zeros(1, int(self.config.chunk_size), 3)


class _NoRTCPolicyStub:
    """A chunked policy that is not flow-matching — no ``init_rtc_processor``."""

    def __init__(self, chunk_size: int = 10) -> None:
        import types

        self.config = types.SimpleNamespace(n_action_steps=chunk_size)

    def predict_action_chunk(self, batch: Any) -> Any:
        import torch

        return torch.zeros(1, int(self.config.n_action_steps), 3)


def test_build_chunk_executor_activates_rtc() -> None:
    policy = _FlowPolicyStub()
    ex = build_chunk_executor(
        {"chunk_prefetch": True, "rtc": {"execution_horizon": 6}},
        policy=policy,
        adapter_name="smolvla",
    )
    assert ex is not None and ex._rtc_enabled
    assert policy.config.rtc_config is not None
    assert policy.config.rtc_config.execution_horizon == 6
    assert policy.rtc_processor is not None
    ex.stop()


def test_build_chunk_executor_rtc_requires_prefetch() -> None:
    with pytest.raises(ROSConfigError, match="chunk_prefetch"):
        build_chunk_executor({"rtc": {}}, policy=_FlowPolicyStub(), adapter_name="smolvla")


def test_build_chunk_executor_rtc_disabled_block_is_plain_executor() -> None:
    policy = _FlowPolicyStub()
    ex = build_chunk_executor(
        {"chunk_prefetch": True, "rtc": {"enabled": False}},
        policy=policy,
        adapter_name="smolvla",
    )
    assert ex is not None and not ex._rtc_enabled
    assert policy.rtc_processor is None
    ex.stop()


def test_build_chunk_executor_rtc_refuses_single_step_policy() -> None:
    """A chunk size of 1 would silently return no executor — RTC must fail loudly."""
    with pytest.raises(ROSConfigError, match="chunked execution"):
        build_chunk_executor(
            {"chunk_prefetch": True, "rtc": {}},
            policy=_FlowPolicyStub(chunk_size=1),
            adapter_name="smolvla",
        )


def test_build_chunk_executor_rtc_refuses_policy_without_processor() -> None:
    with pytest.raises(ROSConfigError, match="init_rtc_processor"):
        build_chunk_executor(
            {"chunk_prefetch": True, "rtc": {}},
            policy=_NoRTCPolicyStub(),
            adapter_name="smolvla",
        )


def test_build_chunk_executor_rtc_refuses_bnb_quantized_policy() -> None:
    """RTC guidance backpropagates through the denoiser — nf4 weights cannot."""
    bnb = pytest.importorskip("bitsandbytes", reason="install the sim group: just sync --group sim")
    import torch

    class _QuantizedFlowPolicy(torch.nn.Module, _FlowPolicyStub):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            _FlowPolicyStub.__init__(self)
            self.proj = bnb.nn.Linear4bit(4, 4)

    with pytest.raises(ROSConfigError, match="bitsandbytes"):
        build_chunk_executor(
            {"chunk_prefetch": True, "rtc": {}},
            policy=_QuantizedFlowPolicy(),
            adapter_name="smolvla",
        )
