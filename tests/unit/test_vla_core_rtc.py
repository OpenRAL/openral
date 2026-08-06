"""policy_extras.rtc parsing → lerobot RTCConfig."""

import pytest
from openral_core.exceptions import ROSConfigError
from openral_rskill._vla_core import _parse_rtc_config


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
        {"prefix_attention_schedule": "cubic"},  # unknown schedule
        {"bogus_knob": 1},  # unknown key
    ],
)
def test_invalid_blocks_raise(block: object) -> None:
    with pytest.raises(ROSConfigError):
        _parse_rtc_config({"rtc": block}, adapter_name="smolvla")


def test_non_flow_matching_adapter_rejected() -> None:
    with pytest.raises(ROSConfigError, match="flow-matching"):
        _parse_rtc_config({"rtc": {}}, adapter_name="act")
