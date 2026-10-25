"""Gate branches of the ACT device-resident NVMM path (``_maybe_build_act_nvmm``).

The executor itself ships in the private openral-pro-trt package and is
exercised by openral-pro's GPU suite. What lives here is the open-side gate:
each condition that sends ACT to the host path must return ``None`` (never
raise), and the Pro modules are reached only through
``resolve_pro_module`` — a host with TensorRT but no Pro install degrades to
the host path instead of crashing on a missing ``openral_pro_trt`` import.
"""

from __future__ import annotations

import importlib.util

import pytest
from openral_sim.policies.act import _maybe_build_act_nvmm

_HAS_TENSORRT = importlib.util.find_spec("tensorrt") is not None


def test_flag_off_is_host_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRAL_ACT_TRT", raising=False)
    monkeypatch.setenv("OPENRAL_ACT_DEVICE_ONNX", "act_device.onnx")
    assert (
        _maybe_build_act_nvmm(None, "lerobot/act_aloha_sim_transfer_cube_human", "cuda:0", None)
        is None
    )


def test_no_device_onnx_is_host_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_ACT_TRT", "1")
    monkeypatch.delenv("OPENRAL_ACT_DEVICE_ONNX", raising=False)
    assert (
        _maybe_build_act_nvmm(None, "lerobot/act_aloha_sim_transfer_cube_human", "cuda:0", None)
        is None
    )


def test_cpu_device_is_host_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_ACT_TRT", "1")
    monkeypatch.setenv("OPENRAL_ACT_DEVICE_ONNX", "act_device.onnx")
    assert (
        _maybe_build_act_nvmm(None, "lerobot/act_aloha_sim_transfer_cube_human", "cpu", None)
        is None
    )


@pytest.mark.skipif(_HAS_TENSORRT, reason="tensorrt is installed; this asserts its absence")
def test_no_tensorrt_is_host_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_ACT_TRT", "1")
    monkeypatch.setenv("OPENRAL_ACT_DEVICE_ONNX", "act_device.onnx")
    assert (
        _maybe_build_act_nvmm(None, "lerobot/act_aloha_sim_transfer_cube_human", "cuda:0", None)
        is None
    )


@pytest.mark.skipif(not _HAS_TENSORRT, reason="needs tensorrt to reach the Pro-module lookup")
def test_tensorrt_without_pro_modules_is_host_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRAL_ACT_TRT", "1")
    monkeypatch.setenv("OPENRAL_ACT_DEVICE_ONNX", "act_device.onnx")
    monkeypatch.setattr("openral_rskill.backend_registry.entry_points", lambda group: ())
    assert (
        _maybe_build_act_nvmm(None, "lerobot/act_aloha_sim_transfer_cube_human", "cuda:0", None)
        is None
    )
