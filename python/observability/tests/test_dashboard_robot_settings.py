"""The robot console's own reasoner settings (GET/PUT /api/settings, robot_settings).

The operator at the robot sets the provider keys and the model in the console's
settings pane; a deploy maps them to the reasoner's variables at start. A key never
comes back in the clear, the LAN may only read, and nothing but a curated model is kept.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import httpx
import pytest
from openral_observability.dashboard import TelemetryStore, create_app
from openral_observability.dashboard.robot_settings import settings_path

_KEY = "sk-ant-api03-test-0000-1234"


@pytest.fixture(autouse=True)
def _config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config" / "robot.json"
    monkeypatch.setenv("OPENRAL_ROBOT_ENROLMENT", str(path))
    for var in ("OPENRAL_REASONER_MODEL", "OPENRAL_REASONER_API_KEY", "OPENRAL_REASONER_ENDPOINT"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    return path


def _client(client: tuple[str, int] = ("127.0.0.1", 50000)) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app(TelemetryStore()), client=client)
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:4318", timeout=10.0)


@pytest.mark.asyncio
async def test_the_local_operator_sets_a_key_and_a_model_and_never_reads_the_key_back() -> None:
    async with _client() as c:
        empty = (await c.get("/api/settings")).json()
        assert (empty["scope"], empty["canWrite"], empty["values"]) == ("robot", True, {})
        assert any(
            o["id"] == "claude-sonnet-5-5"
            for o in empty["settings"]["OPENRAL_REASONER_MODEL"]["options"]
        )
        r = await c.put("/api/settings", json={"name": "ANTHROPIC_API_KEY", "value": _KEY})
        assert r.status_code == 200, r.text
        r = await c.put(
            "/api/settings", json={"name": "OPENRAL_REASONER_MODEL", "value": "claude-sonnet-5-5"}
        )
        body = r.json()
    assert body["values"]["ANTHROPIC_API_KEY"] == {"set": True, "hint": "…1234"}
    assert body["values"]["OPENRAL_REASONER_MODEL"] == {"set": True, "value": "claude-sonnet-5-5"}
    assert _KEY not in r.text
    assert stat.S_IMODE(os.stat(settings_path()).st_mode) == 0o600


@pytest.mark.asyncio
async def test_unknown_settings_and_uncurated_models_are_refused() -> None:
    async with _client() as c:
        assert (
            await c.put(
                "/api/settings", json={"name": "OPENRAL_DASHBOARD_REMOTE_CONTROL", "value": "1"}
            )
        ).status_code == 400
        assert (
            await c.put(
                "/api/settings", json={"name": "OPENRAL_REASONER_MODEL", "value": "llama-3"}
            )
        ).status_code == 400
        assert (await c.put("/api/settings", json=["not", "an", "object"])).status_code == 400
    assert not settings_path().exists()


@pytest.mark.asyncio
async def test_the_lan_reads_but_cannot_write_and_another_site_cannot_write() -> None:
    async with _client(("192.168.1.20", 50000)) as lan:
        body = (await lan.get("/api/settings")).json()
        assert body["canWrite"] is False
        assert (
            await lan.put("/api/settings", json={"name": "ANTHROPIC_API_KEY", "value": _KEY})
        ).status_code == 403
    async with _client() as c:
        r = await c.put(
            "/api/settings",
            json={"name": "ANTHROPIC_API_KEY", "value": _KEY},
            headers={"Origin": "https://evil.example"},
        )
        assert r.status_code == 403
    assert not settings_path().exists()


def test_a_deploy_applies_them_with_the_environment_winning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openral_cli.robot import apply_console_settings
    from openral_observability.dashboard.robot_settings import save_setting

    save_setting("OPENAI_API_KEY", "sk-openai-5678")
    save_setting("OPENRAL_REASONER_MODEL", "gpt-5.5")
    applied = apply_console_settings(echo=lambda _: None)
    assert applied == [
        "OPENRAL_REASONER_API_KEY",
        "OPENRAL_REASONER_ENDPOINT",
        "OPENRAL_REASONER_MODEL",
    ]
    assert (os.environ["OPENRAL_REASONER_MODEL"], os.environ["OPENRAL_REASONER_ENDPOINT"]) == (
        "gpt-5.5",
        "openai",
    )
    # A variable set before the deploy wins over the console.
    monkeypatch.setenv("OPENRAL_REASONER_MODEL", "claude-opus-5-5")
    monkeypatch.delenv("OPENRAL_REASONER_API_KEY")
    monkeypatch.delenv("OPENRAL_REASONER_ENDPOINT")
    assert "OPENRAL_REASONER_MODEL" not in apply_console_settings(echo=lambda _: None)
    assert os.environ["OPENRAL_REASONER_MODEL"] == "claude-opus-5-5"


def test_a_cloud_model_without_its_key_leaves_the_account_settings_in_charge() -> None:
    """A model saved, its key cleared: the console chooses nothing (the account path runs)."""
    from openral_observability.dashboard.robot_settings import (
        load_settings,
        reasoner_env,
        save_setting,
    )

    save_setting("OPENAI_API_KEY", "sk-openai-5678")
    save_setting("OPENRAL_REASONER_MODEL", "gpt-5.5")
    assert reasoner_env(load_settings())
    save_setting("OPENAI_API_KEY", "")
    assert reasoner_env(load_settings()) == {}
