"""Unit tests for the SmolVLA processor stats-fallback path.

Regression coverage for the fallback that was lost when PR #163's merge
(commit 9c9a0d5) accepted the cleaner-cli side of a conflict on
``python/sim/src/openral_sim/policies/smolvla.py`` and silently dropped
commit 1d76a44's ``_load_lerobot_dataset_stats`` + try/except wrapper.
Without the fallback, every community SmolVLA checkpoint uploaded with
only ``config.json`` + ``model.safetensors`` (e.g.
``Calvert0921/smolvla_franka_liftcube_1000``) raises ``Entry Not Found``
at load time.

Coverage
--------
- :func:`_is_processor_missing` — pure cause-chain walker; no deps.
- :func:`_load_lerobot_dataset_stats` — v3 (``meta/stats.json``) and
  v2.1 (``meta/episodes_stats.jsonl``) layouts. ``hf_hub_download`` is
  mocked at the network boundary (CLAUDE.md §5.4 allows this) to point
  at a real on-disk fixture; the aggregation logic and JSON parsing
  run for real against ``lerobot.datasets.compute_stats``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from huggingface_hub.errors import EntryNotFoundError
from openral_core.exceptions import ROSConfigError

# ``RemoteEntryNotFoundError`` subclasses ``EntryNotFoundError`` but its
# constructor requires a real ``httpx.Response``. The smolvla adapter's
# ``_PROCESSOR_MISSING_EXC`` catches both via the parent, so the parent
# class is sufficient (and far cheaper) to exercise the fallback paths.


# ── _is_processor_missing ────────────────────────────────────────────────────


def test_is_processor_missing_direct_entry() -> None:
    from openral_sim.policies.smolvla import _is_processor_missing

    assert _is_processor_missing(EntryNotFoundError("nope"))


def test_is_processor_missing_walks_cause_chain() -> None:
    from openral_sim.policies.smolvla import _is_processor_missing

    inner = EntryNotFoundError("missing")
    wrapped = ROSConfigError("processor_dir failed")
    wrapped.__cause__ = inner
    assert _is_processor_missing(wrapped)


def test_is_processor_missing_walks_context_chain() -> None:
    from openral_sim.policies.smolvla import _is_processor_missing

    inner = EntryNotFoundError("missing")
    wrapped = ROSConfigError("processor_dir failed")
    wrapped.__context__ = inner
    assert _is_processor_missing(wrapped)


def test_is_processor_missing_false_for_unrelated() -> None:
    from openral_sim.policies.smolvla import _is_processor_missing

    assert not _is_processor_missing(ValueError("typo"))
    assert not _is_processor_missing(ROSConfigError("bad manifest"))


# ── _load_lerobot_dataset_stats ──────────────────────────────────────────────


def _write_episodes_stats_fixture(path: Path, n_episodes: int = 2, n_action: int = 8) -> None:
    """Write a tiny but real-shaped LeRobotDataset v2.1 episodes_stats.jsonl."""
    rng = np.random.default_rng(0)
    with path.open("w") as f:
        for ep in range(n_episodes):
            action = rng.normal(size=(20, n_action)).astype(np.float64)
            state = rng.normal(size=(20, 9)).astype(np.float64)
            record = {
                "episode_index": ep,
                "stats": {
                    "action": {
                        "min": action.min(axis=0).tolist(),
                        "max": action.max(axis=0).tolist(),
                        "mean": action.mean(axis=0).tolist(),
                        "std": action.std(axis=0).tolist(),
                        "count": [len(action)],
                    },
                    "observation.state": {
                        "min": state.min(axis=0).tolist(),
                        "max": state.max(axis=0).tolist(),
                        "mean": state.mean(axis=0).tolist(),
                        "std": state.std(axis=0).tolist(),
                        "count": [len(state)],
                    },
                },
            }
            f.write(json.dumps(record) + "\n")


def test_load_dataset_stats_v3_layout(tmp_path: Path) -> None:
    """v3 datasets ship a single ``meta/stats.json``."""
    from openral_sim.policies.smolvla import _load_lerobot_dataset_stats

    stats_path = tmp_path / "stats.json"
    payload = {
        "action": {"mean": [0.0, 1.0], "std": [1.0, 2.0]},
        "observation.state": {"mean": [0.5], "std": [0.25]},
    }
    stats_path.write_text(json.dumps(payload))

    def fake_download(*, repo_id: str, filename: str, **_: Any) -> str:
        assert filename == "meta/stats.json"
        return str(stats_path)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        out = _load_lerobot_dataset_stats("hf://Calvert0921/SmolVLA_LiftCube_Franka_1000")

    assert set(out) == {"action", "observation.state"}
    assert isinstance(out["action"]["mean"], np.ndarray)
    np.testing.assert_array_equal(out["action"]["mean"], np.asarray([0.0, 1.0]))


def test_load_dataset_stats_v21_fallback(tmp_path: Path) -> None:
    """v2.1 datasets ship per-episode JSONL; aggregator runs for real."""
    pytest.importorskip("lerobot.datasets.compute_stats")
    from openral_sim.policies.smolvla import _load_lerobot_dataset_stats

    episodes_path = tmp_path / "episodes_stats.jsonl"
    _write_episodes_stats_fixture(episodes_path)

    def fake_download(*, repo_id: str, filename: str, **_: Any) -> str:
        if filename == "meta/stats.json":
            raise EntryNotFoundError("not a v3 dataset")
        if filename == "meta/episodes_stats.jsonl":
            return str(episodes_path)
        raise AssertionError(f"unexpected filename {filename!r}")

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
        out = _load_lerobot_dataset_stats("hf://Calvert0921/SmolVLA_LiftCube_Franka_1000")

    assert set(out) == {"action", "observation.state"}
    # Real aggregation: mean must be inside the per-episode min/max envelope.
    action_mean = out["action"]["mean"]
    assert action_mean.shape == (8,)
    assert np.all(out["action"]["min"] <= action_mean)
    assert np.all(action_mean <= out["action"]["max"])


def test_load_dataset_stats_raises_when_both_missing() -> None:
    from openral_sim.policies.smolvla import _load_lerobot_dataset_stats

    def fake_download(*, repo_id: str, filename: str, **_: Any) -> str:
        raise EntryNotFoundError(f"{filename} missing")

    with (
        patch("huggingface_hub.hf_hub_download", side_effect=fake_download),
        pytest.raises(ROSConfigError, match=r"neither meta/stats\.json"),
    ):
        _load_lerobot_dataset_stats("hf://owner/empty-dataset")


def test_load_dataset_stats_rejects_non_hf_uri() -> None:
    from openral_sim.policies.smolvla import _load_lerobot_dataset_stats

    with pytest.raises(ROSConfigError, match="only hf:// URIs"):
        _load_lerobot_dataset_stats("file:///tmp/nope")
