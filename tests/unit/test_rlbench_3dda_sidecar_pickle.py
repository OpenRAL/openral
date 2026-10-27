"""3D Diffuser Actor: what the sidecar loads cannot run code, and is provenance-gated.

``instructions.pkl`` is a plain ``pickle.dump`` of CLIP-embedding tensors; the
sidecar's ``_TensorPickleLoader`` must rebuild those and refuse every other
global (CWE-502). The adapter applies ``OPENRAL_REQUIRE_SIGNED_SKILLS`` before
the sidecar ever sees the checkpoint.
"""

from __future__ import annotations

import importlib.util
import io
import pickle
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

_SIDECAR = Path(__file__).resolve().parents[2] / "tools" / "rlbench_3dda_sidecar.py"


def _sidecar() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rlbench_3dda_sidecar", _SIDECAR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_instructions_pickle_round_trips_tensors() -> None:
    embedding = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    blob = pickle.dumps({"close_jar": {0: embedding}})

    loaded = _sidecar()._TensorPickleLoader(io.BytesIO(blob), torch).load()

    assert torch.equal(loaded["close_jar"][0], embedding)


class _Payload:
    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (print, ("instructions.pkl ran code",))


def test_instructions_pickle_refuses_arbitrary_globals() -> None:
    blob = pickle.dumps({"close_jar": {0: _Payload()}})

    with pytest.raises(pickle.UnpicklingError, match=r"builtins\.print"):
        _sidecar()._TensorPickleLoader(io.BytesIO(blob), torch).load()


def test_required_signatures_refuse_before_the_sidecar_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openral_core import VLASpec
    from openral_core.exceptions import ROSConfigError
    from openral_sim.policies.rlbench_3dda import _build_diffuser_actor

    checkpoint = tmp_path / "diffuser_actor_peract.pth"
    checkpoint.touch()
    monkeypatch.setenv("OPENRAL_3DDA_REPO", str(tmp_path))
    monkeypatch.setenv("OPENRAL_3DDA_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("OPENRAL_REQUIRE_SIGNED_SKILLS", "1")
    skill = Path(__file__).resolve().parents[2] / "rskills" / "3d-diffuser-actor-rlbench"
    env = type("Env", (), {"vla": VLASpec(id="diffuser_actor", weights_uri=str(skill))})()
    with pytest.raises(ROSConfigError, match="OPENRAL_REQUIRE_SIGNED_SKILLS"):
        _build_diffuser_actor(env)  # only .vla is read before the gate
