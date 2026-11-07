"""MakerMods' π0.5 LoRA on its own teleop data, open loop: the merge makes the policy.

Drives the real adapter (``_build_pi05``) on ``rskills/pi05-metal-pick-place``:
``lerobot/pi05_base`` loaded, ``makermods/pi05_metal_pick_place_lora`` merged
(``openral_sim.policies._lora``), the LoRA repo's processors. For recorded
frames of ``HappyEthan/metal_pick_place`` (the training set: top + wrist video,
7-D state) the state goes robot units -> ``PolicyIOCodec`` -> the adapter's
``step`` and the first ``_HORIZON`` actions of the chunk are compared with the
recorded ones. Then the LoRA is subtracted again from the same weights and the
base alone runs the same frames: it must be clearly worse, or the merge did
nothing. Also asserts the PaliGemma prompt tokenises to real ids (a cache
missing ``tokenizer.model`` silently yields ``<unk>`` and a blind policy).

Needs ~10 GiB of free device memory, the 14.5 GB base, the adapter and two
dataset video files from the Hub; absent any is the legitimate skip
(CLAUDE.md §1.12). π0.5 bf16 peaks above an 8 GiB card.
"""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.policies.pi05.modeling_pi05")
av = pytest.importorskip("av")
pytest.importorskip("pyarrow")

_REPO = Path(__file__).resolve().parents[2]
_RSKILL = _REPO / "rskills" / "pi05-metal-pick-place"
_ROBOT = _REPO / "robots" / "makermods_metal" / "robot.yaml"
_DATASET = "HappyEthan/metal_pick_place"
_DATASET_REV = "aaf1afccb83bc99b48a0f8c39ce146bbbc5b4626"
_EPISODES = (0, 1)
# Frame 0 is skipped: the leader arm is already commanding the closed gripper
# (action 1.2) while the follower still reads open (state 88.8 in episode 0), a
# teleop start transient no observation-conditioned policy can predict.
_STARTS = (30, 150, 270, 390, 510)
_HORIZON = 10
_MIN_FREE_GIB = 10.0
_JOINT_MAE_DEG = 3.5
_GRIPPER_MAE = 10.0
_BASE_RATIO = 2.5

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _dataset_frames() -> list[dict[str, Any]]:
    """Recorded (top, wrist, state, next-_HORIZON actions) at each start of each episode."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            _DATASET,
            repo_type="dataset",
            revision=_DATASET_REV,
            allow_patterns=[
                "meta/**",
                "data/chunk-000/file-000.parquet",
                "videos/*/chunk-000/file-000.mp4",
            ],
        )
    )
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    data = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    index = np.asarray(data["index"])
    states = np.asarray(data["observation.state"], dtype=np.float32)
    actions = np.asarray(data["action"], dtype=np.float32)
    out = []
    for ep in _EPISODES:
        meta = episodes[ep]
        assert meta["data/file_index"] == 0
        frames: dict[str, dict[int, NDArray[np.uint8]]] = {}
        for cam in ("top", "wrist"):
            key = f"videos/observation.images.{cam}"
            assert meta[f"{key}/file_index"] == 0
            t0 = meta[f"{key}/from_timestamp"]
            wanted = {round((t0 + s / 30.0) * 30.0): s for s in _STARTS}
            frames[cam] = {}
            with av.open(str(root / key / "chunk-000/file-000.mp4")) as c:
                for f in c.decode(c.streams.video[0]):
                    s = wanted.get(round(float(f.time) * 30.0))
                    if s is not None:
                        frames[cam][s] = f.to_ndarray(format="rgb24")
                    if len(frames[cam]) == len(_STARTS):
                        break
        for s in _STARTS:
            row = int(np.flatnonzero(index == meta["dataset_from_index"] + s)[0])
            out.append(
                {
                    "episode": ep,
                    "start": s,
                    "top": frames["top"][s],
                    "wrist": frames["wrist"][s],
                    "state": states[row],
                    "actions": actions[row : row + _HORIZON],
                }
            )
    return out


@pytest.fixture(scope="module")
def adapter() -> Iterator[Any]:
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    from openral_core import VLASpec
    from openral_sim.policies.pi05 import _build_pi05

    free, _ = torch.cuda.mem_get_info()
    if free / 2**30 < _MIN_FREE_GIB:
        pytest.skip(f"π0.5 bf16 needs >= {_MIN_FREE_GIB} GiB free; {free / 2**30:.1f} free")
    spec = VLASpec(
        id="pi05",
        weights_uri=str(_RSKILL),
        device="cuda:0",
        extra={"camera_keys": ["top", "wrist"]},
    )
    try:
        built = _build_pi05(SimpleNamespace(vla=spec, scene=None))
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
        pytest.skip(f"weights not fetchable: {exc}")
    yield built
    built.close()
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def frames() -> list[dict[str, Any]]:
    try:
        return _dataset_frames()
    except OSError as exc:
        pytest.skip(f"dataset not fetchable: {exc}")


@pytest.fixture(scope="module")
def codec() -> Any:
    from openral_core.schemas import RobotDescription, RSkillManifest
    from openral_rskill._policy_io import PolicyIOCodec

    manifest = RSkillManifest.from_yaml(str(_RSKILL / "rskill.yaml"))
    return PolicyIOCodec.from_manifest(manifest, RobotDescription.from_yaml(str(_ROBOT)))


def _robot_units(policy_state: NDArray[np.float32]) -> NDArray[np.float64]:
    """Dataset units (deg, stroke) -> robot units (rad, normalised gripper)."""
    return np.concatenate([np.radians(policy_state[:6]), policy_state[6:] / 100.0])


def _rollout(
    adapter: Any, codec: Any, frames: list[dict[str, Any]], prompt: str
) -> NDArray[np.float32]:
    """Per frame, the first _HORIZON policy actions; returns |pred - recorded| (N, H, 7)."""
    errs = []
    for i, fr in enumerate(frames):
        state = codec.to_policy_state(_robot_units(fr["state"]))
        np.testing.assert_allclose(state, fr["state"], atol=1e-3)  # codec round trip
        obs = {"images": {"top": fr["top"], "wrist": fr["wrist"]}, "state": state}
        torch.manual_seed(1000 + i)
        adapter.reset()
        pred = np.stack([adapter.step(obs, prompt) for _ in range(_HORIZON)])
        errs.append(np.abs(pred - fr["actions"]))
    return np.stack(errs)


def _unmerge(adapter: Any) -> None:
    """Subtract the LoRA delta from the live weights again: the base policy, same load."""
    from huggingface_hub import hf_hub_download
    from openral_core.schemas import RSkillManifest
    from openral_rskill.loader import resolve_rskill_to_hf_with_revision
    from openral_sim.policies._lora import read_lora_config
    from safetensors.torch import load_file

    repo, rev = resolve_rskill_to_hf_with_revision(str(_RSKILL))
    assert RSkillManifest.from_yaml(str(_RSKILL / "rskill.yaml")).base_weights_uri
    path = Path(hf_hub_download(repo, "adapter_model.safetensors", revision=rev))
    cfg = read_lora_config(path.parent)
    tensors = load_file(str(path))
    modules = dict(adapter._policy.named_modules())
    with torch.no_grad():
        for key, a in tensors.items():
            if not key.endswith(".lora_A.weight"):
                continue
            name = key[len("base_model.model.") : -len(".lora_A.weight")]
            b = tensors[key.replace("lora_A", "lora_B")]
            w = modules[name].weight
            delta = cfg.scale * (b.to(w.device).float() @ a.to(w.device).float())
            w.copy_((w.float() - delta).to(w.dtype))


def test_prompt_tokenises_to_real_ids(adapter: Any, frames: list[dict[str, Any]]) -> None:
    from openral_core.schemas import RSkillManifest

    prompt = RSkillManifest.from_yaml(str(_RSKILL / "rskill.yaml")).default_prompt
    assert prompt is not None
    fr = frames[0]
    batch = adapter._prepared_batch(
        {"images": {"top": fr["top"], "wrist": fr["wrist"]}, "state": fr["state"]}, prompt
    )
    tokens = batch["observation.language.tokens"][0]
    mask = batch["observation.language.attention_mask"][0].bool()
    tok = next(
        s for s in adapter._preprocessor.steps if type(s).__name__ == "TokenizerProcessorStep"
    ).input_tokenizer
    real = tokens[mask].tolist()
    print("prompt tokens:", len(real), tok.decode(real)[:160])
    assert tok.unk_token_id not in real
    assert "cube" in tok.decode(real)


def test_lora_tracks_recorded_actions_and_base_does_not(
    adapter: Any, codec: Any, frames: list[dict[str, Any]]
) -> None:
    from openral_core.schemas import RSkillManifest

    prompt = RSkillManifest.from_yaml(str(_RSKILL / "rskill.yaml")).default_prompt
    assert prompt is not None
    lora = _rollout(adapter, codec, frames, prompt)
    _unmerge(adapter)
    base = _rollout(adapter, codec, frames, prompt)
    report = {
        name: {
            "joint_mae_deg_per_joint": e[..., :6].mean(axis=(0, 1)).round(3).tolist(),
            "joint_mae_deg": float(e[..., :6].mean()),
            "gripper_mae_stroke": float(e[..., 6].mean()),
        }
        for name, e in (("lora", lora), ("base", base))
    }
    print(json.dumps(report, indent=1))
    assert report["lora"]["joint_mae_deg"] < _JOINT_MAE_DEG
    assert report["lora"]["gripper_mae_stroke"] < _GRIPPER_MAE
    assert report["base"]["joint_mae_deg"] > _BASE_RATIO * report["lora"]["joint_mae_deg"]
    assert math.isfinite(report["base"]["joint_mae_deg"])
