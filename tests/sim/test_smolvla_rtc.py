"""Guided SmolVLA chunk on the real eraser_place checkpoint (GPU).

Two claims about Real-Time Chunking that no CPU or stub test can make:

1. **The guidance survives OpenRAL's inference seam.** Every VLA call in this
   repo goes through :func:`openral_rskill._vla_core.run_inference`, which wraps
   the policy in ``torch.no_grad()``. lerobot's ``RTCProcessor.denoise_step``
   opens a ``torch.enable_grad()`` block and calls
   ``torch.autograd.grad(x1_t, x_t, ...)`` inside it. The VJP it takes happens
   to be the identity (``v_t`` is computed *before* ``x_t.requires_grad_``), so
   the guidance does not need gradients *through the model* — but the
   ``autograd.grad`` call still executes, and under an un-escaped ``no_grad``
   (or an ``inference_mode``, which ``enable_grad`` cannot escape) it raises.
2. **The guidance is not a no-op.** It has to actually pull the new chunk's
   prefix toward the previous chunk's unconsumed tail, which is the whole point
   of RTC: the executor swaps chunks mid-flight and the seam must not jump.

Real everything (CLAUDE.md §1.11): the shipped manifest's own ``policy_extras.rtc``
block parsed by the production :func:`openral_rskill._vla_core._parse_rtc_config`,
the real checkpoint at its shipped precision, and a real frame from its training
dataset — the same fixture ``tests/integration/test_smolvla_eraser_place_rskill.py``
replays, including the real training instruction ("place the erase on the blue
square", upstream typo and all) carried by the dataset row. Skipped without
CUDA, without lerobot, or when the checkpoint/dataset is neither cached nor
reachable; never faked.

Measured on an RTX 4070 Laptop: 20 s end-to-end (warm cache), guided prefix gap
0.718 against an unguided 1.151 — the guidance closes 38% of the seam.

Named for the behavior under test rather than the ``test_<robot>_<vla>_<sim>.py``
convention of this directory: there is no simulator here, only the policy and
its real inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import pytest
from openral_core.schemas import RSkillManifest

# `just test-sim` selects on `-m sim`; the CPU `sim-mujoco.yml` job deselects
# `slow`, which is where a 450 M-param GPU checkpoint belongs.
pytestmark = [pytest.mark.sim, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "rskills" / "rskill-smolvla-so101-eraser_place-bf16" / "rskill.yaml"

# The frame the manifest acceptance test is calibrated on: mid-episode, gripper
# closing on the eraser, so the chunk carries real motion in every joint.
_FRAME = 100

# Steps the arm consumes while the prefetch is in flight. ``ChunkedExecutor``
# passes its measured value here; 3 is a representative 100 ms at 30 Hz.
_DELAY = 3


class _Rollout(NamedTuple):
    """The two chunks under comparison, plus the tail the second was guided by."""

    base: Any  # (1, chunk, dim) torch tensor — unguided reference chunk
    guided: Any  # (1, chunk, dim) torch tensor — same noise, RTC kwargs populated
    prev: Any  # (horizon, dim) torch tensor — ``base``'s unconsumed tail
    horizon: int


@pytest.fixture(scope="module")
def rollout() -> _Rollout:
    """Load the checkpoint once and produce the unguided + guided chunk pair.

    Both calls share one observation and one fixed noise tensor, so the only
    difference between them is the RTC guidance kwargs.
    """
    torch = pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("lerobot", reason="lerobot not installed")
    pytest.importorskip("datasets", reason="lerobot[dataset] extra not installed")
    pytest.importorskip("transformers", reason="lerobot[smolvla] extra not installed")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: the 450 M-param denoiser runs 50 steps twice here")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from openral_rskill._vla_core import _parse_rtc_config, run_inference
    from openral_rskill.loader import resolve_rskill_to_hf_with_revision

    device = "cuda:0"
    manifest = RSkillManifest.from_yaml(str(_MANIFEST_PATH))
    assert manifest.dataset_uri is not None
    assert manifest.image_preprocessing is not None

    # Same resolution path a deploy takes: hand the loader the rSkill reference
    # and let it split the `@<sha>` pin off `weights_uri` into its own kwarg.
    repo_id, revision = resolve_rskill_to_hf_with_revision(str(_MANIFEST_PATH.parent))
    try:
        dataset = LeRobotDataset(manifest.dataset_uri.removeprefix("hf://"))
        config = PreTrainedConfig.from_pretrained(repo_id, revision=revision)
        config.device = device
        policy = (
            SmolVLAPolicy.from_pretrained(repo_id, config=config, revision=revision)
            .to(device)
            .eval()
        )
    except OSError as exc:  # offline and not cached — skip, never fake
        pytest.skip(f"checkpoint or dataset unavailable for {repo_id!r}: {exc}")

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=repo_id,
        revision=revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # Replay the deploy wiring: the runner re-keys the robot's sensors onto the
    # camera1/camera2 slots, then the manifest's aliases rename those slots to
    # the checkpoint's own input-feature keys.
    aliases = manifest.image_preprocessing.aliases
    sample = dataset[_FRAME]
    batch = preprocessor(
        {
            "observation.state": sample["observation.state"][None].to(device),
            f"observation.images.{aliases['camera1']}": (
                sample["observation.images.front"][None].to(device)
            ),
            f"observation.images.{aliases['camera2']}": (
                sample["observation.images.wrist"][None].to(device)
            ),
            "task": sample["task"],
        }
    )

    # The manifest's own RTC block, through the production parser.
    rtc_cfg = _parse_rtc_config(dict(manifest.policy_extras), adapter_name="smolvla")
    assert rtc_cfg is not None and rtc_cfg.enabled, (
        f"{_MANIFEST_PATH} must ship an enabled policy_extras.rtc block"
    )
    policy.config.rtc_config = rtc_cfg
    policy.init_rtc_processor()

    cfg = policy.config
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=generator).to(device)

    # `predict_action_chunk` stacks the observation queues in place, so each
    # call gets its own dict view and a cleared queue.
    policy.reset()
    base = run_inference(
        policy,
        dict(batch),
        kind="foreground",
        chunk_size=cfg.chunk_size,
        call=policy.predict_action_chunk,
        call_kwargs={"noise": noise},
    )

    # The tail the executor would still have queued when the prefetch fires:
    # everything past `inference_delay`, truncated to the execution horizon the
    # way `ChunkedExecutor._launch_prefetch` truncates it (guidance weights are
    # zero past the horizon anyway).
    horizon = int(rtc_cfg.execution_horizon)
    prev = base.squeeze(0)[_DELAY : _DELAY + horizon].clone()

    policy.reset()
    guided = run_inference(
        policy,
        dict(batch),
        kind="prefetch",
        chunk_size=cfg.chunk_size,
        call=policy.predict_action_chunk,
        call_kwargs={
            "noise": noise,
            "inference_delay": _DELAY,
            "prev_chunk_left_over": prev,
        },
    )
    return _Rollout(base=base, guided=guided, prev=prev, horizon=horizon)


def test_guided_chunk_survives_the_no_grad_seam(rollout: _Rollout) -> None:
    """The autograd call inside ``run_inference``'s ``no_grad`` produced a real chunk.

    Reaching this assertion at all is most of the claim — an un-escaped
    ``no_grad`` makes ``torch.autograd.grad`` raise inside the fixture. The last
    assertion rules out the quieter failure: RTC kwargs accepted, silently
    dropped, chunk returned unchanged.
    """
    torch = pytest.importorskip("torch", reason="torch not installed")

    assert rollout.guided.shape == rollout.base.shape
    assert torch.isfinite(rollout.guided).all()
    assert not torch.equal(rollout.guided, rollout.base), (
        "guided and unguided chunks are bit-identical — the RTC kwargs never reached the denoiser"
    )


def test_guidance_pulls_the_prefix_toward_the_previous_tail(rollout: _Rollout) -> None:
    """The guided chunk's prefix lands closer to the executing chunk's tail.

    Both chunks come from the same observation and the same noise, so the gap
    difference is attributable to the guidance alone.
    """
    horizon = rollout.horizon
    prefix_gap = (rollout.guided.squeeze(0)[:horizon] - rollout.prev).norm().item()
    unguided_gap = (rollout.base.squeeze(0)[:horizon] - rollout.prev).norm().item()

    assert prefix_gap < unguided_gap, (
        f"RTC guidance did not pull the prefix toward the previous tail: "
        f"guided gap {prefix_gap:.4f} >= unguided gap {unguided_gap:.4f}"
    )
