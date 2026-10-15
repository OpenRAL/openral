"""ChunkedExecutor ``boundary_blend_steps``: a new chunk's head continues from the served tail.

Regression for openral/openral#378. Without RTC the executor serves the old chunk to
its last row and then row 0 of the next one, which on the OpenArm restock π0.5 sat
0.5-1 rad from where the arm had been sent (#370 r41: the kernel's workspace speed
check read those jumps as fast commands and braked hundreds of times per run). The
opt-in blend spreads that offset over the first ``K`` rows of the new chunk, so the
boundary step is ``1/(K+1)`` of the raw jump and row ``K`` onward is the policy's own.

Real ``ChunkedExecutor`` + ``build_chunk_executor``; the chunk producers are plain
NumPy / torch functions through the documented ``chunk_fn`` API.
"""

from __future__ import annotations

import functools
import itertools
from typing import Any

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from openral_core.exceptions import ROSConfigError
from openral_rskill._vla_core import build_chunk_executor
from openral_rskill.executor import ChunkedExecutor

CHUNK, DOF = 8, 4
JUMP = 1.0  # rad between consecutive chunk heads — the #370 magnitude
SLOPE = 0.01  # rad per step inside a chunk


class _JumpingProducer:
    """Chunk ``c`` ramps from ``c * JUMP`` by ``SLOPE`` per row, on every dim.

    Chunk ``c`` ends at ``c * JUMP + (CHUNK - 1) * SLOPE`` and chunk ``c + 1`` starts
    at ``(c + 1) * JUMP``, so every boundary jumps by ``JUMP - (CHUNK - 1) * SLOPE``.
    """

    def __init__(self, *, as_torch: bool = False) -> None:
        self.calls = 0
        self._as_torch = as_torch

    def __call__(self, payload: Any) -> list[Any]:
        base = self.calls * JUMP
        self.calls += 1
        rows = [np.full(DOF, base + i * SLOPE, dtype=np.float32) for i in range(CHUNK)]
        if self._as_torch:
            return [torch.from_numpy(r).unsqueeze(0) for r in rows]
        return rows


def _raw_row(chunk: int, row: int) -> NDArray[np.float32]:
    return np.full(DOF, chunk * JUMP + row * SLOPE, dtype=np.float32)


def _serve(ex: ChunkedExecutor, n: int) -> list[NDArray[np.float32]]:
    ex.start()
    out: list[NDArray[np.float32]] = [
        np.asarray(ex.select_action({"obs": 0}), dtype=np.float32).reshape(-1) for _ in range(n)
    ]
    ex.stop()
    return out


def _sync(steps: int = 0, dims: tuple[int, ...] | None = None, **kw: Any) -> ChunkedExecutor:
    return ChunkedExecutor(
        chunk_fn=kw.pop("chunk_fn", _JumpingProducer()),
        chunk_size=CHUNK,
        prefetch_at=kw.pop("prefetch_at", 0),
        boundary_blend_steps=steps,
        blend_dims=dims,
        **kw,
    )


class TestBoundaryBlend:
    def test_default_is_byte_identical_to_unblended(self) -> None:
        served = _serve(_sync(), 3 * CHUNK)
        for c in range(3):
            for i in range(CHUNK):
                np.testing.assert_array_equal(served[c * CHUNK + i], _raw_row(c, i))

    def test_row_zero_continues_from_the_served_tail(self) -> None:
        k = 3
        served = _serve(_sync(k), 2 * CHUNK)
        tail, head = served[CHUNK - 1], served[CHUNK]
        raw_jump = _raw_row(1, 0) - tail
        # The boundary step is the raw jump split into K + 1 equal parts.
        np.testing.assert_allclose(head - tail, raw_jump / (k + 1), rtol=1e-5)
        assert np.max(np.abs(head - tail)) < np.max(np.abs(raw_jump))

    def test_offset_decays_to_the_raw_chunk_after_k_rows(self) -> None:
        k = 3
        served = _serve(_sync(k), 2 * CHUNK)
        delta = served[CHUNK - 1] - _raw_row(1, 0)
        for i in range(k):
            expected = _raw_row(1, i) + delta * (1.0 - (i + 1) / (k + 1))
            np.testing.assert_allclose(served[CHUNK + i], expected, rtol=1e-5)
        for i in range(k, CHUNK):
            np.testing.assert_array_equal(served[CHUNK + i], _raw_row(1, i))

    def test_the_first_chunk_after_construction_is_not_blended(self) -> None:
        served = _serve(_sync(4), CHUNK)
        for i in range(CHUNK):
            np.testing.assert_array_equal(served[i], _raw_row(0, i))

    def test_dims_outside_the_mask_are_untouched(self) -> None:
        k, dims = 4, (0, 2)
        served = _serve(_sync(k, dims), 2 * CHUNK)
        raw_head = _raw_row(1, 0)
        for d in (1, 3):  # e.g. a gripper or a delta-action dim
            for i in range(CHUNK):
                assert served[CHUNK + i][d] == _raw_row(1, i)[d]
        for d in dims:
            assert served[CHUNK][d] != raw_head[d]

    def test_reset_clears_the_tail(self) -> None:
        producer = _JumpingProducer()
        ex = _sync(3, chunk_fn=producer)
        _serve(ex, CHUNK)
        ex.reset()
        # Chunk 1 is now the first of a new episode: served raw, no carry-over.
        served = _serve(ex, CHUNK)
        for i in range(CHUNK):
            np.testing.assert_array_equal(served[i], _raw_row(1, i))

    def test_torch_actions_blend_like_numpy(self) -> None:
        k = 2
        ex = _sync(k, chunk_fn=_JumpingProducer(as_torch=True))
        ex.start()
        served = [ex.select_action({"obs": 0}) for _ in range(2 * CHUNK)]
        ex.stop()
        assert all(isinstance(a, torch.Tensor) and a.shape == (1, DOF) for a in served)
        tail, head = served[CHUNK - 1], served[CHUNK]
        raw_jump = torch.from_numpy(_raw_row(1, 0)).unsqueeze(0) - tail
        torch.testing.assert_close(head - tail, raw_jump / (k + 1))

    # prefetch_at 3 on a chunk of 8: a merged chunk keeps 8 - 3 - 1 = 4 rows, so K = 3
    # is the longest blend that still ends on a policy row.
    @pytest.mark.parametrize("k", [2, 3])
    def test_prefetch_path_blends_only_the_first_k_rows_after_each_boundary(self, k: int) -> None:
        """Background pre-fetch, with the stale-row drop: the blend is applied to the
        rows that are actually served, starting at the first current one, and finishes
        inside every merged chunk."""
        prefetch_at = 3

        def run(steps: int) -> list[NDArray[np.float32]]:
            calls = [0]

            def step_clock(payload: dict[str, int]) -> list[NDArray[np.float32]]:
                # Action i targets step k + i, plus an offset each re-plan shifts by 0.5.
                offset = 0.5 * calls[0]
                calls[0] += 1
                return [
                    np.full(DOF, SLOPE * (payload["step"] + i) + offset, np.float32)
                    for i in range(CHUNK)
                ]

            ex = ChunkedExecutor(
                chunk_fn=step_clock,
                chunk_size=CHUNK,
                prefetch_at=prefetch_at,
                boundary_blend_steps=steps,
            )
            ex.start()
            out = [ex.select_action(functools.partial(dict, step=s)) for s in range(4 * CHUNK)]
            ex.stop()
            return out

        raw, blended = run(0), run(k)
        changed = [s for s in range(len(raw)) if not np.array_equal(raw[s], blended[s])]
        assert changed, "the pre-fetch boundaries carried a jump and must have been blended"
        for s in changed:
            assert s > 0
        steps_raw = [float(np.max(np.abs(raw[s] - raw[s - 1]))) for s in changed]
        steps_blended = [float(np.max(np.abs(blended[s] - blended[s - 1]))) for s in changed]
        assert max(steps_blended) < max(steps_raw)
        # Every blended stretch is at most K rows long.
        longest = current = 1
        for a, b in itertools.pairwise(changed):
            current = current + 1 if b == a + 1 else 1
            longest = max(longest, current)
        # Stretches never run into the next boundary: each merged chunk ends on policy rows.
        assert longest <= k

    def test_bfloat16_torch_actions_blend_and_log(self) -> None:
        """bf16 rows (a pi05 / GR00T dtype) blend; the DEBUG jump norm must not go
        through NumPy, which has no bfloat16."""
        k = 2
        bf16 = _JumpingProducer(as_torch=True)
        ex = _sync(k, (0, 1), chunk_fn=lambda p: [r.to(torch.bfloat16) for r in bf16(p)])
        ex.start()
        served = [ex.select_action({"obs": 0}) for _ in range(2 * CHUNK)]
        ex.stop()
        assert all(a.dtype == torch.bfloat16 for a in served)
        tail, head = served[CHUNK - 1].float(), served[CHUNK].float()
        raw_head = torch.from_numpy(_raw_row(1, 0)).unsqueeze(0).to(torch.bfloat16).float()
        torch.testing.assert_close(
            head[..., :2] - tail[..., :2], (raw_head - tail)[..., :2] / (k + 1), atol=2e-2, rtol=0
        )
        torch.testing.assert_close(head[..., 2:], raw_head[..., 2:])

    def test_non_finite_values_in_unblended_dims_stay_as_produced(self) -> None:
        """A dim outside ``blend_dims`` carrying inf (e.g. an unused slot) must stay inf,
        not become NaN through the masked delta."""
        inner = _JumpingProducer()

        def with_inf(payload: Any) -> list[NDArray[np.float32]]:
            rows = inner(payload)
            for r in rows:
                r[3] = np.inf
            return rows

        served = _serve(_sync(3, (0, 1, 2), chunk_fn=with_inf), 2 * CHUNK)
        for a in served:
            assert np.isposinf(a[3])
            assert np.all(np.isfinite(a[:3]))
        assert served[CHUNK][0] != _raw_row(1, 0)[0]  # the blend still ran on dims 0..2

    def test_a_caller_mutating_its_action_does_not_move_the_tail(self) -> None:
        """The stored tail is a copy: in-place edits of a returned action (clipping,
        a hand-off buffer) must not change what the next chunk is stitched to."""
        k = 3
        reference = _serve(_sync(k), 2 * CHUNK)
        ex = _sync(k)
        ex.start()
        served = []
        for _ in range(2 * CHUNK):
            action = ex.select_action({"obs": 0})
            served.append(np.array(action, copy=True))
            action[...] = 1e3  # the caller scribbles over what it was handed
        ex.stop()
        for got, want in zip(served, reference, strict=True):
            np.testing.assert_array_equal(got, want)


class TestBoundaryBlendConfig:
    @pytest.mark.parametrize("steps", [-1, True, 1.5, CHUNK, CHUNK + 1])
    def test_bad_steps_are_rejected(self, steps: Any) -> None:
        with pytest.raises(ROSConfigError, match="boundary_blend_steps"):
            ChunkedExecutor(
                chunk_fn=_JumpingProducer(), chunk_size=CHUNK, boundary_blend_steps=steps
            )

    @pytest.mark.parametrize("dims", [(), (-1,), (0, 0), (True,), (1.0,)])
    def test_bad_dims_are_rejected(self, dims: Any) -> None:
        with pytest.raises(ROSConfigError, match="blend_dims"):
            ChunkedExecutor(
                chunk_fn=_JumpingProducer(),
                chunk_size=CHUNK,
                boundary_blend_steps=2,
                blend_dims=dims,
            )

    def test_dims_without_steps_are_rejected(self) -> None:
        with pytest.raises(ROSConfigError, match="blend_dims"):
            ChunkedExecutor(chunk_fn=_JumpingProducer(), chunk_size=CHUNK, blend_dims=(0,))

    def test_dim_beyond_the_action_is_rejected_before_any_action_is_served(self) -> None:
        ex = _sync(2, (0, DOF))
        ex.start()
        with pytest.raises(ROSConfigError, match="blend_dims"):
            ex.select_action({"obs": 0})
        ex.stop()

    @pytest.mark.parametrize(
        ("prefetch_at", "longest_ok"),
        [
            (3, CHUNK - 3 - 2),  # a merged chunk keeps CHUNK - 3 - 1 = 4 rows -> K <= 3
            (1, CHUNK - 1 - 2),
            (CHUNK - 1, CHUNK - 1),  # every merge is fully stale -> synchronous re-plans
        ],
    )
    def test_steps_are_bounded_by_the_rows_a_prefetched_chunk_keeps(
        self, prefetch_at: int, longest_ok: int
    ) -> None:
        ChunkedExecutor(
            chunk_fn=_JumpingProducer(),
            chunk_size=CHUNK,
            prefetch_at=prefetch_at,
            boundary_blend_steps=longest_ok,
        )
        with pytest.raises(ROSConfigError, match="boundary_blend_steps"):
            ChunkedExecutor(
                chunk_fn=_JumpingProducer(),
                chunk_size=CHUNK,
                prefetch_at=prefetch_at,
                boundary_blend_steps=longest_ok + 1,
            )

    def test_rtc_and_blending_are_mutually_exclusive(self) -> None:
        rtc = pytest.importorskip("lerobot.policies.rtc")
        with pytest.raises(ROSConfigError, match="RTC"):
            ChunkedExecutor(
                chunk_fn=_JumpingProducer(),
                chunk_size=CHUNK,
                prefetch_at=4,
                rtc_config=rtc.RTCConfig(enabled=True, execution_horizon=4),
                boundary_blend_steps=2,
            )


class TestBuildChunkExecutorWiring:
    def test_policy_extras_block_reaches_the_executor(self) -> None:
        ex = build_chunk_executor(
            {"chunk_boundary_blend": {"steps": 3, "dims": [0, 1, 2]}},
            chunk_fn=_JumpingProducer(),
            chunk_size=CHUNK,
        )
        assert ex is not None
        served = [np.asarray(ex.select_action({"obs": 0})) for _ in range(CHUNK + 1)]
        ex.stop()
        np.testing.assert_allclose(
            served[CHUNK][:3] - served[CHUNK - 1][:3],
            (_raw_row(1, 0) - served[CHUNK - 1])[:3] / 4,
            rtol=1e-5,
        )
        assert served[CHUNK][3] == _raw_row(1, 0)[3]

    def test_no_block_means_no_blend(self) -> None:
        ex = build_chunk_executor({}, chunk_fn=_JumpingProducer(), chunk_size=CHUNK)
        assert ex is not None
        served = [np.asarray(ex.select_action({"obs": 0})) for _ in range(CHUNK + 1)]
        ex.stop()
        np.testing.assert_array_equal(served[CHUNK], _raw_row(1, 0))

    @pytest.mark.parametrize(
        "block",
        [
            5,
            {"steps": 0},
            {"steps": -2},
            {"steps": True},
            {"steps": CHUNK},
            {"steps": CHUNK + 1},
            {"dims": [0]},
            {"steps": 2, "dims": [0, 0]},
            {"steps": 2, "dims": []},
            {"steps": 2, "dims": "0,1"},
            {"steps": 2, "dims": [0, -1]},
            {"steps": 2, "mask": [0]},
        ],
    )
    def test_malformed_blocks_are_rejected(self, block: Any) -> None:
        with pytest.raises(ROSConfigError, match="chunk_boundary_blend"):
            build_chunk_executor(
                {"chunk_boundary_blend": block}, chunk_fn=_JumpingProducer(), chunk_size=CHUNK
            )

    @pytest.mark.parametrize("block", [{"steps": 2}, {"steps": 2, "dims": None}])
    def test_manifest_block_requires_explicit_dims(self, block: dict[str, Any]) -> None:
        """No all-dims default in a manifest: a binarized gripper or a delta-action dim
        would be faded into commands the policy never emitted."""
        with pytest.raises(ROSConfigError, match=r"chunk_boundary_blend\.dims"):
            build_chunk_executor(
                {"chunk_boundary_blend": block}, chunk_fn=_JumpingProducer(), chunk_size=CHUNK
            )

    def test_prefetch_drop_bounds_steps_in_policy_extras(self) -> None:
        extras: dict[str, Any] = {"chunk_prefetch": True, "chunk_prefetch_at": 3}
        ok = build_chunk_executor(
            {**extras, "chunk_boundary_blend": {"steps": 3, "dims": [0]}},
            chunk_fn=_JumpingProducer(),
            chunk_size=CHUNK,
        )
        assert ok is not None
        ok.stop()
        with pytest.raises(ROSConfigError, match=r"chunk_boundary_blend\.steps.*stale-row drop"):
            build_chunk_executor(
                {**extras, "chunk_boundary_blend": {"steps": 4, "dims": [0]}},
                chunk_fn=_JumpingProducer(),
                chunk_size=CHUNK,
            )

    @pytest.mark.parametrize(
        ("steps", "dims", "prefetch_at"),
        [
            (1, None, 0),
            (CHUNK - 1, None, 0),
            (CHUNK, None, 0),
            (3, None, 3),
            (4, None, 3),
            (CHUNK - 1, None, CHUNK - 1),
            (2, [0, 2], 0),
            (2, [0, 0], 0),
            (2, [-1], 0),
            (2, [], 2),
        ],
    )
    def test_parser_and_executor_apply_the_same_rules(
        self, steps: int, dims: list[int] | None, prefetch_at: int
    ) -> None:
        """One rule set: a block is accepted by ``build_chunk_executor`` exactly when
        the equivalent ``ChunkedExecutor`` arguments are."""

        def outcome(build: Any) -> bool:
            try:
                ex = build()
            except ROSConfigError:
                return False
            ex.stop()
            return True

        # A manifest has no all-dims default: ``dims=None`` is spelled out as every index.
        block: dict[str, Any] = {"steps": steps, "dims": list(range(DOF)) if dims is None else dims}
        extras: dict[str, Any] = {"chunk_boundary_blend": block}
        if prefetch_at:
            extras |= {"chunk_prefetch": True, "chunk_prefetch_at": prefetch_at}
        via_extras = outcome(
            lambda: build_chunk_executor(extras, chunk_fn=_JumpingProducer(), chunk_size=CHUNK)
        )
        direct = outcome(
            lambda: ChunkedExecutor(
                chunk_fn=_JumpingProducer(),
                chunk_size=CHUNK,
                prefetch_at=prefetch_at,
                boundary_blend_steps=steps,
                blend_dims=dims,
            )
        )
        assert via_extras == direct

    def test_single_step_policy_refuses_a_blend(self) -> None:
        class _OneStep:
            class config:  # noqa: N801 - mirrors lerobot's policy.config attribute
                n_action_steps = 1

            def predict_action_chunk(self, batch: Any) -> Any:
                return torch.zeros(1, 1, DOF)

        with pytest.raises(ROSConfigError, match="chunk_boundary_blend"):
            build_chunk_executor(
                {"chunk_boundary_blend": {"steps": 1, "dims": [0]}}, policy=_OneStep()
            )

    def test_rtc_block_and_blend_are_refused_together(self) -> None:
        pytest.importorskip("lerobot.policies.rtc")
        with pytest.raises(ROSConfigError, match="chunk_boundary_blend"):
            build_chunk_executor(
                {
                    "chunk_prefetch": True,
                    "rtc": {"enabled": True, "execution_horizon": 4},
                    "chunk_boundary_blend": {"steps": 2, "dims": [0]},
                },
                chunk_fn=_JumpingProducer(),
                chunk_size=CHUNK,
                adapter_name="smolvla",
            )


_finite = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, width=32)


@settings(max_examples=60, deadline=None)
@given(
    tail=st.lists(_finite, min_size=DOF, max_size=DOF),
    head=st.lists(_finite, min_size=DOF, max_size=DOF),
    k=st.integers(min_value=1, max_value=CHUNK - 1),
)
def test_blended_boundary_step_never_exceeds_the_raw_jump(
    tail: list[float], head: list[float], k: int
) -> None:
    """Per dim, the step tail -> served row 0 is at most the raw jump tail -> chunk row 0,
    and rows from K on are exactly the policy's."""
    chunks = [
        [np.asarray(tail, np.float32)] * CHUNK,
        [np.asarray(head, np.float32) + np.float32(i * SLOPE) for i in range(CHUNK)],
    ]
    calls = iter(chunks)
    ex = ChunkedExecutor(
        chunk_fn=lambda _: next(calls), chunk_size=CHUNK, prefetch_at=0, boundary_blend_steps=k
    )
    served = _serve(ex, 2 * CHUNK)
    raw_jump = np.abs(chunks[1][0] - chunks[0][-1])
    blended_step = np.abs(served[CHUNK] - served[CHUNK - 1])
    assert np.all(blended_step <= raw_jump + 1e-5)
    for i in range(k, CHUNK):
        np.testing.assert_array_equal(served[CHUNK + i], chunks[1][i])
