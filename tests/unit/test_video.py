from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from openral_sim._video import _annotate_panel


def test_annotate_panel_preserves_shape_and_writes_banner() -> None:
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    out = _annotate_panel(frame, "world render (not a camera)")
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert int(out[:24].sum()) > 0


def test_cursor_lands_where_matplotlib_draws_it() -> None:
    """The numpy cursor sits on the column matplotlib's own axvline would use."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import numpy as np
    from openral_sim._video import _CANVAS_W, _PLOT_H, _JointPlotRenderer

    states = np.stack([np.sin(np.linspace(0, 6, 300)), np.cos(np.linspace(0, 6, 300))], axis=1)
    r = _JointPlotRenderer(states.astype(np.float32), width=_CANVAS_W, height=_PLOT_H, title="t")
    for step in (75, 150, 299):  # step 0 sits under the axis spine
        # Reference: let matplotlib draw a red line at this step on the same figure.
        line = r._ax.axvline(step, color="red", linewidth=1.2)
        r._fig.canvas.draw()
        ref = r._snapshot()
        line.remove()
        ours = r.render_at_step(step)

        def changed_cols(img: NDArray[np.uint8]) -> set[int]:
            diff = np.abs(img.astype(int) - r._static_rgb.astype(int)).sum(axis=2)
            return set(np.nonzero(diff[r._rows].sum(axis=0))[0].tolist())

        ref_cols, our_cols = changed_cols(ref), changed_cols(ours)
        assert ref_cols and ref_cols <= our_cols, (step, sorted(ref_cols), sorted(our_cols))
        assert len(our_cols) <= 2 * 1 + 1
    # The background raster is never modified in place.
    r._fig.canvas.draw()
    assert np.array_equal(r._static_rgb, r._snapshot())


def test_a_60_hz_episode_is_written_at_30_fps_in_real_time(tmp_path: Path) -> None:
    """A 60 Hz recording plays at 30 fps with half the frames: same duration."""
    import imageio.v2 as iio
    import numpy as np
    from openral_sim._video import save_episode_mp4
    from openral_sim.rollout import EpisodeResult

    n = 120  # 2 s at 60 Hz
    frame = np.full((48, 64, 3), 90, dtype=np.uint8)
    result = EpisodeResult(
        steps=n,
        frames=[frame] * n,
        joint_positions=[np.full(3, i / n, dtype=np.float32) for i in range(n)],
    )
    out = save_episode_mp4(result, tmp_path / "ep.mp4", fps=60)
    reader = iio.get_reader(str(out))
    try:
        meta = reader.get_meta_data()
        count = reader.count_frames()  # type: ignore[attr-defined]  # reason: the ffmpeg plugin's reader has it, imageio's Reader type does not
    finally:
        reader.close()
    assert meta["fps"] == 30
    assert count == n // 2
